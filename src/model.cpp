#include <sys/mman.h>
#include <assert.h>
#include <arm_neon.h>
#include <vector>
#include <cmath>
#include <algorithm>

#include "model.h"
#include "ops.h"
#include "quantize.h"

Model::~Model(){
    delete[] h;
    if (mmap_data) {
        munmap(mmap_data, mmap_siz);
    }
}

void LayerNorm::apply(Tensor<1> &out, const Tensor<1> &in){
  float *data_ptr = in.data;
  float sum_ele = 0.0f;
  float sum_sq = 0.0f;

  int n = in.shape[0];
  const int simd_size = 4;
  int simd_end = (n / simd_size) * simd_size;

  for(int i = 0; i < simd_end; i += simd_size){
    float32x4_t val = vld1q_f32(data_ptr + i);
    sum_ele += vaddvq_f32(val);
    sum_sq += vaddvq_f32(vmulq_f32(val, val));
  };

  //remaining stuff
  for(int i = simd_end; i < n; i++){
    float v = data_ptr[i];
    sum_ele += v;
    sum_sq += v * v;
  }

  float mean = sum_ele / n;
  float variance = sum_sq / n - mean * mean; // var = E[x2]−(E[x])2
  const float eps = 1e-5;  // maybe add to a config?
  float invstddev = 1.0 / sqrt(variance + eps);

  float *w = weight.data;
  float *b = bias.data;
  float *o = out.data;

  float32x4_t mean_vec = vdupq_n_f32(mean);
  float32x4_t invstd_vec = vdupq_n_f32(invstddev);
  
  for (int j = 0; j < simd_end; j += simd_size) {
      float32x4_t x = vld1q_f32(data_ptr + j);
      float32x4_t weight_vec = vld1q_f32(w + j);
      float32x4_t bias_vec = vld1q_f32(b + j);
      
      float32x4_t normalized = vsubq_f32(x, mean_vec);
      normalized = vmulq_f32(normalized, invstd_vec);
      normalized = vmulq_f32(normalized, weight_vec);
      normalized = vaddq_f32(normalized, bias_vec);
      
      vst1q_f32(o + j, normalized);
  }
}

void MLPBlock::apply(const Tensor<1> &out, const Tensor<1> &in, QuantizationType qtype) {
    const int emb_dim = 768;
    const int hidden_dim = 4 * emb_dim;

    assert(in.shape[0] == emb_dim);
    assert(c_fc_bias.shape[0] == hidden_dim);
    assert(c_proj_bias.shape[0] == emb_dim);

    Tensor<1> hbuf(hidden_dim);

    // first linear: h = GELU( W_fc * x + b_fc )
    if (qtype == QuantizationType::FP32) {
        for (int j = 0; j < hidden_dim; j++) {
            float y = c_fc_bias.data[j];
            const float* w_row = &c_fc_weight.data[j * emb_dim]; 
            y += sdot_simd(in.data, w_row, emb_dim);
            float gelu = y / (1.0f + expf(-1.702f * y));
            hbuf.data[j] = gelu;
        }
    } else {
        if (qtype == QuantizationType::INT8) {
            qmatmul_int8(in, q_c_fc_weight, hbuf);
        } else {
            qmatmul_int4(in, q_c_fc_weight, hbuf);
        }
        
        for (int j = 0; j < hidden_dim; j++) {
            float y = hbuf.data[j] + c_fc_bias.data[j];
            float gelu = y / (1.0f + expf(-1.702f * y));
            hbuf.data[j] = gelu;
        }
    }

    // 2. Projection: out += W_proj * h + b_proj
    if (qtype == QuantizationType::FP32) {
        for (int j = 0; j < emb_dim; j++) {
            float sum = c_proj_bias.data[j];
            const float* w_row = &c_proj_weight.data[j * hidden_dim]; 
            sum += sdot_simd(hbuf.data, w_row, hidden_dim);
            out.data[j] += sum;
        }
    } else {
        Tensor<1> proj_out(emb_dim);
        if (qtype == QuantizationType::INT8) {
            qmatmul_int8(hbuf, q_c_proj_weight, proj_out);
        } else {
            qmatmul_int4(hbuf, q_c_proj_weight, proj_out);
        }
        
        for (int j = 0; j < emb_dim; j++) {
            out.data[j] += proj_out.data[j] + c_proj_bias.data[j];
        }
    }
}

void Model::apply_lm_head(Tensor<1> &emb_in, Tensor<1> &logits) {
  assert(emb_in.shape[0] == embedding_dim);
  // layernorm and dot with embedding matrix
  ln_f.apply(emb_in, emb_in);
  const int ntokens = logits.shape[0];
  
  if (qtype == QuantizationType::FP32) {
      float *w = wte_weight.data; // (50257, 768)
      float m = -INFINITY;
      for (int j = 0; j < ntokens; j++) {
        logits[j] = sdot_simd(emb_in.data, w, embedding_dim);
        if (logits[j] > m) {
          m = logits[j];
        }
        w += embedding_dim;
      }
  } else {
      if (qtype == QuantizationType::INT8) {
          qmatmul_int8(emb_in, q_wte_weight, logits);
      } else {
          qmatmul_int4(emb_in, q_wte_weight, logits);
      }
      // Find max for numerical stability
      float m = -INFINITY;
      for (int j = 0; j < ntokens; j++) {
          if (logits[j] > m) m = logits[j];
      }
      // Subtract max done below
  }

  // subtract max for numerical stability
  // Note: m is not propagated from else block if I declared it inside.
  // Let's fix this.
  float m = -INFINITY;
  for (int j = 0; j < ntokens; j++) {
      if (logits[j] > m) m = logits[j];
  }
  
  for (int j = 0; j < ntokens; j++) {
    logits[j] -= m;
  }
}

void CausalSelfAttention::apply(const Tensor<1> &out, const Tensor<1> &xbuf,
                                int pos, const Tensor<2> &kvbuf, QuantizationType qtype) {
    const int emb_dim = 768;
    const int num_heads = 12;
    const int head_dim = emb_dim / num_heads;  // 64
    
    assert(head_dim * num_heads == emb_dim); 
    assert(xbuf.shape[0] == emb_dim);

    Tensor<1> query_buf(emb_dim);      // q vector
    Tensor<1> attn_out_buf(emb_dim);   // attn output
    
    // Compute Q, K, V
    // In FP32, it was done in a loop.
    // In Quantized, we can use qmatmul if we had a single weight matrix.
    // But c_attn_weight is (3*emb_dim, emb_dim).
    // So qmatmul will produce (3*emb_dim) output.
    
    if (qtype == QuantizationType::FP32) {
        float* weight_ptr = c_attn_weight.data;  // [2304, 768]
        float* bias_ptr = c_attn_bias.data;      // [2304]
        const float* input_ptr = xbuf.data;      // [768]
        
        // compute q
        for (int d = 0; d < emb_dim; d++) {
            query_buf[d] = bias_ptr[d] + sdot_simd(input_ptr, weight_ptr, emb_dim);
            weight_ptr += emb_dim;
        }
        bias_ptr += emb_dim;
        
        float* kv_cache_ptr = &kvbuf(pos, 0);
        for (int kv_idx = 0; kv_idx < 2 * emb_dim; kv_idx++) {
            kv_cache_ptr[kv_idx] = bias_ptr[kv_idx] + sdot_simd(input_ptr, weight_ptr, emb_dim);
            weight_ptr += emb_dim;
        }
    } else {
        Tensor<1> qkv_out(3 * emb_dim);
        if (qtype == QuantizationType::INT8) {
            qmatmul_int8(xbuf, q_c_attn_weight, qkv_out);
        } else {
            qmatmul_int4(xbuf, q_c_attn_weight, qkv_out);
        }
        
        // Distribute to query_buf and kvbuf
        float* qkv_ptr = qkv_out.data;
        float* bias_ptr = c_attn_bias.data;
        
        for (int d = 0; d < emb_dim; d++) {
            query_buf[d] = qkv_ptr[d] + bias_ptr[d];
        }
        
        float* kv_cache_ptr = &kvbuf(pos, 0);
        for (int kv_idx = 0; kv_idx < 2 * emb_dim; kv_idx++) {
            kv_cache_ptr[kv_idx] = qkv_ptr[emb_dim + kv_idx] + bias_ptr[emb_dim + kv_idx];
        }
    }
    
    std::fill(attn_out_buf.data, attn_out_buf.data + emb_dim, 0.0f);
    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    static std::vector<float> attention_scores;
    attention_scores.resize(pos + 1);
    
    for (int head = 0; head < num_heads; head++) {
        const int head_offset = head * head_dim;
        float* query_head = query_buf.data + head_offset;
        float* output_head = attn_out_buf.data + head_offset;
        
        float max_score = -INFINITY;
        for (int prev_pos = 0; prev_pos <= pos; prev_pos++) {
            const float* key_head = kvbuf.data + prev_pos * kvbuf.shape[1] + head_offset;
            float score = sdot_simd(query_head, key_head, head_dim) * attn_scale;
            
            attention_scores[prev_pos] = score;
            max_score = std::max(max_score, score);
        }
        
        // softmax
        float sum_exp = 0.0f;
        for (int prev_pos = 0; prev_pos <= pos; prev_pos++) {
            float exp_score = std::exp(attention_scores[prev_pos] - max_score);
            attention_scores[prev_pos] = exp_score;
            sum_exp += exp_score;
        }
        
        const float inv_sum_exp = 1.0f / sum_exp;
        for (int prev_pos = 0; prev_pos <= pos; prev_pos++) {
            attention_scores[prev_pos] *= inv_sum_exp;
        }
        
        // weighted sum of value vectors
        for (int prev_pos = 0; prev_pos <= pos; prev_pos++) {
            const float attention_weight = attention_scores[prev_pos];
            const float* value_head = kvbuf.data + prev_pos * kvbuf.shape[1] + emb_dim + head_offset;
            
            for (int d = 0; d < head_dim; d++) {
                output_head[d] += attention_weight * value_head[d];
            }
        }
    }
    
    // final proj layer
    if (qtype == QuantizationType::FP32) {
        float* weight_ptr = c_proj_weight.data;
        for (int d = 0; d < emb_dim; d++) {
            out.data[d] += c_proj_bias[d] + sdot_simd(attn_out_buf.data, weight_ptr, emb_dim);
            weight_ptr += emb_dim;
        }
    } else {
        Tensor<1> proj_out(emb_dim);
        if (qtype == QuantizationType::INT8) {
            qmatmul_int8(attn_out_buf, q_c_proj_weight, proj_out);
        } else {
            qmatmul_int4(attn_out_buf, q_c_proj_weight, proj_out);
        }
        
        for (int d = 0; d < emb_dim; d++) {
            out.data[d] += proj_out.data[d] + c_proj_bias.data[d];
        }
    }
}

void TransformerBlock::apply(const Tensor<1> &x, int i, const Tensor<2> &kvbuf, QuantizationType qtype) {
    Tensor<1> xbuf(x.shape[0]);

    ln_1.apply(xbuf, x);
    attn.apply(x, xbuf, i, kvbuf, qtype);
    ln_2.apply(xbuf, x);
    mlp.apply(x, xbuf, qtype);
  }

void Model::apply_transformer(int token_id, int input_pos,
                              const Tensor<3> &kvbuf,
                              const Tensor<1> &emb_out) {
  for (int k = 0; k < embedding_dim; k++) {
    emb_out[k] = wte_weight(token_id, k) + wpe_weight(input_pos, k);
  }
  for (int layer = 0; layer < 12; layer++) {
    h[layer].apply(emb_out, input_pos, kvbuf.slice(layer), qtype); 
  }
}