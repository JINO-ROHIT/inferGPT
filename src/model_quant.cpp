#include <sys/mman.h>
#include <assert.h>
#include <arm_neon.h>

#include "model_quant.h"
#include "ops.h"

ModelQuant::~ModelQuant(){
    delete[] h;
    if (mmap_data) {
        munmap(mmap_data, mmap_siz);
    }
}

void LayerNormQuant::apply(Tensor_Quant<1> &out, const Tensor_Quant<1> &in){
    float sum_ele = 0.0f;
    float sum_sq = 0.0f;
    int n = in.shape[0];

    for(int i = 0; i < n ; i++){
        float v = in.get_dequantized(i);
        sum_ele += v;
        sum_sq += v * v;
    }

    float mean = sum_ele / n;
    float variance = sum_sq / n - mean * mean;
    const float eps = 1e-5f; 
    float invstddev = 1.0f / sqrtf(variance + eps);

    for (int j = 0; j < n; j++) {
        float x_norm = (in.get_dequantized(j) - mean) * invstddev;
        float w_val = weight.get_dequantized(j);
        float b_val = bias.get_dequantized(j);
        float out_val = x_norm * w_val + b_val;

        // TODO: this scale is an approximation
        float output_scale = 0.01f;
        out[j] = static_cast<int8_t>(roundf(out_val * output_scale));
        out.scale = output_scale;
    }
}

void MLPBlockQuant::apply(const Tensor_Quant<1> &out, const Tensor_Quant<1> &in) {
    const int emb_dim = 768;
    const int hidden_dim = 4 * emb_dim;

    assert(in.shape[0] == emb_dim);
    assert(c_fc_weight.shape[0] == hidden_dim);
    assert(c_fc_weight.shape[1] == emb_dim);
    assert(c_fc_bias.shape[0] == hidden_dim);
    assert(c_proj_weight.shape[0] == emb_dim);
    assert(c_proj_weight.shape[1] == hidden_dim);
    assert(c_proj_bias.shape[0] == emb_dim);

    Tensor_Quant<1> hbuf(hidden_dim);

    for (int j = 0; j < hidden_dim; j++) {
        float y = c_fc_bias.get_dequantized(j);
        const int8_t* w_row = &c_fc_weight(j, 0);

        float dot_product = sdot_simd(in.data, w_row, emb_dim);
        y += dot_product * in.scale * c_fc_weight.scale;

        float gelu = y / (1.0f + expf(-1.702f * y));

        hbuf[j] = static_cast<int8_t>(roundf(gelu * 0.01f)); // TODO: proper scale computation
        hbuf.scale = 0.01f; // TODO: compute proper scale
    }

    for (int j = 0; j < emb_dim; j++) {
        float sum = c_proj_bias.get_dequantized(j);
        const int8_t* w_row = &c_proj_weight(j, 0); 

        float dot_product = sdot_simd(hbuf.data, w_row, hidden_dim);
        sum += dot_product * hbuf.scale * c_proj_weight.scale;

        float current_out = out.get_dequantized(j);
        float new_out = current_out + sum;

        out[j] = static_cast<int8_t>(roundf(new_out * out.scale));
    }
}

void ModelQuant::apply_lm_head(Tensor_Quant<1> &emb_in, Tensor_Quant<1> &logits) {
  assert(emb_in.shape[0] == embedding_dim);

  ln_f.apply(emb_in, emb_in);
  const int ntokens = logits.shape[0];
  float m = -INFINITY;

  for (int j = 0; j < ntokens; j++) {
    const int8_t* w_row = &wte_weight(j, 0); // (50257, 768)
    float dot_product = sdot_simd(emb_in.data, w_row, embedding_dim);
    float logit_val = dot_product * emb_in.scale * wte_weight.scale;
    logits[j] = static_cast<int8_t>(roundf(logit_val * logits.scale));
    if (logit_val > m) {
      m = logit_val;
    }
  }

  for (int j = 0; j < ntokens; j++) {
    float current_logit = logits.get_dequantized(j);
    float new_logit = current_logit - m;
    logits[j] = static_cast<int8_t>(roundf(new_logit * logits.scale));
  }
}

void CausalSelfAttentionQuant::apply(const Tensor_Quant<1> &out, const Tensor_Quant<1> &xbuf,
                                int pos, const Tensor_Quant<2> &kvbuf) {
    const int emb_dim = 768;
    const int num_heads = 12;
    const int head_dim = emb_dim / num_heads;  

    assert(head_dim * num_heads == emb_dim);
    assert(xbuf.shape[0] == emb_dim);

    Tensor_Quant<1> query_buf(emb_dim);      
    Tensor_Quant<1> attn_out_buf(emb_dim);  

    for (int d = 0; d < emb_dim; d++) {
        float bias_val = c_attn_bias.get_dequantized(d);
        const int8_t* weight_row = &c_attn_weight(d, 0);
        float dot_product = sdot_simd(xbuf.data, weight_row, emb_dim);
        float q_val = bias_val + dot_product * xbuf.scale * c_attn_weight.scale;
        query_buf[d] = static_cast<int8_t>(roundf(q_val * query_buf.scale));
    }

    for (int kv_idx = 0; kv_idx < 2 * emb_dim; kv_idx++) {
        float bias_val = c_attn_bias.get_dequantized(emb_dim + kv_idx);
        const int8_t* weight_row = &c_attn_weight(emb_dim + kv_idx, 0);
        float dot_product = sdot_simd(xbuf.data, weight_row, emb_dim);
        float kv_val = bias_val + dot_product * xbuf.scale * c_attn_weight.scale;
        kvbuf(pos, kv_idx) = static_cast<int8_t>(roundf(kv_val * kvbuf.scale));
    }

    for (int i = 0; i < emb_dim; i++) {
        attn_out_buf[i] = 0;
    }
    attn_out_buf.scale = 0.01f; // TODO: compute proper scale

    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    static std::vector<float> attention_scores;
    attention_scores.resize(pos + 1);

    for (int head = 0; head < num_heads; head++) {
        const int head_offset = head * head_dim;

        float max_score = -INFINITY;
        for (int prev_pos = 0; prev_pos <= pos; prev_pos++) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                float q_val = query_buf.get_dequantized(head_offset + d);
                float k_val = kvbuf.get_dequantized(prev_pos, head_offset + d);
                score += q_val * k_val;
            }
            score *= attn_scale;

            attention_scores[prev_pos] = score;
            max_score = std::max(max_score, score);
        }


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

        for (int prev_pos = 0; prev_pos <= pos; prev_pos++) {
            const float attention_weight = attention_scores[prev_pos];

            for (int d = 0; d < head_dim; d++) {
                float v_val = kvbuf.get_dequantized(prev_pos, emb_dim + head_offset + d);
                float current_out = attn_out_buf.get_dequantized(head_offset + d);
                float new_out = current_out + attention_weight * v_val;
                attn_out_buf[head_offset + d] = static_cast<int8_t>(roundf(new_out * attn_out_buf.scale));
            }
        }
    }

    for (int d = 0; d < emb_dim; d++) {
        float bias_val = c_proj_bias.get_dequantized(d);
        const int8_t* weight_row = &c_proj_weight(d, 0);
        float dot_product = sdot_simd(attn_out_buf.data, weight_row, emb_dim);
        float proj_val = bias_val + dot_product * attn_out_buf.scale * c_proj_weight.scale;

        float current_out = out.get_dequantized(d);
        float new_out = current_out + proj_val;
        out[d] = static_cast<int8_t>(roundf(new_out * out.scale));
    }
}

void TransformerBlockQuant::apply(const Tensor_Quant<1> &x, int i, const Tensor_Quant<2> &kvbuf) {
    Tensor_Quant<1> xbuf(x.shape[0]);

    ln_1.apply(xbuf, x);
    attn.apply(x, xbuf, i, kvbuf);
    ln_2.apply(xbuf, x);
    mlp.apply(x, xbuf);
  }

void ModelQuant::apply_transformer(int token_id, int input_pos,
                              const Tensor_Quant<3> &kvbuf,
                              const Tensor_Quant<1> &emb_out) {
  for (int k = 0; k < embedding_dim; k++) {
    float wte_val = wte_weight.get_dequantized(token_id, k);
    float wpe_val = wpe_weight.get_dequantized(input_pos, k);
    float emb_val = wte_val + wpe_val;
    emb_out[k] = static_cast<int8_t>(roundf(emb_val * emb_out.scale)); 
  }

  for (int layer = 0; layer < 12; layer++) {
    h[layer].apply(emb_out, input_pos, kvbuf.slice(layer));
  }
}