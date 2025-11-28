#include <sys/mman.h>
#include <assert.h>

#include "model.h"
#include "ops.h"

Model::~Model(){
    delete[] h;
    if (mmap_data) {
        munmap(mmap_data, mmap_siz);
    }
}
//reference formula - https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
void LayerNorm::apply(Tensor<1> &out, const Tensor<1> &in){
    float sum_ele = 0.0f;
    float sum_sq = 0.0f;
    float *data_ptr = in.data;
    int n = in.shape[0];

    for(int i = 0; i < n ; i++){
        float v = data_ptr[i];
        sum_ele += v;
        sum_sq += v * v;
    }

    float mean = sum_ele / in.shape[0];
    float variance = sum_sq / in.shape[0] - mean * mean; // var = E[x2]−(E[x])2
    const float eps = 1e-5;  // maybe add to a config?
    float invstddev = 1.0 / sqrt(variance + eps);
    float *w = weight.data;
    float *b = bias.data;
    float *o = out.data;
    for (int j = 0; j < n; j++) {
        o[j] = (data_ptr[j] - mean) * invstddev * w[j] + b[j];
    }
}

void MLPBlock::apply(const Tensor<1> &out, const Tensor<1> &in) {
    const int emb_dim = 768;
    const int hidden_dim = 4 * emb_dim;

    assert(in.shape[0] == emb_dim);
    assert(c_fc_weight.shape[0] == hidden_dim);
    assert(c_fc_weight.shape[1] == emb_dim);
    assert(c_fc_bias.shape[0] == hidden_dim);
    assert(c_proj_weight.shape[0] == emb_dim);
    assert(c_proj_weight.shape[1] == hidden_dim);
    assert(c_proj_bias.shape[0] == emb_dim);

    Tensor<1> hbuf(hidden_dim);

    // first linear: h = GELU( W_fc * x + b_fc )

    for (int j = 0; j < hidden_dim; j++) {

        // dot product w[j] dot input
        float y = c_fc_bias.data[j];
        const float* w_row = &c_fc_weight.data[j * emb_dim]; // remember matrix rows are stored contiguously (3072, 768)

        y += sdot(in.data, w_row, emb_dim);

        // gelu approximation 
        float gelu = y / (1.0f + expf(-1.702f * y));

        hbuf.data[j] = gelu;
    }

    // 2. Projection: out += W_proj * h + b_proj
    for (int j = 0; j < emb_dim; j++) {

        float sum = c_proj_bias.data[j];
        const float* w_row = &c_proj_weight.data[j * hidden_dim]; // (768, 3072)

        sum += sdot(hbuf.data, w_row, hidden_dim);

        out.data[j] += sum;
    }
}

void Model::apply_lm_head(Tensor<1> &emb_in, Tensor<1> &logits) {
  assert(emb_in.shape[0] == embedding_dim);
  // layernorm and dot with embedding matrix
  ln_f.apply(emb_in, emb_in);
  const int ntokens = logits.shape[0];
  float *w = wte_weight.data; // (50257, 768)
  float m = -INFINITY;
  for (int j = 0; j < ntokens; j++) {
    logits[j] = sdot(emb_in.data, w, embedding_dim);
    if (logits[j] > m) {
      m = logits[j];
    }
    w += embedding_dim;
  }

  // subtract max for numerical stability
  for (int j = 0; j < ntokens; j++) {
    logits[j] -= m;
  }
}

void CausalSelfAttention::apply(const Tensor<1> &out, const Tensor<1> &xbuf,
                                int i, const Tensor<2> &kvbuf) {
  const int emb_siz = 768;
  const int num_heads = 12;
  const int head_siz = 64;

  assert(xbuf.shape[0] == emb_siz);
  assert(emb_siz / num_heads == head_siz);

  float attn_scale = 1.0 / sqrt(head_siz);

  // Buffer for query projection
  Tensor<1> qbuf(emb_siz);
  Tensor<1> ybuf(emb_siz);

  {
    float *w = c_attn_weight.data;
    float *x = xbuf.data;
    float *b = c_attn_bias.data;
    float *q = qbuf.data;

    // Compute Q = Qx + b_q
    for (int k = 0; k < emb_siz; k++) {
      *q++ = (*b++) + sdot(x, w, emb_siz);
      w += emb_siz;
    }

    // Compute K, V and cache them
    float *kv = &kvbuf(i, 0);
    for (int k = 0; k < 2 * emb_siz; k++) {
      *kv++ = (*b++) + sdot(x, w, emb_siz);
      w += emb_siz;
    }
  }

  {
    memset(ybuf.data, 0, emb_siz * sizeof(float));

    // Process each attention head
    for (int h = 0; h < num_heads; h++) {
      int head_offset = h * head_siz;
      float *q_head = qbuf.data + head_offset;
      float *y_head = ybuf.data + head_offset;

      // Compute attention scores for this head across all previous tokens
      float *scores = new float[i + 1];
      float max_score = -INFINITY;

      for (int j = 0; j <= i; j++) {
        float *k_head = kvbuf.data + j * kvbuf.shape[1] + head_offset;
        float score = sdot(q_head, k_head, head_siz) * attn_scale;
        scores[j] = score;
        if (score > max_score) {
          max_score = score;
        }
      }

      float sum_exp = 0.0f;
      for (int j = 0; j <= i; j++) {
        scores[j] = expf(scores[j] - max_score);
        sum_exp += scores[j];
      }
      for (int j = 0; j <= i; j++) {
        scores[j] /= sum_exp;
      }

      for (int j = 0; j <= i; j++) {
        float *v_head = kvbuf.data + j * kvbuf.shape[1] + emb_siz + head_offset;
        float weight = scores[j];
        for (int k = 0; k < head_siz; k++) {
          y_head[k] += weight * v_head[k];
        }
      }

      delete[] scores;
    }
  }

  // final projection: out += c_proj_weight @ ybuf + c_proj_bias
  {
    float *w = c_proj_weight.data;
    float *y = ybuf.data;
    float *o = out.data;
    for (int j = 0; j < emb_siz; j++) {
      *o++ += c_proj_bias[j] + sdot(y, w, emb_siz);
      w += emb_siz;
    }
  }
}

void TransformerBlock::apply(const Tensor<1> &x, int i, const Tensor<2> &kvbuf) {
    Tensor<1> xbuf(x.shape[0]);

    ln_1.apply(xbuf, x);
    attn.apply(x, xbuf, i, kvbuf);
    ln_2.apply(xbuf, x);
    mlp.apply(x, xbuf);
  }

void Model::apply_transformer(int token_id, int input_pos,
                              const Tensor<3> &kvbuf,
                              const Tensor<1> &emb_out) {
  for (int k = 0; k < embedding_dim; k++) {
    emb_out[k] = wte_weight(token_id, k) + wpe_weight(input_pos, k);
  }
  for (int layer = 0; layer < 12; layer++) {
    h[layer].apply(emb_out, input_pos, kvbuf.slice(layer)); // h is the transformer block basically
  }
}