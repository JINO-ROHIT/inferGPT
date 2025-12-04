#pragma once

#include "tensor.h"
#include "quantize.h"

struct LayerNorm{
    Tensor<1> bias;
    Tensor<1> weight;

    void apply(Tensor<1> &out, const Tensor<1> &in);
};

struct MLPBlock {
  Tensor<1> c_fc_bias;
  Tensor<2> c_fc_weight;
  Tensor<1> c_proj_bias;
  Tensor<2> c_proj_weight;

  // Quantized weights
  QuantizedTensor q_c_fc_weight;
  QuantizedTensor q_c_proj_weight;

  void apply(const Tensor<1> &out, const Tensor<1> &in, QuantizationType qtype);
};

struct CausalSelfAttention {
  int num_heads;
  Tensor<1> c_attn_bias;
  Tensor<2> c_attn_weight;
  Tensor<1> c_proj_bias;
  Tensor<2> c_proj_weight;

  // Quantized weights
  QuantizedTensor q_c_attn_weight;
  QuantizedTensor q_c_proj_weight;

  void apply(const Tensor<1> &out, const Tensor<1> &xbuf, int i, const Tensor<2> &kvbuf, QuantizationType qtype);
};


struct TransformerBlock {
  CausalSelfAttention attn;
  LayerNorm ln_1, ln_2;
  MLPBlock mlp;

  void normalize();
  void apply(const Tensor<1> &x, int i, const Tensor<2> &kvbuf, QuantizationType qtype);
};

struct Model {
  int embedding_dim;
  int num_tokens;
  int context_len;
  int ntokens;

  char *mmap_data;
  size_t mmap_siz;

  Tensor<2> wte_weight;
  Tensor<2> wpe_weight;
  
  // Quantized weights
  QuantizedTensor q_wte_weight;
  
  LayerNorm ln_f;

  TransformerBlock *h;
  
  QuantizationType qtype;

  Model() {
    h = NULL;
    mmap_data = NULL;
    qtype = QuantizationType::FP32;
  }

  ~Model();


  void apply_transformer(int token_id, int input_pos, const Tensor<3> &kvbuf,
             const Tensor<1> &emb_out);

  void apply_lm_head(Tensor<1> &emb_in, Tensor<1> &logits);
};