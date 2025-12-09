#pragma once

#include "tensor.h"

struct LayerNormQuant{
    Tensor_Quant<1> bias;
    Tensor_Quant<1> weight;

    void apply(Tensor_Quant<1> &out, const Tensor_Quant<1> &in);
};

struct MLPBlockQuant {
  Tensor_Quant<1> c_fc_bias;
  Tensor_Quant<2> c_fc_weight;
  Tensor_Quant<1> c_proj_bias;
  Tensor_Quant<2> c_proj_weight;

  void apply(const Tensor_Quant<1> &out, const Tensor_Quant<1> &in);
};

struct CausalSelfAttentionQuant {
  int num_heads;
  Tensor_Quant<1> c_attn_bias;
  Tensor_Quant<2> c_attn_weight;
  Tensor_Quant<1> c_proj_bias;
  Tensor_Quant<2> c_proj_weight;

  void apply(const Tensor_Quant<1> &out, const Tensor_Quant<1> &xbuf,
                                int pos, const Tensor_Quant<2> &kvbuf);
};


struct TransformerBlockQuant {
  CausalSelfAttentionQuant attn;
  LayerNormQuant ln_1, ln_2;
  MLPBlockQuant mlp;

  void normalize();
  void apply(const Tensor_Quant<1> &x, int i, const Tensor_Quant<2> &kvbuf);
};

struct ModelQuant {
  int embedding_dim;
  int num_tokens;
  int context_len;
  int ntokens;

  char *mmap_data;
  size_t mmap_siz;

  Tensor_Quant<2> wte_weight;
  Tensor_Quant<2> wpe_weight;
  LayerNormQuant ln_f;

  TransformerBlockQuant *h;

  ModelQuant() {
    h = NULL;
    mmap_data = NULL;
  }

  ~ModelQuant();


  void apply_transformer(int token_id, int input_pos, const Tensor_Quant<3> &kvbuf,
             const Tensor_Quant<1> &emb_out);

  void apply_lm_head(Tensor_Quant<1> &emb_in, Tensor_Quant<1> &logits);
};