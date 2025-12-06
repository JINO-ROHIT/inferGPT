#include <string>
#include <chrono>

#include "bpe.h"
#include "tensor.h"
#include "model.h"
#include "ops.h"

const int ctx_max = 1024;
extern bool load_gpt2_model(Model &m);

int generate_greedy(const char *prompt, int ntokens, Model &m, BPEEncoder &encoder, BPEDecoder &decoder, float temperature) {
    int ctx_tokens[ctx_max+1];

    int N;
    encoder.encode(prompt, ctx_tokens, ctx_max, &N);

    Tensor<3> kvbuf(12, ctx_max, 2*m.embedding_dim); // we have 12 layers
    Tensor<1> ybuf(m.embedding_dim);
    Tensor<1> logitbuf(m.ntokens);

    for (int j = 0; j < ntokens; j++) {
        m.apply_transformer(ctx_tokens[j], j, kvbuf, ybuf);

        if (j < N - 1) continue;  

        m.apply_lm_head(ybuf, logitbuf);

        int token = sample_greedy(logitbuf, temperature);
        ctx_tokens[j+1] = token;

        printf("%s", decoder.vocab[token].c_str());
        fflush(stdout); // to stream tokens on terminal
    }

    return ntokens;
};

int generate_temperature_scaling(const char *prompt, int ntokens, Model &m, BPEEncoder &encoder, BPEDecoder &decoder, float temperature){
    int ctx_tokens[ctx_max+1];

    int N;
    encoder.encode(prompt, ctx_tokens, ctx_max, &N);

    Tensor<3> kvbuf(12, ctx_max, 2*m.embedding_dim); 
    Tensor<1> ybuf(m.embedding_dim);
    Tensor<1> logitbuf(m.ntokens);

    for (int j = 0; j < ntokens; j++) {
        m.apply_transformer(ctx_tokens[j], j, kvbuf, ybuf);

        if (j < N - 1) continue;   

        m.apply_lm_head(ybuf, logitbuf);

        int token = temperature_sampling(logitbuf, temperature);
        ctx_tokens[j+1] = token;

        printf("%s", decoder.vocab[token].c_str());
        fflush(stdout); 
    }

    return ntokens;

}

// applies to the prompt given
float calculate_ppl(const char *prompt, int ntokens, Model &m, BPEEncoder &encoder, BPEDecoder &decoder, float temperature){
      int ctx_tokens[ctx_max+1];
      int N;
      float nll = 0.0; 

      encoder.encode(prompt, ctx_tokens, ctx_max, &N);

      Tensor<3> kvbuf(12, ctx_max, 2*m.embedding_dim);
      Tensor<1> ybuf(m.embedding_dim);
      Tensor<1> logitbuf(m.ntokens);

      for (int j = 0; j < N - 1; j++) { 
          m.apply_transformer(ctx_tokens[j], j, kvbuf, ybuf);
          m.apply_lm_head(ybuf, logitbuf);

          for (int i = 0; i < m.ntokens; i++) {
              logitbuf[i] /= temperature;
          }

          float max_logit = -INFINITY;
          for (int i = 0; i < m.ntokens; i++) {
              max_logit = std::max(max_logit, logitbuf[i]);
          }

          float sum_exp = 0.0;
          for (int i = 0; i < m.ntokens; i++) {
              sum_exp += std::exp(logitbuf[i] - max_logit);
          }

          int target_token = ctx_tokens[j + 1];
          float target_logit = logitbuf[target_token];
          float target_prob = std::exp(target_logit - max_logit) / sum_exp;

          nll += -std::log(target_prob + 1e-10);
      }

      return std::exp(nll / (N - 1));
  }

int main(){

    Model m;
    if (!load_gpt2_model(m)) {
        fprintf(stderr, "Failed to load model\n");
        exit(1);
    }

    BPEEncoder encoder;
    BPEDecoder decoder;

    if (!decoder.load("model/vocab.bin")) {
        fprintf(stderr, "Failed to load vocabulary\n");
        exit(1);
    }

    if (!encoder.load(decoder.vocab)) {
        fprintf(stderr, "Failed to initialize encoder\n");
        exit(1);
    }

    const int ntokens = 50; 
    const float temperature = 0.9f;
    std::string prompt = "what is the weather today?";

    printf("Prompt: \"%s\"\n", prompt.c_str());
    printf("Generated: ");

    auto sampled_start = std::chrono::steady_clock::now();
    int sampled_tokens = generate_temperature_scaling(prompt.c_str(), ntokens, m, encoder, decoder, temperature);
    auto sampled_end = std::chrono::steady_clock::now();
    float tps_sampled = sampled_tokens / std::chrono::duration<float>(sampled_end - sampled_start).count();
    printf("\n\nGenerated %d tokens total.\n", sampled_tokens);
    printf("\n\ntemperature sampling speed: %.2f tokens/sec\n", tps_sampled);

    float ppl = calculate_ppl(prompt.c_str(), ntokens, m, encoder, decoder, temperature);
    printf("\n\nthe perplexity score: %.2f", ppl);
    return 0;
}