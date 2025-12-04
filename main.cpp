#include <string>
#include <chrono>
#include <vector>
#include <iostream>

#include "bpe.h"
#include "tensor.h"
#include "model.h"
#include "ops.h"
#include "quantize.h"
#include "benchmark.h"

const int ctx_max = 1024;
extern bool load_gpt2_model(Model &m);

int generate_greedy(const char *prompt, int ntokens, Model &m, BPEEncoder &encoder, BPEDecoder &decoder, float temperature, bool verbose = true) {
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

        if (verbose) {
            printf("%s", decoder.vocab[token].c_str());
            fflush(stdout); 
        }
    }

    return ntokens;
};

void quantize_model(Model& m, QuantizationType type) {
    std::cout << "Quantizing model to " << (type == QuantizationType::INT8 ? "INT8" : "INT4") << "...\n";
    m.qtype = type;
    
    // Quantize wte
    if (type == QuantizationType::INT8) quantize_tensor_int8(m.wte_weight, m.q_wte_weight);
    else quantize_tensor_int4(m.wte_weight, m.q_wte_weight);

    // Quantize layers
    for (int i = 0; i < 12; i++) {
        auto& block = m.h[i];
        if (type == QuantizationType::INT8) {
            quantize_tensor_int8(block.attn.c_attn_weight, block.attn.q_c_attn_weight);
            quantize_tensor_int8(block.attn.c_proj_weight, block.attn.q_c_proj_weight);
            quantize_tensor_int8(block.mlp.c_fc_weight, block.mlp.q_c_fc_weight);
            quantize_tensor_int8(block.mlp.c_proj_weight, block.mlp.q_c_proj_weight);
        } else {
            quantize_tensor_int4(block.attn.c_attn_weight, block.attn.q_c_attn_weight);
            quantize_tensor_int4(block.attn.c_proj_weight, block.attn.q_c_proj_weight);
            quantize_tensor_int4(block.mlp.c_fc_weight, block.mlp.q_c_fc_weight);
            quantize_tensor_int4(block.mlp.c_proj_weight, block.mlp.q_c_proj_weight);
        }
    }
    std::cout << "Quantization complete.\n";
}

BenchmarkResult run_benchmark(Model& m, BPEEncoder& encoder, BPEDecoder& decoder, const std::string& name) {
    const int ntokens = 20; 
    const float temperature = 0.9f;
    std::string prompt = "The quick brown fox jumps over the lazy dog";
    
    BenchmarkResult res;
    res.name = name;
    
    // Warmup
    generate_greedy(prompt.c_str(), 5, m, encoder, decoder, temperature, false);
    
    BenchmarkTimer timer;
    timer.start();
    
    generate_greedy(prompt.c_str(), ntokens, m, encoder, decoder, temperature, false);
    
    double duration = timer.stop();
    res.tokens_per_sec = ntokens / duration;
    res.latency_ms = (duration * 1000.0) / ntokens;
    res.memory_usage_mb = MemoryTracker::get_peak_memory_usage() / (1024 * 1024);
    
    return res;
}

int main(int argc, char** argv) {
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

    std::vector<BenchmarkResult> results;
    
    // FP32 Benchmark
    std::cout << "Running FP32 Benchmark...\n";
    m.qtype = QuantizationType::FP32;
    results.push_back(run_benchmark(m, encoder, decoder, "FP32"));
    
    // INT8 Benchmark
    quantize_model(m, QuantizationType::INT8);
    std::cout << "Running INT8 Benchmark...\n";
    results.push_back(run_benchmark(m, encoder, decoder, "INT8"));
    
    // INT4 Benchmark
    // Note: We are re-quantizing from FP32 weights which are still in memory
    quantize_model(m, QuantizationType::INT4);
    std::cout << "Running INT4 Benchmark...\n";
    results.push_back(run_benchmark(m, encoder, decoder, "INT4"));
    
    print_benchmark_results(results);
    
    // Interactive mode if requested
    if (argc > 1 && std::string(argv[1]) == "--interactive") {
        // ...
    }

    return 0;
}