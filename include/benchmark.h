#pragma once

#include <string>
#include <vector>
#include <chrono>

struct BenchmarkResult {
    std::string name;
    double tokens_per_sec;
    double latency_ms;
    size_t memory_usage_mb;
    double perplexity; // Optional
};

class BenchmarkTimer {
    std::chrono::high_resolution_clock::time_point start_time;
public:
    void start();
    double stop(); // Returns duration in seconds
};

class MemoryTracker {
public:
    static size_t get_current_memory_usage(); // Returns bytes
    static size_t get_peak_memory_usage();
};

void print_benchmark_results(const std::vector<BenchmarkResult>& results);
