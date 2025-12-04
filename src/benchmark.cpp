#include "benchmark.h"
#include <iostream>
#include <iomanip>
#include <sys/resource.h>
#include <unistd.h>

void BenchmarkTimer::start() {
    start_time = std::chrono::high_resolution_clock::now();
}

double BenchmarkTimer::stop() {
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    return diff.count();
}

size_t MemoryTracker::get_current_memory_usage() {
    // Approximation using RSS
    return get_peak_memory_usage();
}

size_t MemoryTracker::get_peak_memory_usage() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        // ru_maxrss is in kilobytes on Linux, bytes on macOS? 
        // Usually KB on Linux.
        return usage.ru_maxrss * 1024; 
    }
    return 0;
}

void print_benchmark_results(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n=================================================================\n";
    std::cout << "                      BENCHMARK RESULTS                          \n";
    std::cout << "=================================================================\n";
    std::cout << std::left << std::setw(20) << "Model" 
              << std::setw(15) << "Tok/s" 
              << std::setw(15) << "Latency(ms)" 
              << std::setw(15) << "Memory(MB)" << "\n";
    std::cout << "-----------------------------------------------------------------\n";
    
    for (const auto& res : results) {
        std::cout << std::left << std::setw(20) << res.name 
                  << std::setw(15) << std::fixed << std::setprecision(2) << res.tokens_per_sec 
                  << std::setw(15) << std::fixed << std::setprecision(2) << res.latency_ms 
                  << std::setw(15) << (res.memory_usage_mb) << "\n";
    }
    std::cout << "=================================================================\n";
}
