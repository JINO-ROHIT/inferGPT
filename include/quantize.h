#pragma once

#include <cstdint>
#include <vector>
#include <cmath>
#include "tensor.h"

enum class QuantizationType {
    FP32,
    INT8,
    INT4
};

struct QuantizationParams {
    float scale;
    float zero_point; // Usually 0 for symmetric quantization
};

// Simple container for quantized data
struct QuantizedTensor {
    void* data;       // int8_t* for INT8, uint8_t* for INT4 (packed)
    float* scales;    // Per-channel or per-tensor scales
    int shape[3];
    int ndim;
    QuantizationType type;
    size_t data_size; // Size in bytes

    QuantizedTensor() : data(nullptr), scales(nullptr), ndim(0), type(QuantizationType::FP32), data_size(0) {}
    
    ~QuantizedTensor() {
        if (data) delete[] (uint8_t*)data;
        if (scales) delete[] scales;
    }

    void alloc(size_t size, QuantizationType qtype) {
        data_size = size;
        type = qtype;
        data = new uint8_t[size];
    }
};

// Function declarations
void quantize_tensor_int8(const Tensor<2>& input, QuantizedTensor& output);
void quantize_tensor_int4(const Tensor<2>& input, QuantizedTensor& output);

void dequantize_tensor_int8(const QuantizedTensor& input, Tensor<2>& output);
void dequantize_tensor_int4(const QuantizedTensor& input, Tensor<2>& output);

// Quantized matrix multiplication: C = A * B
// A is typically activation (FP32), B is weights (Quantized)
// We dequantize B on the fly or use specialized kernels
void qmatmul_int8(const Tensor<1>& A, const QuantizedTensor& B, Tensor<1>& C);
void qmatmul_int4(const Tensor<1>& A, const QuantizedTensor& B, Tensor<1>& C);
