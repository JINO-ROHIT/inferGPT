#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <arm_neon.h>

#include "quantize.h"
#include "ops.h"

// Helper to get max absolute value in a row
static float get_max_abs(const float* data, int n) {
    float max_val = 0.0f;
    for (int i = 0; i < n; i++) {
        float abs_val = std::abs(data[i]);
        if (abs_val > max_val) max_val = abs_val;
    }
    return max_val;
}

void quantize_tensor_int8(const Tensor<2>& input, QuantizedTensor& output) {
    int rows = input.shape[0];
    int cols = input.shape[1];
    
    output.ndim = 2;
    output.shape[0] = rows;
    output.shape[1] = cols;
    output.alloc(rows * cols * sizeof(int8_t), QuantizationType::INT8);
    output.scales = new float[rows]; // Per-row scaling

    int8_t* out_data = (int8_t*)output.data;

    for (int i = 0; i < rows; i++) {
        const float* row_data = &input.data[i * cols];
        float max_val = get_max_abs(row_data, cols);
        
        // Avoid division by zero
        if (max_val == 0.0f) {
            output.scales[i] = 1.0f;
            memset(&out_data[i * cols], 0, cols);
            continue;
        }

        float scale = 127.0f / max_val;
        output.scales[i] = scale;
        float inv_scale = 1.0f / scale; // Store inverse for dequantization? No, usually store scale. 
        // Wait, standard is: real = quantized * scale. 
        // So if I store scale as (max_val / 127.0f), then real = q * scale.
        // Let's stick to that.
        
        float real_scale = max_val / 127.0f;
        output.scales[i] = real_scale;
        float quant_scale = 1.0f / real_scale;

        for (int j = 0; j < cols; j++) {
            float val = row_data[j] * quant_scale;
            int8_t qval = (int8_t)std::max(-127.0f, std::min(127.0f, roundf(val)));
            out_data[i * cols + j] = qval;
        }
    }
}

void quantize_tensor_int4(const Tensor<2>& input, QuantizedTensor& output) {
    int rows = input.shape[0];
    int cols = input.shape[1];
    
    output.ndim = 2;
    output.shape[0] = rows;
    output.shape[1] = cols;
    // Packed: 2 values per byte
    output.alloc((rows * cols + 1) / 2, QuantizationType::INT4);
    output.scales = new float[rows];

    uint8_t* out_data = (uint8_t*)output.data;
    
    for (int i = 0; i < rows; i++) {
        const float* row_data = &input.data[i * cols];
        float max_val = get_max_abs(row_data, cols);
        
        if (max_val == 0.0f) {
            output.scales[i] = 1.0f;
            // Zero out this row's packed data
            int row_bytes = (cols + 1) / 2;
            int row_offset = (i * cols) / 2; // Assuming contiguous packing across rows? 
            // Better to byte-align rows for easier access
            // But for now let's assume tight packing or row-byte-aligned?
            // Let's do row-byte-aligned for simplicity in matmul
            continue; 
        }

        float real_scale = max_val / 7.0f;
        output.scales[i] = real_scale;
        float quant_scale = 1.0f / real_scale;

        for (int j = 0; j < cols; j += 2) {
            // First value (lower 4 bits)
            float val0 = row_data[j] * quant_scale;
            int8_t qval0 = (int8_t)std::max(-7.0f, std::min(7.0f, roundf(val0)));
            
            // Second value (upper 4 bits)
            int8_t qval1 = 0;
            if (j + 1 < cols) {
                float val1 = row_data[j+1] * quant_scale;
                qval1 = (int8_t)std::max(-7.0f, std::min(7.0f, roundf(val1)));
            }

            // Pack: (qval1 << 4) | (qval0 & 0x0F)
            // Note: qval is signed -7 to 7. 
            // We need to mask it to 4 bits properly.
            uint8_t packed = ((uint8_t)qval1 << 4) | ((uint8_t)qval0 & 0x0F);
            
            // Calculate offset. If we pack tightly:
            // But wait, if we want to access rows easily, we should ensure rows start on byte boundaries.
            // For GPT-2, cols is usually multiple of 2 (e.g. 768), so it's fine.
            out_data[(i * cols + j) / 2] = packed;
        }
    }
}

void dequantize_tensor_int8(const QuantizedTensor& input, Tensor<2>& output) {
    int rows = input.shape[0];
    int cols = input.shape[1];
    
    // Output must be allocated
    int8_t* in_data = (int8_t*)input.data;
    
    for (int i = 0; i < rows; i++) {
        float scale = input.scales[i];
        float* out_row = &output.data[i * cols];
        int8_t* in_row = &in_data[i * cols];
        
        for (int j = 0; j < cols; j++) {
            out_row[j] = in_row[j] * scale;
        }
    }
}

void dequantize_tensor_int4(const QuantizedTensor& input, Tensor<2>& output) {
    int rows = input.shape[0];
    int cols = input.shape[1];
    
    uint8_t* in_data = (uint8_t*)input.data;
    
    for (int i = 0; i < rows; i++) {
        float scale = input.scales[i];
        float* out_row = &output.data[i * cols];
        
        for (int j = 0; j < cols; j += 2) {
            uint8_t packed = in_data[(i * cols + j) / 2];
            
            // Unpack
            int8_t val0 = (int8_t)(packed << 4); // Move lower 4 bits to top to sign extend
            val0 = val0 >> 4; // Arithmetic shift right to restore sign
            
            int8_t val1 = (int8_t)packed; // Upper 4 bits are already at top? No.
            // Packed was: (qval1 << 4) | (qval0 & 0x0F)
            // So qval1 is in upper 4 bits.
            val1 = val1 >> 4; // Arithmetic shift right to sign extend
            
            out_row[j] = val0 * scale;
            if (j + 1 < cols) {
                out_row[j+1] = val1 * scale;
            }
        }
    }
}

// Optimized INT8 dot product using NEON
// Dequantizes on the fly: sum(A[i] * B[i] * scale) = scale * sum(A[i] * B[i])
// But A is float, B is int8.
// So we compute sum(A[i] * (B[i] * scale)) = scale * sum(A[i] * B[i])
// Wait, A is float. We can't just multiply int8 * float easily in SIMD without conversion.
// We can convert int8 to float, then FMA.
void qmatmul_int8(const Tensor<1>& A, const QuantizedTensor& B, Tensor<1>& C) {
    // C = B * A (where B is weights [rows, cols], A is input [cols])
    // But wait, the function signature says C = A * B?
    // In model.cpp: apply_lm_head: logits[j] = sdot_simd(emb_in.data, w, embedding_dim);
    // w is a row of wte_weight.
    // So we are doing dot product of input A with each row of B.
    
    int rows = B.shape[0];
    int cols = B.shape[1];
    int8_t* b_data = (int8_t*)B.data;
    
    for (int i = 0; i < rows; i++) {
        float scale = B.scales[i];
        const float* a_ptr = A.data;
        const int8_t* b_ptr = &b_data[i * cols];
        
        float sum = 0.0f;
        int j = 0;
        
        // NEON loop
        for (; j <= cols - 16; j += 16) {
            float32x4_t sum_vec0 = vdupq_n_f32(0.0f);
            float32x4_t sum_vec1 = vdupq_n_f32(0.0f);
            float32x4_t sum_vec2 = vdupq_n_f32(0.0f);
            float32x4_t sum_vec3 = vdupq_n_f32(0.0f);
            
            // Load 16 floats from A
            float32x4_t a0 = vld1q_f32(a_ptr + j);
            float32x4_t a1 = vld1q_f32(a_ptr + j + 4);
            float32x4_t a2 = vld1q_f32(a_ptr + j + 8);
            float32x4_t a3 = vld1q_f32(a_ptr + j + 12);
            
            // Load 16 int8s from B
            int8x16_t b_int8 = vld1q_s8(b_ptr + j);
            
            // Expand int8 to int16
            int16x8_t b_low = vmovl_s8(vget_low_s8(b_int8));
            int16x8_t b_high = vmovl_s8(vget_high_s8(b_int8));
            
            // Expand int16 to int32, then convert to float
            // Low part (first 8)
            int32x4_t b0_i32 = vmovl_s16(vget_low_s16(b_low));
            int32x4_t b1_i32 = vmovl_s16(vget_high_s16(b_low));
            
            float32x4_t b0_f = vcvtq_f32_s32(b0_i32);
            float32x4_t b1_f = vcvtq_f32_s32(b1_i32);
            
            // High part (next 8)
            int32x4_t b2_i32 = vmovl_s16(vget_low_s16(b_high));
            int32x4_t b3_i32 = vmovl_s16(vget_high_s16(b_high));
            
            float32x4_t b2_f = vcvtq_f32_s32(b2_i32);
            float32x4_t b3_f = vcvtq_f32_s32(b3_i32);
            
            // FMA
            sum_vec0 = vmlaq_f32(sum_vec0, a0, b0_f);
            sum_vec1 = vmlaq_f32(sum_vec1, a1, b1_f);
            sum_vec2 = vmlaq_f32(sum_vec2, a2, b2_f);
            sum_vec3 = vmlaq_f32(sum_vec3, a3, b3_f);
            
            // Horizontal sum
            sum += vaddvq_f32(sum_vec0) + vaddvq_f32(sum_vec1) + 
                   vaddvq_f32(sum_vec2) + vaddvq_f32(sum_vec3);
        }
        
        // Handle remaining
        for (; j < cols; j++) {
            sum += a_ptr[j] * (float)b_ptr[j];
        }
        
        C.data[i] = sum * scale;
    }
}

void qmatmul_int4(const Tensor<1>& A, const QuantizedTensor& B, Tensor<1>& C) {
    int rows = B.shape[0];
    int cols = B.shape[1];
    uint8_t* b_data = (uint8_t*)B.data;
    
    for (int i = 0; i < rows; i++) {
        float scale = B.scales[i];
        const float* a_ptr = A.data;
        
        float sum = 0.0f;
        
        // Process 2 values at a time
        for (int j = 0; j < cols; j += 2) {
            uint8_t packed = b_data[(i * cols + j) / 2];
            
            // Unpack
            int8_t val0 = (int8_t)(packed << 4);
            val0 = val0 >> 4;
            
            int8_t val1 = (int8_t)packed;
            val1 = val1 >> 4;
            
            sum += a_ptr[j] * (float)val0;
            if (j + 1 < cols) {
                sum += a_ptr[j+1] * (float)val1;
            }
        }
        
        C.data[i] = sum * scale;
    }
}
