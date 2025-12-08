#pragma once

#include "tensor.h"

int sample_greedy(const Tensor<1>& logits, float temperature);
int temperature_sampling(const Tensor<1>& logits, float temperature);
float sdot(const float* a, const float* b, int n);
float sdot_simd(const float* a, const float* b, int n);
void saxpy(int n, float a, const float* x, float* y);
void sxpby(int n, const float* x, float b, float* y);
void sscal(int n, float a, float* x);

int sample_greedy(const Tensor_Quant<1>& logits, float temperature);
int temperature_sampling(const Tensor_Quant<1>& logits, float temperature);
float sdot(const int8_t* a, const int8_t* b, int n);
float sdot_simd(const int8_t* a, const int8_t* b, int n);

