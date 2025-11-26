#pragma once

#include "tensor.h"

// Function declarations for operations
int sample_greedy(const Tensor<1>& logits);
float sdot(const float* a, const float* b, int n);
void saxpy(int n, float a, const float* x, float* y);
void sxpby(int n, const float* x, float b, float* y);
void sscal(int n, float a, float* x);
