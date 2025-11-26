#include "include/tensor.h"
#include "include/ops.h"

float sdot(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// single-precision vector addition: y = (a * x) + y
void saxpy(int n, float a, const float* x, float* y) {
    for (int i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

// variant on saxpy: y = x + (b * y)
void sxpby(int n, const float* x, float b, float* y) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i] + b * y[i];
    }
}

// x = a * x
void sscal(int n, float a, float* x) {
    for (int i = 0; i < n; i++) {
        x[i] *= a;
    }
}


//greedily sample the highest logit
int sample_greedy(const Tensor<1>& logits) {
    int n = logits.shape[0];
    float *data = logits.data;
    int argmax = 0;
    for (int i = 1; i < n; i++)
        if (data[i] > data[argmax])
            argmax = i;
    return argmax;
}