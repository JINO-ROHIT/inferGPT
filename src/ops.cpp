#include <arm_neon.h>

#include "tensor.h"
#include "ops.h"

float sdot(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

 float sdot_simd(const float* a, const float* b, int n) {
      const int simd_size = 4; // 128 bit / 32 bit = 4 floats
      int simd_end = (n / simd_size) * simd_size;

      float32x4_t sum_vec = vdupq_n_f32(0.0f);

      for (int i = 0; i < simd_end; i += simd_size) {
          float32x4_t va = vld1q_f32(a + i);
          float32x4_t vb = vld1q_f32(b + i);
          sum_vec = vmlaq_f32(sum_vec, va, vb); 
      }

      float sum = vaddvq_f32(sum_vec);

      // remaining elements
      for (int i = simd_end; i < n; ++i) {
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


//greedily sample the highest logit after softmax
int sample_greedy(const Tensor<1>& logits, float temperature) {
    int n = logits.shape[0];

    int best_idx = 0;
    float best_score = logits[0] / temperature;

    for (int i = 1; i < n; i++) {
        float score = logits[i] / temperature;
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    return best_idx;
}



int temperature_sampling(const Tensor<1>& logits, float temperature) {
    int n = logits.shape[0];

    float* scores = new float[n];

    // Scale logits by temperature
    float max_score = -INFINITY; // keep track of this for softmax
    for (int i = 0; i < n; i++) {
        scores[i] = logits[i] / temperature;
        if (scores[i] > max_score) {
            max_score = scores[i];
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < n; i++) {
        scores[i] = expf(scores[i] - max_score);
        sum_exp += scores[i];
    }

    for (int i = 0; i < n; i++) {
        scores[i] /= sum_exp;
    }

    float r = (float)rand() / RAND_MAX;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += scores[i];
        if (r <= cdf) {
            delete[] scores;
            return i; 
        }
    }

    delete[] scores;
    return n - 1;
}
