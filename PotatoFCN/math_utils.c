#include "math_utils.h"
#include "tensor.h"
#include <math.h>
#include <assert.h>
#include <immintrin.h>

void relu(Tensor* t) {
    int total_elements = get_tensor_size(t);
    float* p = t->values;
    __m256 ymm_zeros = _mm256_setzero_ps();

    int i = 0;
    for (; i <= total_elements - 8; i += 8) {
        __m256 ymm_vals = _mm256_loadu_ps(p + i);
        ymm_vals = _mm256_max_ps(ymm_vals, ymm_zeros);
        _mm256_storeu_ps(p + i, ymm_vals);
    }

    for (; i < total_elements; i++) {
        if (p[i] < 0) {
            p[i] = 0.0f;
        }
    }
}

void relu_derivative(Tensor* t) {
    int total_elements = get_tensor_size(t);
    float* p = t->values;
    __m256 ymm_zeros = _mm256_setzero_ps();
    __m256 ymm_ones = _mm256_set1_ps(1.0f);

    int i = 0;
    for (; i <= total_elements - 8; i += 8) {
        __m256 ymm_vals = _mm256_loadu_ps(p + i);
        __m256 ymm_mask = _mm256_cmp_ps(ymm_vals, ymm_zeros, _CMP_GT_OQ);
        __m256 ymm_result = _mm256_and_ps(ymm_mask, ymm_ones);
        _mm256_storeu_ps(p + i, ymm_result);
    }

    for (; i < total_elements; i++) {
        p[i] = p[i] > 0 ? 1.0f : 0.0f;
    }
}


void softmax(Tensor* t) {
    assert(t->dims == 2);
    for (int i = 0; i < t->shape[0]; i++) {
        float* row = &t->values[i * t->shape[1]];
        int num_cols = t->shape[1];
        float max_val = row[0];
        for (int j = 1; j < num_cols; j++) {
            if (row[j] > max_val) max_val = row[j];
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < num_cols; j++) {
            sum_exp += expf(row[j] - max_val);
        }
        for (int j = 0; j < num_cols; j++) {
            row[j] = expf(row[j] - max_val) / sum_exp;
        }
    }
}