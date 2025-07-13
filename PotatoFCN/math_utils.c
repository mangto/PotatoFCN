#include "math_utils.h"
#include <math.h>
#include <assert.h>

void relu(Tensor* t) {
    int total_elements = get_tensor_size(t);
    for (int i = 0; i < total_elements; i++) {
        if (t->values[i] < 0) {
            t->values[i] = 0.0f;
        }
    }
}

void relu_derivative(Tensor* t) {
    int total_elements = get_tensor_size(t);
    for (int i = 0; i < total_elements; i++) {
        t->values[i] = t->values[i] > 0 ? 1.0f : 0.0f;
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
