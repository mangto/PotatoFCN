#include "math_utils.h"
#include <math.h>
#include <assert.h>

void relu(Matrix* mat) {
    int total_elements = mat->shape[0] * mat->shape[1];
    for (int i = 0; i < total_elements; i++) {
        if (mat->values[i] < 0) {
            mat->values[i] = 0.0f;
        }
    }
}

void relu_derivative(Matrix* m) {
    int total_elements = m->shape[0] * m->shape[1];
    for (int i = 0; i < total_elements; i++) {
        if (m->values[i] > 0) {
            m->values[i] = 1.0f;
        }
        else {
            m->values[i] = 0.0f;
        }
    }
}

void softmax(Matrix* mat) {
    assert(mat->dims == 2);
    for (int i = 0; i < mat->shape[0]; i++) {
        float* row = &mat->values[i * mat->shape[1]];
        int num_cols = mat->shape[1];

        float max_val = row[0];
        for (int j = 1; j < num_cols; j++) {
            if (row[j] > max_val) {
                max_val = row[j];
            }
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