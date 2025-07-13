#include "fnn.h"
#include "matrix.h"
#include "math_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

void init_fnn(FNN* fnn, int num_layers, const int* layer_sizes) {
    fnn->num_layers = num_layers;
    fnn->layer_sizes = (int*)malloc(num_layers * sizeof(int));
    memcpy(fnn->layer_sizes, layer_sizes, num_layers * sizeof(int));

    fnn->weights = (Matrix**)malloc((num_layers - 1) * sizeof(Matrix*));
    fnn->biases = (Matrix**)malloc((num_layers - 1) * sizeof(Matrix*));
    fnn->pre_activations = (Matrix*)malloc((num_layers - 1) * sizeof(Matrix));
    fnn->activations = (Matrix*)malloc(num_layers * sizeof(Matrix));

    fnn->activations[0].values = NULL;
    for (int i = 0; i < num_layers - 1; i++) {
        fnn->pre_activations[i].values = NULL;
        fnn->activations[i + 1].values = NULL;
    }

    for (int i = 0; i < num_layers - 1; i++) {
        int input_size = layer_sizes[i];
        int output_size = layer_sizes[i + 1];
        int shape_weights[] = { input_size, output_size };
        int shape_biases[] = { 1, output_size };

        fnn->weights[i] = create_matrix(shape_weights, 2);
        fnn->biases[i] = create_matrix(shape_biases, 2);

        float scale = sqrtf(2.0f / input_size);
        for (int j = 0; j < input_size * output_size; j++) {
            fnn->weights[i]->values[j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
        }
        for (int j = 0; j < output_size; j++) {
            fnn->biases[i]->values[j] = 0.0f;
        }
    }
}

void free_fnn(FNN* fnn) {
    for (int i = 0; i < fnn->num_layers - 1; i++) {
        free_matrix(fnn->weights[i]);
        free(fnn->weights[i]);
        free_matrix(fnn->biases[i]);
        free(fnn->biases[i]);
    }
    free(fnn->weights);
    free(fnn->biases);

    if (fnn->activations[0].values != NULL) free_matrix(&fnn->activations[0]);
    for (int i = 0; i < fnn->num_layers - 1; i++) {
        if (fnn->pre_activations[i].values != NULL) free_matrix(&fnn->pre_activations[i]);
        if (fnn->activations[i + 1].values != NULL) free_matrix(&fnn->activations[i + 1]);
    }
    free(fnn->pre_activations);
    free(fnn->activations);

    free(fnn->layer_sizes);
}

void forward_fnn(FNN* fnn, Matrix* input) {
    assert(input->shape[1] == fnn->layer_sizes[0]);

    if (fnn->activations[0].values != NULL) free_matrix(&fnn->activations[0]);
    for (int i = 0; i < fnn->num_layers - 1; ++i) {
        if (fnn->pre_activations[i].values != NULL) free_matrix(&fnn->pre_activations[i]);
        if (fnn->activations[i + 1].values != NULL) free_matrix(&fnn->activations[i + 1]);
    }

    fnn->activations[0] = copy_matrix(input);

    for (int i = 0; i < fnn->num_layers - 1; i++) {
        Matrix y = mat_dot(&fnn->activations[i], fnn->weights[i]);
        fnn->pre_activations[i] = mat_add_broadcast(&y, fnn->biases[i]);

        fnn->activations[i + 1] = copy_matrix(&fnn->pre_activations[i]);
        if (i == fnn->num_layers - 2) {
            softmax(&fnn->activations[i + 1]);
        }
        else {
            relu(&fnn->activations[i + 1]);
        }

        free_matrix(&y);
    }
}

void backward_fnn(FNN* fnn, Matrix* input, Matrix* target, float learning_rate) {
    int last_layer_idx = fnn->num_layers - 2;

    Matrix dL_dZ = mat_sub(&fnn->activations[last_layer_idx + 1], target);

    for (int i = last_layer_idx; i >= 0; i--) {
        Matrix A_T = mat_transpose(&fnn->activations[i]);
        Matrix dL_dW = mat_dot(&A_T, &dL_dZ);

        Matrix grad_B = create_matrix_from_array(NULL, fnn->biases[i]->shape, 2);
        for (int r = 0; r < dL_dZ.shape[0]; r++) {
            for (int c = 0; c < dL_dZ.shape[1]; c++) {
                grad_B.values[c] += dL_dZ.values[r * dL_dZ.shape[1] + c];
            }
        }

        float batch_size = (float)dL_dZ.shape[0];
        int total_weights = fnn->weights[i]->shape[0] * fnn->weights[i]->shape[1];
        for (int j = 0; j < total_weights; j++) {
            fnn->weights[i]->values[j] -= learning_rate * (dL_dW.values[j] / batch_size);
        }

        int total_biases = fnn->biases[i]->shape[1];
        for (int j = 0; j < total_biases; j++) {
            fnn->biases[i]->values[j] -= learning_rate * (grad_B.values[j] / batch_size);
        }

        if (i > 0) {
            Matrix W_T = mat_transpose(fnn->weights[i]);
            Matrix prev_dL_dA = mat_dot(&dL_dZ, &W_T);

            Matrix prev_dZ = copy_matrix(&fnn->pre_activations[i - 1]);
            relu_derivative(&prev_dZ);

            free_matrix(&dL_dZ);
            dL_dZ = mat_elemwise_mul(&prev_dL_dA, &prev_dZ);

            free_matrix(&W_T);
            free_matrix(&prev_dL_dA);
            free_matrix(&prev_dZ);
        }

        free_matrix(&A_T);
        free_matrix(&dL_dW);
        free_matrix(&grad_B);
    }

    free_matrix(&dL_dZ);
}