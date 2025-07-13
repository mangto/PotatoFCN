#include "tensor.h"
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <float.h>
#include <stdio.h>

// --- Tensor creation and memory management ---
Tensor* create_tensor(const int* shape, int dims) {
    return create_tensor_from_array(NULL, shape, dims);
}

Tensor* create_tensor_from_array(const float* values, const int* shape, int dims) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    assert(t != NULL);

    t->dims = dims;
    t->shape = (int*)malloc(dims * sizeof(int));
    assert(t->shape != NULL);
    memcpy(t->shape, shape, dims * sizeof(int));

    int total_elements = get_tensor_size(t);
    t->values = (float*)malloc(total_elements * sizeof(float));
    assert(t->values != NULL);

    if (values != NULL) {
        memcpy(t->values, values, total_elements * sizeof(float));
    }
    else {
        memset(t->values, 0, total_elements * sizeof(float));
    }
    return t;
}

Tensor* copy_tensor(const Tensor* src) {
    return create_tensor_from_array(src->values, src->shape, src->dims);
}

void free_tensor(Tensor* t) {
    if (t) {
        if (t->values) free(t->values);
        if (t->shape) free(t->shape);
        free(t);
    }
}

int get_tensor_size(const Tensor* t) {
    if (!t) return 0;
    int size = 1;
    for (int i = 0; i < t->dims; i++) {
        size *= t->shape[i];
    }
    return size;
}

void print_tensor(const Tensor* t, int print_values) {
    if (!t) {
        printf("NULL Tensor\n");
        return;
    }
    printf("Tensor (dims=%d): [", t->dims);
    for (int i = 0; i < t->dims; i++) {
        printf("%d%s", t->shape[i], i == t->dims - 1 ? "" : ", ");
    }
    printf("], Total elements: %d\n", get_tensor_size(t));
    if (print_values) {
        int size = get_tensor_size(t);
        for (int i = 0; i < size && i < 10; i++) { // Print first 10 values
            printf("%f ", t->values[i]);
        }
        if (size > 10) printf("...");
        printf("\n");
    }
}


// --- Basic operations ---
void tensor_update(Tensor* t, const Tensor* grad, float learning_rate, int batch_size) {
    int size = get_tensor_size(t);
    for (int i = 0; i < size; i++) {
        t->values[i] -= (learning_rate * grad->values[i]) / batch_size;
    }
}

void tensor_reshape(Tensor* t, int* new_shape, int new_dims) {
    int old_size = get_tensor_size(t);
    int new_size = 1;
    for (int i = 0; i < new_dims; i++) {
        new_size *= new_shape[i];
    }
    assert(old_size == new_size);

    free(t->shape);
    t->shape = (int*)malloc(new_dims * sizeof(int));
    memcpy(t->shape, new_shape, new_dims * sizeof(int));
    t->dims = new_dims;
}

Tensor* tensor_sub(const Tensor* t1, const Tensor* t2) {
    assert(get_tensor_size(t1) == get_tensor_size(t2));
    Tensor* result = copy_tensor(t1);
    int size = get_tensor_size(result);
    for (int i = 0; i < size; i++) {
        result->values[i] -= t2->values[i];
    }
    return result;
}

Tensor* tensor_elemwise_mul(const Tensor* t1, const Tensor* t2) {
    assert(get_tensor_size(t1) == get_tensor_size(t2));
    Tensor* result = copy_tensor(t1);
    int size = get_tensor_size(result);
    for (int i = 0; i < size; i++) {
        result->values[i] *= t2->values[i];
    }
    return result;
}

Tensor* tensor_dot(const Tensor* t1, const Tensor* t2) {
    assert(t1->dims == 2 && t2->dims == 2);
    assert(t1->shape[1] == t2->shape[0]);
    int m = t1->shape[0], k = t1->shape[1], n = t2->shape[1];
    Tensor* result = create_tensor((int[]) { m, n }, 2);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += t1->values[i * k + l] * t2->values[l * n + j];
            }
            result->values[i * n + j] = sum;
        }
    }
    return result;
}

Tensor* tensor_transpose(const Tensor* t) {
    assert(t->dims == 2);
    Tensor* result = create_tensor((int[]) { t->shape[1], t->shape[0] }, 2);
    for (int i = 0; i < t->shape[0]; i++) {
        for (int j = 0; j < t->shape[1]; j++) {
            result->values[j * t->shape[0] + i] = t->values[i * t->shape[1] + j];
        }
    }
    return result;
}

Tensor* tensor_add_broadcast(const Tensor* main, const Tensor* broadcast) {
    assert(main->dims == 2 && broadcast->dims == 2);
    assert(main->shape[1] == broadcast->shape[1] && broadcast->shape[0] == 1);
    Tensor* result = copy_tensor(main);
    for (int i = 0; i < main->shape[0]; i++) {
        for (int j = 0; j < main->shape[1]; j++) {
            result->values[i * main->shape[1] + j] += broadcast->values[j];
        }
    }
    return result;
}

Tensor* tensor_sum_along_axis(const Tensor* t, int axis) {
    assert(axis == 0 && t->dims == 2);
    Tensor* result = create_tensor((int[]) { 1, t->shape[1] }, 2);
    for (int j = 0; j < t->shape[1]; j++) {
        float sum = 0.0f;
        for (int i = 0; i < t->shape[0]; i++) {
            sum += t->values[i * t->shape[1] + j];
        }
        result->values[j] = sum;
    }
    return result;
}

// --- CNN operations ---

Tensor* tensor_conv2d(const Tensor* input, const Tensor* weights, const Tensor* biases, int stride, int padding) {
    assert(input->dims == 4 && weights->dims == 4 && biases->dims == 1);
    assert(input->shape[1] == weights->shape[1]);

    int N = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
    int F = weights->shape[0], K = weights->shape[2];
    int H_out = (H - K + 2 * padding) / stride + 1;
    int W_out = (W - K + 2 * padding) / stride + 1;

    Tensor* output = create_tensor((int[]) { N, F, H_out, W_out }, 4);

    for (int n = 0; n < N; n++)
        for (int f = 0; f < F; f++)
            for (int ho = 0; ho < H_out; ho++)
                for (int wo = 0; wo < W_out; wo++) {
                    float sum = 0.0f;
                    for (int c = 0; c < C; c++)
                        for (int kh = 0; kh < K; kh++)
                            for (int kw = 0; kw < K; kw++) {
                                int hi = ho * stride + kh - padding;
                                int wi = wo * stride + kw - padding;
                                if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                                    float in_val = input->values[n * C * H * W + c * H * W + hi * W + wi];
                                    float w_val = weights->values[f * C * K * K + c * K * K + kh * K + kw];
                                    sum += in_val * w_val;
                                }
                            }
                    output->values[n * F * H_out * W_out + f * H_out * W_out + ho * W_out + wo] = sum + biases->values[f];
                }
    return output;
}

Tensor* tensor_maxpool(const Tensor* input, int pool_size, int stride, Tensor** max_indices_tensor) {
    assert(input->dims == 4);
    int N = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
    int H_out = (H - pool_size) / stride + 1;
    int W_out = (W - pool_size) / stride + 1;

    Tensor* output = create_tensor((int[]) { N, C, H_out, W_out }, 4);
    if (max_indices_tensor) {
        *max_indices_tensor = create_tensor((int[]) { N, C, H_out, W_out }, 4);
    }

    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
            for (int ho = 0; ho < H_out; ho++)
                for (int wo = 0; wo < W_out; wo++) {
                    float max_val = -FLT_MAX;
                    int max_idx = -1;
                    for (int ph = 0; ph < pool_size; ph++)
                        for (int pw = 0; pw < pool_size; pw++) {
                            int hi = ho * stride + ph;
                            int wi = wo * stride + pw;
                            int current_flat_idx = hi * W + wi;
                            float val = input->values[n * C * H * W + c * H * W + current_flat_idx];
                            if (val > max_val) {
                                max_val = val;
                                max_idx = current_flat_idx; // Store the 1D index of the 2D feature map
                            }
                        }
                    output->values[n * C * H_out * W_out + c * H_out * W_out + ho * W_out + wo] = max_val;
                    if (max_indices_tensor) {
                        (*max_indices_tensor)->values[n * C * H_out * W_out + c * H_out * W_out + ho * W_out + wo] = (float)max_idx;
                    }
                }
    return output;
}

// --- CNN backward pass ---

Tensor* tensor_maxpool_backward(const Tensor* grad_output, const Tensor* original_input, const Tensor* max_indices) {
    Tensor* grad_input = create_tensor(original_input->shape, original_input->dims); // Initialized to zeros
    int N = grad_output->shape[0], C = grad_output->shape[1];
    int H_out = grad_output->shape[2], W_out = grad_output->shape[3];
    int H_in = original_input->shape[2], W_in = original_input->shape[3];

    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
            for (int ho = 0; ho < H_out; ho++)
                for (int wo = 0; wo < W_out; wo++) {
                    float grad = grad_output->values[n * C * H_out * W_out + c * H_out * W_out + ho * W_out + wo];
                    int max_idx_flat = (int)max_indices->values[n * C * H_out * W_out + c * H_out * W_out + ho * W_out + wo];
                    int grad_input_idx = n * C * H_in * W_in + c * H_in * W_in + max_idx_flat;
                    grad_input->values[grad_input_idx] += grad; // Add gradient to the max value position
                }
    return grad_input;
}

Tensor* tensor_conv_grad_bias(const Tensor* grad_output) {
    int F = grad_output->shape[1];
    Tensor* grad_biases = create_tensor((int[]) { F }, 1);
    int N = grad_output->shape[0], H_out = grad_output->shape[2], W_out = grad_output->shape[3];

    for (int f = 0; f < F; f++) {
        float sum = 0.0f;
        for (int n = 0; n < N; n++)
            for (int h = 0; h < H_out; h++)
                for (int w = 0; w < W_out; w++) {
                    sum += grad_output->values[n * F * H_out * W_out + f * H_out * W_out + h * W_out + w];
                }
        grad_biases->values[f] = sum;
    }
    return grad_biases;
}

// [BUG FIX] Convolutional weight gradient calculation logic
Tensor* tensor_conv_grad_weights(const Tensor* input, const Tensor* grad_output, int stride, int padding) {
    int N = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
    int F = grad_output->shape[1], H_out = grad_output->shape[2], W_out = grad_output->shape[3];
    // The size of the weight (filter) cannot be inferred from the grad_output size,
    // so it is calculated based on the input and output shapes.
    int K = H - (H_out - 1) * stride + 2 * padding;

    Tensor* grad_weights = create_tensor((int[]) { F, C, K, K }, 4);

    for (int f = 0; f < F; f++)
        for (int c = 0; c < C; c++)
            for (int kh = 0; kh < K; kh++)
                for (int kw = 0; kw < K; kw++) {
                    float sum = 0.0f;
                    for (int n = 0; n < N; n++)
                        for (int ho = 0; ho < H_out; ho++)
                            for (int wo = 0; wo < W_out; wo++) {
                                int hi = ho * stride + kh - padding;
                                int wi = wo * stride + kw - padding;
                                if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                                    float grad_val = grad_output->values[n * F * H_out * W_out + f * H_out * W_out + ho * W_out + wo];
                                    float in_val = input->values[n * C * H * W + c * H * W + hi * W + wi];
                                    sum += grad_val * in_val;
                                }
                            }
                    grad_weights->values[f * C * K * K + c * K * K + kh * K + kw] = sum;
                }
    return grad_weights;
}

// [BUG FIX] Convolutional input gradient calculation logic (Transposed Convolution)
Tensor* tensor_conv_grad_input(const Tensor* grad_output, const Tensor* weights, int stride, int padding, const int* output_shape) {
    Tensor* grad_input = create_tensor(output_shape, 4); // Same shape as the input to the original conv layer

    int N = grad_output->shape[0], F = grad_output->shape[1], H_out = grad_output->shape[2], W_out = grad_output->shape[3];
    int C = weights->shape[1], K = weights->shape[2];
    int H_in = output_shape[2], W_in = output_shape[3];

    for (int n = 0; n < N; n++)
        for (int f = 0; f < F; f++)
            for (int ho = 0; ho < H_out; ho++)
                for (int wo = 0; wo < W_out; wo++) {
                    float grad_val = grad_output->values[n * F * H_out * W_out + f * H_out * W_out + ho * W_out + wo];
                    for (int c = 0; c < C; c++)
                        for (int kh = 0; kh < K; kh++)
                            for (int kw = 0; kw < K; kw++) {
                                int hi = ho * stride + kh - padding;
                                int wi = wo * stride + kw - padding;
                                if (hi >= 0 && hi < H_in && wi >= 0 && wi < W_in) {
                                    float w_val = weights->values[f * C * K * K + c * K * K + kh * K + kw];
                                    // Add the gradient to the corresponding input location
                                    grad_input->values[n * C * H_in * W_in + c * H_in * W_in + hi * W_in + wi] += grad_val * w_val;
                                }
                            }
                }
    return grad_input;
}
