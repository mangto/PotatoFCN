#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <stdio.h>
#include <stdlib.h>

// --- Tensor 구조체 정의 ---
typedef struct {
    float* values;
    int* shape;
    int dims;
} Tensor;

// --- 기본 함수 ---
Tensor* create_tensor(const int* shape, int dims);
Tensor* create_tensor_from_array(const float* values, const int* shape, int dims);
Tensor* copy_tensor(const Tensor* src);
void free_tensor(Tensor* tensor);
int get_tensor_size(const Tensor* t);
void print_tensor(const Tensor* t, int print_values);
Tensor* tensor_elemwise_mul(const Tensor* t1, const Tensor* t2);
Tensor* tensor_concatenate(Tensor* t1, Tensor* t2, int axis);
void tensor_concatenate_backward(Tensor* grad_t1, Tensor* grad_t2, Tensor* grad_output, int axis, int c1_channels);

// --- im2col + BLAS 기반 연산 함수 ---
Tensor* tensor_conv2d(const Tensor* input, const Tensor* weights, const Tensor* biases, int stride, int padding, Tensor** col_buffer_ptr);
Tensor* tensor_maxpool(const Tensor* input, int pool_size, int stride, Tensor** max_indices_tensor);
Tensor* tensor_maxpool_backward(const Tensor* grad_output, const Tensor* original_input, const Tensor* max_indices);
Tensor* tensor_conv_grad_bias(const Tensor* grad_output);
Tensor* tensor_conv_grad_weights(const Tensor* input, const Tensor* grad_output, const Tensor* weights, int stride, int padding, Tensor** col_buffer_ptr);
Tensor* tensor_conv_grad_input(const Tensor* grad_output, const Tensor* weights, int stride, int padding, Tensor* col_buffer);
Tensor* tensor_transposed_conv2d(Tensor* input, Tensor* weights, Tensor* biases, int stride);
void tensor_transposed_conv2d_backward(Tensor* grad_input, Tensor* grad_weights, Tensor* grad_biases, Tensor* grad_output, Tensor* input, Tensor* weights, int stride);

#endif