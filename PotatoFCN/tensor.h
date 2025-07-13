#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <stdio.h>
#include <stdlib.h>

typedef struct {
    float* values; // Tensor Data
    int* shape;    // Tensor Shape (e.g., [64, 1, 28, 28])
    int dims;      // Tensor dimensions
} Tensor;


Tensor* create_tensor(const int* shape, int dims);
Tensor* create_tensor_from_array(const float* values, const int* shape, int dims);
Tensor* copy_tensor(const Tensor* src);
void free_tensor(Tensor* t);
int get_tensor_size(const Tensor* t);
void print_tensor(const Tensor* t, int print_values);


void tensor_update(Tensor* t, const Tensor* grad, float learning_rate, int batch_size);
void tensor_reshape(Tensor* t, int* new_shape, int new_dims);
Tensor* tensor_sub(const Tensor* t1, const Tensor* t2);
Tensor* tensor_elemwise_mul(const Tensor* t1, const Tensor* t2);
Tensor* tensor_dot(const Tensor* t1, const Tensor* t2);
Tensor* tensor_transpose(const Tensor* t);
Tensor* tensor_add_broadcast(const Tensor* mat_main, const Tensor* mat_broadcast);
Tensor* tensor_sum_along_axis(const Tensor* t, int axis);


Tensor* tensor_conv2d(const Tensor* input, const Tensor* weights, const Tensor* biases, int stride, int padding);
Tensor* tensor_maxpool(const Tensor* input, int pool_size, int stride, Tensor** max_indices_tensor);


Tensor* tensor_conv_grad_bias(const Tensor* grad_output);
Tensor* tensor_conv_grad_weights(const Tensor* input, const Tensor* grad_output, int stride, int padding);
Tensor* tensor_conv_grad_input(const Tensor* grad_output, const Tensor* weights, int stride, int padding, const int* output_shape);
Tensor* tensor_maxpool_backward(const Tensor* grad_output, const Tensor* original_input, const Tensor* max_indices);

#endif // !__TENSOR_H__
