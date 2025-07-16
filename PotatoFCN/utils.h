#ifndef UTILS_H
#define UTILS_H

#include "tensor.h"

// Dataset loading
void load_preprocessed_data(
    const char* path,
    int* num_samples,
    int* h,
    int* w,
    float** images,
    float** masks
);

// Loss functions
double mse_loss(const Tensor* pred, const Tensor* target);
Tensor* mse_loss_backward(const Tensor* pred, const Tensor* target);

float bce_loss(const Tensor* pred, const Tensor* target);
Tensor* bce_loss_backward(const Tensor* pred, const Tensor* target);

float focal_loss(const Tensor* pred, const Tensor* target, float gamma);
Tensor* focal_loss_backward(const Tensor* pred, const Tensor* target, float gamma);

#endif // UTILS_H
