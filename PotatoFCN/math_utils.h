#ifndef __MATH_UTILS_H__
#define __MATH_UTILS_H__

#include "tensor.h"

void relu(Tensor* t);
void relu_derivative(Tensor* t);
void softmax(Tensor* t);
void sigmoid_inplace(Tensor* t);
void sigmoid_derivative_inplace(Tensor* t);


#endif // !__MATH_UTILS_H__
