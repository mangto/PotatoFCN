#ifndef __MATH_UTILS_H__
#define __MATH_UTILS_H__

#include "matrix.h"

void relu(Matrix* mat);
void relu_derivative(Matrix* m);
void softmax(Matrix* mat);

#endif // !__MATH_UTILS_H__