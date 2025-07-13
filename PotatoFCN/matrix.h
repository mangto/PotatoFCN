#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <assert.h>

typedef struct Matrix {
    float* values;
    int* shape;
    int dims;
    float (*get)(struct Matrix*, ...);
    void  (*set)(struct Matrix*, float, ...);
} Matrix;

Matrix copy_matrix(const Matrix* src);
Matrix* create_matrix(const int* shape, int dims);
Matrix create_matrix_from_array(const float* values, const int* shape, int dims);
void init_matrix(Matrix* mat, float* values, int* shape, int dims);
void print_matrix(Matrix* mat);
void free_matrix(Matrix* mat);
float mat_get(Matrix* mat, ...);
void mat_set(Matrix* mat, float val, ...);
int is_same_shape(Matrix* mat1, Matrix* mat2);
Matrix mat_add(Matrix* mat1, Matrix* mat2);
Matrix mat_add_broadcast(Matrix* mat_main, Matrix* mat_broadcast);
Matrix mat_sub(Matrix* mat1, Matrix* mat2);
Matrix mat_dot(Matrix* mat1, Matrix* mat2);
Matrix mat_transpose(Matrix* mat);
Matrix mat_elemwise_mul(Matrix* mat1, Matrix* mat2);

#endif // !__MATRIX_H__
