#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <assert.h>
#include <string.h>

Matrix create_matrix_from_array(const float* values, const int* shape, int dims) {
    Matrix mat;
    int total_elements = 1;
    for (int i = 0; i < dims; i++) {
        total_elements *= shape[i];
    }

    mat.values = (float*)malloc(total_elements * sizeof(float));
    assert(mat.values != NULL);

    mat.shape = (int*)malloc(dims * sizeof(int));
    assert(mat.shape != NULL);
    memcpy(mat.shape, shape, dims * sizeof(int));

    if (values != NULL) {
        memcpy(mat.values, values, total_elements * sizeof(float));
    }
    else {
        memset(mat.values, 0, total_elements * sizeof(float));
    }

    mat.dims = dims;
    mat.get = NULL; // Function pointers can be set if needed
    mat.set = NULL;

    return mat;
}

Matrix* create_matrix(const int* shape, int dims) {
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    if (mat == NULL) {
        fprintf(stderr, "Failed to allocate memory for Matrix struct.\n");
        exit(1);
    }
    *mat = create_matrix_from_array(NULL, shape, dims);
    return mat;
}

Matrix copy_matrix(const Matrix* src) {
    return create_matrix_from_array(src->values, src->shape, src->dims);
}

void free_matrix(Matrix* mat) {
    if (mat && mat->values) {
        free(mat->values);
        mat.values = NULL;
    }
    if (mat && mat->shape) {
        free(mat.shape);
        mat.shape = NULL;
    }
}

int is_same_shape(Matrix* mat1, Matrix* mat2) {
    if (mat1->dims != mat2->dims) { return 0; }
    for (int i = 0; i < mat1->dims; i++) {
        if (mat1->shape[i] != mat2->shape[i]) { return 0; }
    }
    return 1;
}

Matrix mat_dot(Matrix* mat1, Matrix* mat2) {
    assert(mat1->dims == 2 && mat2->dims == 2);
    assert(mat1->shape[1] == mat2->shape[0]);

    int m = mat1->shape[0];
    int k = mat1->shape[1];
    int n = mat2->shape[1];
    int result_shape[] = { m, n };
    Matrix result = create_matrix_from_array(NULL, result_shape, 2);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += mat1->values[i * k + l] * mat2->values[l * n + j];
            }
            result.values[i * n + j] = sum;
        }
    }
    return result;
}

Matrix mat_sub(Matrix* mat1, Matrix* mat2) {
    assert(is_same_shape(mat1, mat2));
    Matrix result = copy_matrix(mat1);
    int total_elements = 1;
    for (int i = 0; i < mat1->dims; i++) {
        total_elements *= mat1->shape[i];
    }
    for (int i = 0; i < total_elements; i++) {
        result.values[i] -= mat2->values[i];
    }
    return result;
}

Matrix mat_elemwise_mul(Matrix* mat1, Matrix* mat2) {
    assert(is_same_shape(mat1, mat2));
    Matrix result = copy_matrix(mat1);
    int total_elements = 1;
    for (int i = 0; i < mat1->dims; i++) {
        total_elements *= mat1->shape[i];
    }
    for (int i = 0; i < total_elements; i++) {
        result.values[i] *= mat2->values[i];
    }
    return result;
}

Matrix mat_transpose(Matrix* mat) {
    assert(mat->dims == 2);
    int new_shape[] = { mat->shape[1], mat->shape[0] };
    Matrix result = create_matrix_from_array(NULL, new_shape, 2);
    for (int i = 0; i < mat->shape[0]; i++) {
        for (int j = 0; j < mat->shape[1]; j++) {
            result.values[j * mat->shape[0] + i] = mat->values[i * mat->shape[1] + j];
        }
    }
    return result;
}

Matrix mat_add_broadcast(Matrix* mat_main, Matrix* mat_broadcast) {
    assert(mat_main->dims == 2 && mat_broadcast->dims == 2);
    assert(mat_main->shape[1] == mat_broadcast->shape[1]);
    assert(mat_broadcast->shape[0] == 1);

    Matrix result = copy_matrix(mat_main);
    int M = mat_main->shape[0];
    int N = mat_main->shape[1];

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            result.values[i * N + j] += mat_broadcast->values[j];
        }
    }
    return result;
}

void print_matrix(Matrix* mat) {
    if (!mat || !mat->values || !mat->shape) {
        printf("Invalid matrix.\n");
        return;
    }
    printf("Matrix (dims=%d): %d x %d\n", mat->dims, mat->shape[0], mat->dims > 1 ? mat->shape[1] : 1);
    for (int i = 0; i < mat->shape[0]; i++) {
        for (int j = 0; j < (mat->dims > 1 ? mat->shape[1] : 1); j++) {
            printf("%8.4f ", mat->values[i * (mat->dims > 1 ? mat->shape[1] : 1) + j]);
        }
        printf("\n");
    }
    printf("\n");
}