#ifndef __FNN_H__
#define __FNN_H__

#include "matrix.h"

typedef struct {
    int num_layers;
    int* layer_sizes;
    Matrix** weights;
    Matrix** biases;
    Matrix* pre_activations;
    Matrix* activations;
} FNN;

void init_fnn(FNN* fnn, int num_layers, const int* layer_sizes);
void free_fnn(FNN* fnn);
void forward_fnn(FNN* fnn, Matrix* input);
void backward_fnn(FNN* fnn, Matrix* input, Matrix* target, float learning_rate);

#endif // !__FNN_H__