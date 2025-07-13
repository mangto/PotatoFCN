#ifndef __CNN_H__
#define __CNN_H__

#include "tensor.h"

// Enum for different layer types
typedef enum {
    LAYER_DENSE,
    LAYER_RELU,
    LAYER_SOFTMAX,
    LAYER_FLATTEN,
    LAYER_CONV2D,
    LAYER_MAXPOOL
} LayerType;

// --- Structs for parameters of each layer type ---

// Dense (Fully Connected) layer parameters
typedef struct {
    Tensor* weights;      // Weights
    Tensor* biases;       // Biases
    Tensor* grad_weights; // Gradient with respect to weights
    Tensor* grad_biases;  // Gradient with respect to biases
} DenseParams;

// Flatten layer parameters (to restore shape during backward pass)
typedef struct {
    int input_shape[4]; // Shape of the input tensor (e.g., [N, C, H, W])
    int input_dims;     // Number of dimensions of the input tensor
} FlattenParams;

// Conv2D layer parameters
typedef struct {
    Tensor* weights;      // Weights (filters)
    Tensor* biases;       // Biases
    Tensor* grad_weights; // Gradient with respect to weights
    Tensor* grad_biases;  // Gradient with respect to biases
    int stride;           // Stride
    int padding;          // Padding
} Conv2DParams;

// MaxPool layer parameters
typedef struct {
    int pool_size;      // Size of the pooling window
    int stride;         // Stride
    Tensor* max_indices;// Indices of the max values for backward pass
} MaxPoolParams;

// --- Main Layer struct ---
typedef struct Layer {
    LayerType type; // Layer type
    union {
        DenseParams dense;
        FlattenParams flatten;
        Conv2DParams conv2d;
        MaxPoolParams maxpool;
    } params;

    // Store input and output for the backward pass
    Tensor* input;
    Tensor* output;
} Layer;

// --- Main Model struct ---
typedef struct {
    Layer* layers;
    int num_layers;
} Model;


// --- Function Prototypes ---

// Model management functions
void model_init(Model* model);
void model_free(Model* model);

// Functions to add layers
void model_add_dense(Model* model, int input_size, int output_size);
void model_add_activation(Model* model, LayerType type);
void model_add_flatten(Model* model);
void model_add_conv2d(Model* model, int in_channels, int out_channels, int kernel_size, int stride, int padding);
void model_add_maxpool(Model* model, int pool_size, int stride);

// Training and inference functions
Tensor* model_forward(Model* model, Tensor* input, int training);
void model_backward(Model* model, Tensor* y_pred, Tensor* y_true);
void model_update_params(Model* model, float learning_rate, int batch_size);

#endif // !__CNN_H__
