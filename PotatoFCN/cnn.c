#include "cnn.h"
#include "math_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// Backward pass function prototypes
static Tensor* backward_dense(Layer* l, Tensor* grad);
static Tensor* backward_relu(Layer* l, Tensor* grad);
static Tensor* backward_flatten(Layer* l, Tensor* grad);
static Tensor* backward_conv2d(Layer* l, Tensor* grad);
static Tensor* backward_maxpool(Layer* l, Tensor* grad);


// --- Model initialization and memory management ---
void model_init(Model* model) {
    model->layers = NULL;
    model->num_layers = 0;
}

void model_add_layer(Model* model, Layer layer) {
    model->num_layers++;
    model->layers = (Layer*)realloc(model->layers, model->num_layers * sizeof(Layer));
    model->layers[model->num_layers - 1] = layer;
}

void model_free(Model* model) {
    for (int i = 0; i < model->num_layers; i++) {
        Layer* l = &model->layers[i];
        if (l->type == LAYER_DENSE) {
            free_tensor(l->params.dense.weights);
            free_tensor(l->params.dense.biases);
            if (l->params.dense.grad_weights) free_tensor(l->params.dense.grad_weights);
            if (l->params.dense.grad_biases) free_tensor(l->params.dense.grad_biases);
        }
        else if (l->type == LAYER_CONV2D) {
            free_tensor(l->params.conv2d.weights);
            free_tensor(l->params.conv2d.biases);
            if (l->params.conv2d.grad_weights) free_tensor(l->params.conv2d.grad_weights);
            if (l->params.conv2d.grad_biases) free_tensor(l->params.conv2d.grad_biases);
        }
        else if (l->type == LAYER_MAXPOOL) {
            if (l->params.maxpool.max_indices) free_tensor(l->params.maxpool.max_indices);
        }
        if (l->input) free_tensor(l->input);
        if (l->output) free_tensor(l->output);
    }
    free(model->layers);
}


// --- Functions to add layers ---

void model_add_dense(Model* model, int input_size, int output_size) {
    Layer l = { .type = LAYER_DENSE, .input = NULL, .output = NULL };
    l.params.dense.weights = create_tensor((int[]) { input_size, output_size }, 2);
    l.params.dense.biases = create_tensor((int[]) { 1, output_size }, 2);
    l.params.dense.grad_weights = NULL;
    l.params.dense.grad_biases = NULL;

    // He initialization
    float scale = sqrtf(2.0f / input_size);
    for (int i = 0; i < get_tensor_size(l.params.dense.weights); i++) {
        l.params.dense.weights->values[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
    model_add_layer(model, l);
}

void model_add_activation(Model* model, LayerType type) {
    assert(type == LAYER_RELU || type == LAYER_SOFTMAX);
    Layer l = { .type = type, .input = NULL, .output = NULL };
    model_add_layer(model, l);
}

void model_add_flatten(Model* model) {
    Layer l = { .type = LAYER_FLATTEN, .input = NULL, .output = NULL };
    model_add_layer(model, l);
}

void model_add_conv2d(Model* model, int in_channels, int out_channels, int kernel_size, int stride, int padding) {
    Layer l = { .type = LAYER_CONV2D, .input = NULL, .output = NULL };
    l.params.conv2d.stride = stride;
    l.params.conv2d.padding = padding;
    l.params.conv2d.weights = create_tensor((int[]) { out_channels, in_channels, kernel_size, kernel_size }, 4);
    l.params.conv2d.biases = create_tensor((int[]) { out_channels }, 1);
    l.params.conv2d.grad_weights = NULL;
    l.params.conv2d.grad_biases = NULL;

    // He initialization
    float scale = sqrtf(2.0f / (in_channels * kernel_size * kernel_size));
    for (int i = 0; i < get_tensor_size(l.params.conv2d.weights); i++) {
        l.params.conv2d.weights->values[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
    model_add_layer(model, l);
}

void model_add_maxpool(Model* model, int pool_size, int stride) {
    Layer l = { .type = LAYER_MAXPOOL, .input = NULL, .output = NULL };
    l.params.maxpool.pool_size = pool_size;
    l.params.maxpool.stride = stride;
    l.params.maxpool.max_indices = NULL;
    model_add_layer(model, l);
}


// --- Forward and backward pass ---

Tensor* model_forward(Model* model, Tensor* input, int training) {
    Tensor* current_activation = copy_tensor(input);

    for (int i = 0; i < model->num_layers; i++) {
        Layer* l = &model->layers[i];
        if (training) {
            if (l->input) free_tensor(l->input);
            l->input = copy_tensor(current_activation);
        }

        Tensor* next_activation = NULL;
        switch (l->type) {
        case LAYER_DENSE: {
            Tensor* y = tensor_dot(current_activation, l->params.dense.weights);
            next_activation = tensor_add_broadcast(y, l->params.dense.biases);
            free_tensor(y);
            break;
        }
        case LAYER_RELU: {
            next_activation = copy_tensor(current_activation);
            relu(next_activation);
            break;
        }
        case LAYER_SOFTMAX: {
            next_activation = copy_tensor(current_activation);
            softmax(next_activation);
            break;
        }
        case LAYER_FLATTEN: {
            // [BUG FIX] Store input shape for backward pass
            if (training) {
                l->params.flatten.input_dims = current_activation->dims;
                for (int j = 0; j < current_activation->dims; ++j) {
                    l->params.flatten.input_shape[j] = current_activation->shape[j];
                }
            }
            next_activation = copy_tensor(current_activation);
            tensor_reshape(next_activation, (int[]) { next_activation->shape[0], get_tensor_size(next_activation) / next_activation->shape[0] }, 2);
            break;
        }
        case LAYER_CONV2D: {
            next_activation = tensor_conv2d(current_activation, l->params.conv2d.weights, l->params.conv2d.biases, l->params.conv2d.stride, l->params.conv2d.padding);
            break;
        }
        case LAYER_MAXPOOL: {
            if (training) {
                if (l->params.maxpool.max_indices) free_tensor(l->params.maxpool.max_indices);
                l->params.maxpool.max_indices = NULL;
                next_activation = tensor_maxpool(current_activation, l->params.maxpool.pool_size, l->params.maxpool.stride, &l->params.maxpool.max_indices);
            }
            else {
                next_activation = tensor_maxpool(current_activation, l->params.maxpool.pool_size, l->params.maxpool.stride, NULL);
            }
            break;
        }
        }

        free_tensor(current_activation);
        current_activation = next_activation;

        if (training) {
            if (l->output) free_tensor(l->output);
            l->output = copy_tensor(current_activation);
        }
    }
    return current_activation;
}

void model_backward(Model* model, Tensor* y_pred, Tensor* y_true) {
    // The gradient of Softmax with Cross-Entropy Loss is simply (y_pred - y_true).
    Tensor* grad = tensor_sub(y_pred, y_true);

    for (int i = model->num_layers - 1; i >= 0; i--) {
        Layer* l = &model->layers[i];
        Tensor* grad_upstream = NULL;

        switch (l->type) {
        case LAYER_DENSE:   grad_upstream = backward_dense(l, grad); break;
        case LAYER_RELU:    grad_upstream = backward_relu(l, grad); break;
        case LAYER_FLATTEN: grad_upstream = backward_flatten(l, grad); break;
        case LAYER_CONV2D:  grad_upstream = backward_conv2d(l, grad); break;
        case LAYER_MAXPOOL: grad_upstream = backward_maxpool(l, grad); break;
            // The Softmax layer is handled by the loss function, so the gradient is passed through.
        case LAYER_SOFTMAX: grad_upstream = copy_tensor(grad); break;
        }

        free_tensor(grad);
        grad = grad_upstream;
    }
    if (grad) free_tensor(grad);
}

void model_update_params(Model* model, float learning_rate, int batch_size) {
    for (int i = 0; i < model->num_layers; i++) {
        if (model->layers[i].type == LAYER_DENSE) {
            DenseParams* dp = &model->layers[i].params.dense;
            tensor_update(dp->weights, dp->grad_weights, learning_rate, batch_size);
            tensor_update(dp->biases, dp->grad_biases, learning_rate, batch_size);
            free_tensor(dp->grad_weights); dp->grad_weights = NULL;
            free_tensor(dp->grad_biases);  dp->grad_biases = NULL;
        }
        else if (model->layers[i].type == LAYER_CONV2D) {
            Conv2DParams* cp = &model->layers[i].params.conv2d;
            tensor_update(cp->weights, cp->grad_weights, learning_rate, batch_size);
            tensor_update(cp->biases, cp->grad_biases, learning_rate, batch_size);
            free_tensor(cp->grad_weights); cp->grad_weights = NULL;
            free_tensor(cp->grad_biases);  cp->grad_biases = NULL;
        }
    }
}

// --- Backward pass for each layer type ---

static Tensor* backward_conv2d(Layer* l, Tensor* grad) {
    Conv2DParams* cp = &l->params.conv2d;
    Tensor* input = l->input;

    // Free previous gradients if they exist
    if (cp->grad_biases) free_tensor(cp->grad_biases);
    if (cp->grad_weights) free_tensor(cp->grad_weights);

    // 1. Calculate bias gradient (dL/dB)
    cp->grad_biases = tensor_conv_grad_bias(grad);

    // 2. Calculate weight gradient (dL/dW)
    cp->grad_weights = tensor_conv_grad_weights(input, grad, cp->stride, cp->padding);

    // 3. Calculate input gradient (dL/dX) - to be passed to the previous layer
    Tensor* grad_upstream = tensor_conv_grad_input(grad, cp->weights, cp->stride, cp->padding, input->shape);

    return grad_upstream;
}

static Tensor* backward_dense(Layer* l, Tensor* grad) {
    DenseParams* dp = &l->params.dense;

    // Free previous gradients if they exist
    if (dp->grad_weights) free_tensor(dp->grad_weights);
    if (dp->grad_biases) free_tensor(dp->grad_biases);

    // 1. Calculate weight gradient: dL/dW = X^T * dL/dY
    Tensor* XT = tensor_transpose(l->input);
    dp->grad_weights = tensor_dot(XT, grad);
    free_tensor(XT);

    // 2. Calculate bias gradient: dL/db = sum(dL/dY)
    dp->grad_biases = tensor_sum_along_axis(grad, 0);

    // 3. Calculate gradient to pass to the previous layer: dL/dX = dL/dY * W^T
    Tensor* WT = tensor_transpose(dp->weights);
    Tensor* grad_upstream = tensor_dot(grad, WT);
    free_tensor(WT);

    return grad_upstream;
}

static Tensor* backward_relu(Layer* l, Tensor* grad) {
    // dL/dX = dL/dY * (dY/dX)
    // The derivative of ReLU (dY/dX) is 1 if X > 0, and 0 otherwise.
    Tensor* deriv = copy_tensor(l->input);
    relu_derivative(deriv);
    Tensor* grad_upstream = tensor_elemwise_mul(grad, deriv);
    free_tensor(deriv);
    return grad_upstream;
}

static Tensor* backward_flatten(Layer* l, Tensor* grad) {
    // Reshape the gradient to the shape before flattening.
    Tensor* grad_upstream = copy_tensor(grad);
    tensor_reshape(grad_upstream, l->params.flatten.input_shape, l->params.flatten.input_dims);
    return grad_upstream;
}

static Tensor* backward_maxpool(Layer* l, Tensor* grad) {
    // Use the stored max indices to route the gradient to the correct locations.
    Tensor* grad_upstream = tensor_maxpool_backward(grad, l->input, l->params.maxpool.max_indices);
    return grad_upstream;
}
