#include "cnn_unet.h"
#include "math_utils.h"
#include "tensor.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

static void init_weights(Tensor* t) {
    int fan_in = (t->dims > 1) ? t->shape[1] * t->shape[2] * t->shape[3] : t->shape[0];
    float scale = sqrtf(2.0f / fan_in);
    for (int i = 0; i < get_tensor_size(t); i++) {
        t->values[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
}

static void block_add_layer(Block* b, Layer l) {
    b->num_layers++;
    Layer* new_layers = (Layer*)realloc(b->layers, b->num_layers * sizeof(Layer));
    assert(new_layers != NULL && "Failed to reallocate memory for layers");
    b->layers = new_layers;
    b->layers[b->num_layers - 1] = l;
}

static void add_conv(Block* b, int in_c, int out_c, int k, int s, int p) {
    Layer l = { .type = LAYER_CONV2D, .stride = s, .padding = p, .input = NULL, .output = NULL };
    l.weights = create_tensor((int[]) { out_c, in_c, k, k }, 4);
    l.biases = create_tensor((int[]) { out_c }, 1);
    l.grad_weights = create_tensor((int[]) { out_c, in_c, k, k }, 4);
    l.grad_biases = create_tensor((int[]) { out_c }, 1);
    init_weights(l.weights);
    block_add_layer(b, l);
}

static void add_transposed_conv(Block* b, int in_c, int out_c, int k, int s) {
    Layer l = { .type = LAYER_TRANSPOSED_CONV2D, .stride = s, .padding = 0, .input = NULL, .output = NULL };
    l.weights = create_tensor((int[]) { in_c, out_c, k, k }, 4);
    l.biases = create_tensor((int[]) { out_c }, 1);
    l.grad_weights = create_tensor((int[]) { in_c, out_c, k, k }, 4);
    l.grad_biases = create_tensor((int[]) { out_c }, 1);
    init_weights(l.weights);
    block_add_layer(b, l);
}

static void add_relu(Block* b) {
    block_add_layer(b, (Layer) { .type = LAYER_RELU, .weights = NULL, .biases = NULL, .grad_weights = NULL, .grad_biases = NULL, .input = NULL, .output = NULL });
}

static void block_init(Block* b) {
    b->layers = NULL;
    b->num_layers = 0;
}

static void block_free(Block* b) {
    for (int i = 0; i < b->num_layers; ++i) {
        if (b->layers[i].weights) free_tensor(b->layers[i].weights);
        if (b->layers[i].biases) free_tensor(b->layers[i].biases);
        if (b->layers[i].grad_weights) free_tensor(b->layers[i].grad_weights);
        if (b->layers[i].grad_biases) free_tensor(b->layers[i].grad_biases);
        if (b->layers[i].input) free_tensor(b->layers[i].input);
        if (b->layers[i].output) free_tensor(b->layers[i].output);
    }
    if (b->layers) free(b->layers);
}

void unet_free(UNetModel* model) {
    block_free(&model->enc1);
    block_free(&model->enc2);
    block_free(&model->bottleneck);
    block_free(&model->dec_up1);
    block_free(&model->dec_conv1);
    block_free(&model->dec_up2);
    block_free(&model->dec_conv2);
    block_free(&model->final_conv);
}

void unet_free_intermediates(UNetIntermediates* im) {
    if (im) {
        if (im->enc1_out) free_tensor(im->enc1_out);
        if (im->enc2_out) free_tensor(im->enc2_out);
        if (im->pool1_idx) free_tensor(im->pool1_idx);
        if (im->pool2_idx) free_tensor(im->pool2_idx);
        if (im->pred_mask) free_tensor(im->pred_mask);
        free(im);
    }
}

// --- ¸ðµ¨ ±¸Á¶ ºôµå ÇÔ¼ö ---
void unet_build(UNetModel* model) {
    int in_channels = 1;

    // Encoder Block 1
    block_init(&model->enc1);
    add_conv(&model->enc1, in_channels, 16, 3, 1, 1);
    add_relu(&model->enc1);
    add_conv(&model->enc1, 16, 16, 3, 1, 1);
    add_relu(&model->enc1);

    // Encoder Block 2
    block_init(&model->enc2);
    add_conv(&model->enc2, 16, 32, 3, 1, 1);
    add_relu(&model->enc2);
    add_conv(&model->enc2, 32, 32, 3, 1, 1);
    add_relu(&model->enc2);

    // Bottleneck
    block_init(&model->bottleneck);
    add_conv(&model->bottleneck, 32, 64, 3, 1, 1);
    add_relu(&model->bottleneck);

    // Decoder Block 1 (Upsampling + Convolution)
    block_init(&model->dec_up1);
    add_transposed_conv(&model->dec_up1, 64, 32, 2, 2);
    add_relu(&model->dec_up1);

    block_init(&model->dec_conv1);
    add_conv(&model->dec_conv1, 64, 32, 3, 1, 1); // Concat: 32(up) + 32(skip)
    add_relu(&model->dec_conv1);
    add_conv(&model->dec_conv1, 32, 32, 3, 1, 1);
    add_relu(&model->dec_conv1);

    // Decoder Block 2 (Upsampling + Convolution)
    block_init(&model->dec_up2);
    add_transposed_conv(&model->dec_up2, 32, 16, 2, 2);

    block_init(&model->dec_conv2);
    add_conv(&model->dec_conv2, 32, 16, 3, 1, 1); // Concat: 16(up) + 16(skip)
    add_relu(&model->dec_conv2);
    add_conv(&model->dec_conv2, 16, 16, 3, 1, 1);
    add_relu(&model->dec_conv2);

    // Final Layer
    block_init(&model->final_conv);
    add_conv(&model->final_conv, 16, 1, 1, 1, 0); // 1x1 Convolution
}

static Tensor* forward_block(Block* b, Tensor* input, Tensor** col_workspace_ptr) {
    Tensor* current = copy_tensor(input);
    for (int i = 0; i < b->num_layers; ++i) {
        Layer* l = &b->layers[i];

        if (l->input) free_tensor(l->input);
        l->input = copy_tensor(current);

        Tensor* next = NULL;
        switch (l->type) {
        case LAYER_CONV2D: {
            next = tensor_conv2d(current, l->weights, l->biases, l->stride, l->padding, col_workspace_ptr);
            break;
        }
        case LAYER_TRANSPOSED_CONV2D:
            next = tensor_transposed_conv2d(current, l->weights, l->biases, l->stride);
            break;
        case LAYER_RELU:
            next = copy_tensor(current);
            relu(next);
            break;
        }

        if (l->output) free_tensor(l->output);
        l->output = copy_tensor(next);

        free_tensor(current);
        current = next;
    }
    return current;
}


static Tensor* backward_block(Block* b, Tensor* grad, Tensor** col_workspace_ptr) {
    Tensor* current_grad = copy_tensor(grad);
    for (int i = b->num_layers - 1; i >= 0; --i) {
        Layer* l = &b->layers[i];
        Tensor* next_grad = NULL;

        switch (l->type) {
        case LAYER_RELU: {
            Tensor* deriv = copy_tensor(l->input);
            relu_derivative(deriv);
            next_grad = tensor_elemwise_mul(current_grad, deriv);
            free_tensor(deriv);
            break;
        }
        case LAYER_CONV2D: {
            assert(l->input != NULL && l->weights != NULL);

            Tensor* gw = tensor_conv_grad_weights(l->input, current_grad, l->weights, l->stride, l->padding, col_workspace_ptr);
            Tensor* gb = tensor_conv_grad_bias(current_grad);

            for (int j = 0; j < get_tensor_size(gw); ++j) l->grad_weights->values[j] += gw->values[j];
            for (int j = 0; j < get_tensor_size(gb); ++j) l->grad_biases->values[j] += gb->values[j];

            free_tensor(gw);
            free_tensor(gb);

            next_grad = tensor_conv_grad_input(current_grad, l->weights, l->stride, l->padding, *col_workspace_ptr);
            break;
        }
        case LAYER_TRANSPOSED_CONV2D: {
            assert(l->input != NULL && l->weights != NULL);

            Tensor* gw = tensor_conv_grad_weights(l->input, current_grad, l->weights, l->stride, l->padding, col_workspace_ptr);
            Tensor* gb = tensor_conv_grad_bias(current_grad);

            for (int j = 0; j < get_tensor_size(gw); ++j) l->grad_weights->values[j] += gw->values[j];
            for (int j = 0; j < get_tensor_size(gb); ++j) l->grad_biases->values[j] += gb->values[j];

            free_tensor(gw);
            free_tensor(gb);

            Tensor* zero_bias = create_tensor((int[]) { l->biases->shape[0] }, 1);
            next_grad = tensor_conv2d(current_grad, l->weights, zero_bias, l->stride, l->padding, col_workspace_ptr);
            free_tensor(zero_bias);
            break;
        }
        }
        free_tensor(current_grad);
        current_grad = next_grad;
    }
    return current_grad;
}


UNetIntermediates* unet_forward(UNetModel* model, Tensor* input) {
    UNetIntermediates* im = (UNetIntermediates*)calloc(1, sizeof(UNetIntermediates));
    assert(im != NULL);

    Tensor* col_workspace = NULL;

    Tensor* enc1_out_full = forward_block(&model->enc1, input, &col_workspace);
    im->enc1_out = copy_tensor(enc1_out_full);
    Tensor* pool1_out = tensor_maxpool(enc1_out_full, 2, 2, &im->pool1_idx);
    free_tensor(enc1_out_full);

    Tensor* enc2_out_full = forward_block(&model->enc2, pool1_out, &col_workspace);
    im->enc2_out = copy_tensor(enc2_out_full);
    Tensor* pool2_out = tensor_maxpool(enc2_out_full, 2, 2, &im->pool2_idx);
    free_tensor(pool1_out);
    free_tensor(enc2_out_full);

    Tensor* bottleneck_out = forward_block(&model->bottleneck, pool2_out, &col_workspace);
    free_tensor(pool2_out);

    Tensor* dec1_up_out = forward_block(&model->dec_up1, bottleneck_out, &col_workspace);
    Tensor* concat1 = tensor_concatenate(dec1_up_out, im->enc2_out, 1);
    Tensor* dec1_conv_out = forward_block(&model->dec_conv1, concat1, &col_workspace);
    free_tensor(bottleneck_out);
    free_tensor(dec1_up_out);
    free_tensor(concat1);

    Tensor* dec2_up_out = forward_block(&model->dec_up2, dec1_conv_out, &col_workspace);
    Tensor* concat2 = tensor_concatenate(dec2_up_out, im->enc1_out, 1);
    Tensor* dec2_conv_out = forward_block(&model->dec_conv2, concat2, &col_workspace);
    free_tensor(dec1_conv_out);
    free_tensor(dec2_up_out);
    free_tensor(concat2);

    im->pred_mask = forward_block(&model->final_conv, dec2_conv_out, &col_workspace);
    free_tensor(dec2_conv_out);

    if (col_workspace) free_tensor(col_workspace);

    return im;
}


void unet_backward(UNetModel* model, UNetIntermediates* im, Tensor* grad_start) {
    Tensor* col_workspace = NULL;

    // Final Conv
    Tensor* current_grad = backward_block(&model->final_conv, grad_start, &col_workspace);

    // --- Decoder 2 ---
    Tensor* grad_concat2 = backward_block(&model->dec_conv2, current_grad, &col_workspace);
    free_tensor(current_grad);

    // Concat
    Tensor* grad_dec2_up = create_tensor(model->dec_up2.layers[0].output->shape, model->dec_up2.layers[0].output->dims);
    Tensor* grad_enc1_skip = create_tensor(im->enc1_out->shape, im->enc1_out->dims);
    tensor_concatenate_backward(grad_dec2_up, grad_enc1_skip, grad_concat2, 1, grad_dec2_up->shape[1]);
    free_tensor(grad_concat2);

    current_grad = backward_block(&model->dec_up2, grad_dec2_up, &col_workspace);
    free_tensor(grad_dec2_up);

    // --- Decoder 1 ---
    Tensor* grad_concat1 = backward_block(&model->dec_conv1, current_grad, &col_workspace);
    free_tensor(current_grad);

    Tensor* grad_dec1_up = create_tensor(model->dec_up1.layers[0].output->shape, model->dec_up1.layers[0].output->dims);
    Tensor* grad_enc2_skip = create_tensor(im->enc2_out->shape, im->enc2_out->dims);
    tensor_concatenate_backward(grad_dec1_up, grad_enc2_skip, grad_concat1, 1, grad_dec1_up->shape[1]);
    free_tensor(grad_concat1);

    current_grad = backward_block(&model->dec_up1, grad_dec1_up, &col_workspace);
    free_tensor(grad_dec1_up);

    // --- Bottleneck ---
    current_grad = backward_block(&model->bottleneck, current_grad, &col_workspace);

    // --- Encoder 2 ---
    Tensor* grad_from_pool2 = tensor_maxpool_backward(current_grad, im->enc2_out, im->pool2_idx);
    free_tensor(current_grad);
    // Max-pool
    for (int i = 0; i < get_tensor_size(grad_from_pool2); ++i) {
        grad_from_pool2->values[i] += grad_enc2_skip->values[i];
    }
    free_tensor(grad_enc2_skip);

    current_grad = backward_block(&model->enc2, grad_from_pool2, &col_workspace);
    free_tensor(grad_from_pool2);

    // --- Encoder 1 ---
    Tensor* grad_from_pool1 = tensor_maxpool_backward(current_grad, im->enc1_out, im->pool1_idx);
    free_tensor(current_grad);
    // Max-pool
    for (int i = 0; i < get_tensor_size(grad_from_pool1); ++i) {
        grad_from_pool1->values[i] += grad_enc1_skip->values[i];
    }
    free_tensor(grad_enc1_skip);

    current_grad = backward_block(&model->enc1, grad_from_pool1, &col_workspace);
    free_tensor(grad_from_pool1);

    free_tensor(current_grad);

    if (col_workspace) {
        free_tensor(col_workspace);
    }
}


static void update_block(Block* b, float lr) {
    for (int i = 0; i < b->num_layers; ++i) {
        if (b->layers[i].weights) {
            int w_size = get_tensor_size(b->layers[i].weights);
            int b_size = get_tensor_size(b->layers[i].biases);
            for (int j = 0; j < w_size; ++j) {
                b->layers[i].weights->values[j] -= lr * b->layers[i].grad_weights->values[j];
            }
            for (int j = 0; j < b_size; ++j) {
                b->layers[i].biases->values[j] -= lr * b->layers[i].grad_biases->values[j];
            }
        }
    }
}

void unet_update_params(UNetModel* model, float learning_rate) {
    update_block(&model->enc1, learning_rate);
    update_block(&model->enc2, learning_rate);
    update_block(&model->bottleneck, learning_rate);
    update_block(&model->dec_up1, learning_rate);
    update_block(&model->dec_conv1, learning_rate);
    update_block(&model->dec_up2, learning_rate);
    update_block(&model->dec_conv2, learning_rate);
    update_block(&model->final_conv, learning_rate);
}

static void zero_grads_block(Block* b) {
    for (int i = 0; i < b->num_layers; ++i) {
        if (b->layers[i].grad_weights) {
            memset(b->layers[i].grad_weights->values, 0, get_tensor_size(b->layers[i].grad_weights) * sizeof(float));
            memset(b->layers[i].grad_biases->values, 0, get_tensor_size(b->layers[i].grad_biases) * sizeof(float));
        }
    }
}

void unet_zero_grads(UNetModel* model) {
    zero_grads_block(&model->enc1);
    zero_grads_block(&model->enc2);
    zero_grads_block(&model->bottleneck);
    zero_grads_block(&model->dec_up1);
    zero_grads_block(&model->dec_conv1);
    zero_grads_block(&model->dec_up2);
    zero_grads_block(&model->dec_conv2);
    zero_grads_block(&model->final_conv);
}