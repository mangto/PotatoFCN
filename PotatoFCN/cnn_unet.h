#ifndef __CNN_UNET_H__
#define __CNN_UNET_H__

#include "tensor.h"

// --- Structure Definitions ---
typedef enum {
    LAYER_CONV2D,
    LAYER_RELU,
    LAYER_TRANSPOSED_CONV2D
} LayerType;

typedef struct {
    LayerType type;
    Tensor *weights, *biases;
    Tensor *grad_weights, *grad_biases;
    int stride, padding;
    
    // Fields for storing the input and output of each layer
    Tensor *input;
    Tensor *output;
} Layer;

typedef struct {
    Layer* layers;
    int num_layers;
} Block;

typedef struct {
    Block enc1, enc2, bottleneck, dec_up1, dec_conv1, dec_up2, dec_conv2, final_conv;
} UNetModel;

// UNetIntermediates structure: Stores tensors needed for skip-connections and backpropagation
typedef struct {
    Tensor *enc1_out;      // For dec2 skip connection
    Tensor *enc2_out;      // For dec1 skip connection
    Tensor *pool1_idx;     // For maxpool backward
    Tensor *pool2_idx;     // For maxpool backward
    Tensor *pred_mask;     // Final prediction result
} UNetIntermediates;


// --- Function Prototypes ---
void unet_build(UNetModel* model);
void unet_free(UNetModel* model);
UNetIntermediates* unet_forward(UNetModel* model, Tensor* input);

// [New Function Declaration]
UNetIntermediates* unet_forward(UNetModel* model, Tensor* input);
void unet_backward(UNetModel* model, UNetIntermediates* im, Tensor* grad_start);
// ...

// [New Function Declaration]
static Tensor* backward_block(Block* b, Tensor* grad, Tensor* col_workspace);
void unet_update_params(UNetModel* model, float learning_rate);
void unet_zero_grads(UNetModel* model);

// Additional function for freeing intermediate tensors
void unet_free_intermediates(UNetIntermediates* im);

#endif // __CNN_UNET_H__