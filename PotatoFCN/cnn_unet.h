#ifndef __CNN_UNET_H__
#define __CNN_UNET_H__

#include "tensor.h"
#include "conv_bwd_data_primitive.h"
#include "conv_bwd_weights_primitive.h"

#include <stdbool.h>

// --- Structure Definitions ---
typedef enum {
    LAYER_CONV2D,
    LAYER_RELU,
    LAYER_TRANSPOSED_CONV2D,
    LAYER_SIGMOID,
    LAYER_BATCHNORM
} LayerType;

typedef struct {
    LayerType type;
    Tensor *weights, *biases;
    Tensor *grad_weights, *grad_biases;
    int stride, padding;

    Tensor* batch_mean;
    Tensor* batch_var;
    Tensor* running_mean;
    Tensor* running_var;

    Tensor* gamma;
    Tensor* beta;
    Tensor* grad_gamma;
    Tensor* grad_beta;
    
    // Fields for storing the input and output of each layer
    Tensor *input;
    Tensor *output;
    Tensor* m_w, * v_w;    // 1st, 2nd moment for weights
    Tensor* m_b, * v_b;
} Layer;
typedef struct {
    Layer* layers;
    int    num_layers;
    // oneDNN backward primitives (lazy init ¿ë)
    ConvBwdWeightsPrimitive* conv_w_prims;
    ConvBwdDataPrimitive* conv_d_prims;
    ConvBwdWeightsPrimitive* deconv_w_prims;
    ConvBwdDataPrimitive* deconv_d_prims;
    // init flag
    bool* prim_inited;
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

void unet_update_adam(
    UNetModel* model,
    float lr,
    float beta1,
    float beta2,
    float eps
);

// --- Function Prototypes ---
void unet_build(UNetModel* model);
void unet_free(UNetModel* model);
UNetIntermediates* unet_forward(UNetModel* model, Tensor* input);

// [New Function Declaration]
UNetIntermediates* unet_forward(UNetModel* model, Tensor* input);
void unet_backward(UNetModel* model, UNetIntermediates* im, Tensor* grad_start);
// ...

// [New Function Declaration]
static Tensor* backward_block(Block* b, Tensor* grad, Tensor** col_workspace);
void unet_update_params(UNetModel* model, float learning_rate);
void unet_zero_grads(UNetModel* model);

// Additional function for freeing intermediate tensors
void unet_free_intermediates(UNetIntermediates* im);

static void clip_block_gradients(Block* b, float clip_val);
void clip_gradients(UNetModel* m, float clip_val);

#endif // __CNN_UNET_H__