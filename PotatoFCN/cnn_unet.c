#include "cnn_unet.h"
#include "math_utils.h"
#include "tensor.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

static int adam_t = 1;

// 하이퍼파라미터: β₁, β₂, ε 는 호출 시 넘겨주세요.
void unet_update_adam(
    UNetModel* model,
    float lr,
    float beta1,
    float beta2,
    float eps
) {
    // β₁ᵗ, β₂ᵗ 미리 계산
    float bias_correction1 = 1.0f - powf(beta1, adam_t);
    float bias_correction2 = 1.0f - powf(beta2, adam_t);

#define UPDATE_LAYER(L) \
      do { \
        /* weight */ \
        int Wn = get_tensor_size((L)->weights); \
        for (int i = 0; i < Wn; ++i) { \
          float g = (L)->grad_weights->values[i]; \
          /* 1st, 2nd moment 업데이트 */ \
          (L)->m_w->values[i] = beta1 * (L)->m_w->values[i] + (1 - beta1) * g; \
          (L)->v_w->values[i] = beta2 * (L)->v_w->values[i] + (1 - beta2) * (g * g); \
          /* 편향 보정된 모멘트 */ \
          float m_hat = (L)->m_w->values[i] / bias_correction1; \
          float v_hat = (L)->v_w->values[i] / bias_correction2; \
          /* 매개변수 업데이트 */ \
          (L)->weights->values[i] -= lr * m_hat / (sqrtf(v_hat) + eps); \
        } \
        /* bias */ \
        int Bn = get_tensor_size((L)->biases); \
        for (int i = 0; i < Bn; ++i) { \
          float g = (L)->grad_biases->values[i]; \
          (L)->m_b->values[i] = beta1 * (L)->m_b->values[i] + (1 - beta1) * g; \
          (L)->v_b->values[i] = beta2 * (L)->v_b->values[i] + (1 - beta2) * (g * g); \
          float m_hat = (L)->m_b->values[i] / bias_correction1; \
          float v_hat = (L)->v_b->values[i] / bias_correction2; \
          (L)->biases->values[i]  -= lr * m_hat / (sqrtf(v_hat) + eps); \
        } \
      } while (0)

    // 각 블록, 각 레이어에 대해
    for (int i = 0; i < model->enc1.num_layers; ++i) {
        Layer* L = &model->enc1.layers[i];
        if (L->weights) UPDATE_LAYER(L);
    }
    for (int i = 0; i < model->enc2.num_layers; ++i) {
        Layer* L = &model->enc2.layers[i];
        if (L->weights) UPDATE_LAYER(L);
    }
    for (int i = 0; i < model->bottleneck.num_layers; ++i) {
        Layer* L = &model->bottleneck.layers[i];
        if (L->weights) UPDATE_LAYER(L);
    }
    for (int i = 0; i < model->dec_up1.num_layers; ++i) {
        Layer* L = &model->dec_up1.layers[i];
        if (L->weights) UPDATE_LAYER(L);
    }
    for (int i = 0; i < model->dec_conv1.num_layers; ++i) {
        Layer* L = &model->dec_conv1.layers[i];
        if (L->weights) UPDATE_LAYER(L);
    }
    for (int i = 0; i < model->dec_up2.num_layers; ++i) {
        Layer* L = &model->dec_up2.layers[i];
        if (L->weights) UPDATE_LAYER(L);
    }
    for (int i = 0; i < model->dec_conv2.num_layers; ++i) {
        Layer* L = &model->dec_conv2.layers[i];
        if (L->weights) UPDATE_LAYER(L);
    }
    for (int i = 0; i < model->final_conv.num_layers; ++i) {
        Layer* L = &model->final_conv.layers[i];
        if (L->weights) UPDATE_LAYER(L);
    }

#undef UPDATE_LAYER

    adam_t++;
}

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
    Block* blocks[] = {
        &model->enc1, &model->enc2, &model->bottleneck,
        &model->dec_up1, &model->dec_conv1,
        &model->dec_up2, &model->dec_conv2,
        &model->final_conv
    };
    for (int bi = 0; bi < sizeof(blocks) / sizeof(*blocks); ++bi) {
        Block* b = blocks[bi];
        for (int i = 0; i < b->num_layers; ++i) {
            Layer* l = &b->layers[i];
            if (b->prim_inited[i]) {
                if (l->type == LAYER_CONV2D) {
                    conv_bwd_weights_primitive_destroy(&b->conv_w_prims[i]);
                    conv_bwd_data_primitive_destroy(&b->conv_d_prims[i]);
                }
                else if (l->type == LAYER_TRANSPOSED_CONV2D) {
                    conv_bwd_weights_primitive_destroy(&b->deconv_w_prims[i]);
                    conv_bwd_data_primitive_destroy(&b->deconv_d_prims[i]);
                }
            }
        }
        free(b->conv_w_prims);
        free(b->conv_d_prims);
        free(b->deconv_w_prims);
        free(b->deconv_d_prims);
        free(b->prim_inited);
        block_free(b);
    }
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

void add_sigmoid(Block* b) {
    block_add_layer(b, (Layer) {
        .type = LAYER_SIGMOID,
            .weights = NULL,
            .biases = NULL,
            .grad_weights = NULL,
            .grad_biases = NULL,
            .input = NULL,
            .output = NULL
    });
}

static void add_batchnorm(Block* b, int channels) {
    Layer l = {
        .type = LAYER_BATCHNORM,
        .weights = NULL, .biases = NULL,
        .grad_weights = NULL, .grad_biases = NULL,
        .input = NULL, .output = NULL
    };

    l.gamma = create_tensor((int[]) { channels }, 1);
    l.beta = create_tensor((int[]) { channels }, 1);
    l.grad_gamma = create_tensor((int[]) { channels }, 1);
    l.grad_beta = create_tensor((int[]) { channels }, 1);
    l.running_mean = create_tensor((int[]) { channels }, 1);
    l.running_var = create_tensor((int[]) { channels }, 1);
    l.batch_mean = create_tensor((int[]) { channels }, 1);
    l.batch_var = create_tensor((int[]) { channels }, 1);
    // gamma=1, beta=0
    for (int i = 0; i < channels; ++i) {
        l.gamma->values[i] = 1.0f;
        l.beta->values[i] = 0.0f;
    }
    // running_* 은 0 초기화 (create_tensor에서 이미 0)
    block_add_layer(b, l);
}




void unet_build(UNetModel* model) {
    int in_channels = 1;

    // Encoder Block 1
    block_init(&model->enc1);
    add_conv(&model->enc1, in_channels, 16, 3, 1, 1);
    add_batchnorm(&model->enc1, 16);
    add_relu(&model->enc1);
    add_conv(&model->enc1, 16, 16, 3, 1, 1);
    add_batchnorm(&model->enc1, 16);
    add_relu(&model->enc1);

    // Encoder Block 2
    block_init(&model->enc2);
    add_conv(&model->enc2, 16, 32, 3, 1, 1);
    add_batchnorm(&model->enc2, 32);
    add_relu(&model->enc2);
    add_conv(&model->enc2, 32, 32, 3, 1, 1);
    add_batchnorm(&model->enc2, 32);
    add_relu(&model->enc2);

    // Bottleneck
    block_init(&model->bottleneck);
    add_conv(&model->bottleneck, 32, 64, 3, 1, 1);
    add_batchnorm(&model->bottleneck, 64);
    add_relu(&model->bottleneck);

    // Decoder Block 1 (Upsampling + Convolution)
    block_init(&model->dec_up1);
    add_transposed_conv(&model->dec_up1, 64, 32, 2, 2);
    add_batchnorm(&model->dec_up1, 32);
    add_relu(&model->dec_up1);

    block_init(&model->dec_conv1);
    add_conv(&model->dec_conv1, 64, 32, 3, 1, 1); // Concat: 32(up) + 32(skip)
    add_batchnorm(&model->dec_conv1, 32);
    add_relu(&model->dec_conv1);
    add_conv(&model->dec_conv1, 32, 32, 3, 1, 1);
    add_batchnorm(&model->dec_conv1, 32);
    add_relu(&model->dec_conv1);

    // Decoder Block 2 (Upsampling + Convolution)
    block_init(&model->dec_up2);
    add_transposed_conv(&model->dec_up2, 32, 16, 2, 2);
    add_batchnorm(&model->dec_up2, 16);
    add_relu(&model->dec_up2);

    block_init(&model->dec_conv2);
    add_conv(&model->dec_conv2, 32, 16, 3, 1, 1); // Concat: 16(up) + 16(skip)
    add_batchnorm(&model->dec_conv2, 16);
    add_relu(&model->dec_conv2);
    add_conv(&model->dec_conv2, 16, 16, 3, 1, 1);
    add_batchnorm(&model->dec_conv2, 16);
    add_relu(&model->dec_conv2);

    // Final Layer
    block_init(&model->final_conv);
    add_conv(&model->final_conv, 16, 1, 1, 1, 0); // 1x1 Convolution
    add_sigmoid(&model->final_conv);

    // Allocate backward primitives and Adam moments, zero-init
    Block* blocks[] = {
        &model->enc1, &model->enc2, &model->bottleneck,
        &model->dec_up1, &model->dec_conv1,
        &model->dec_up2, &model->dec_conv2,
        &model->final_conv
    };
    int num_blocks = sizeof(blocks) / sizeof(blocks[0]);
    for (int bi = 0; bi < num_blocks; ++bi) {
        Block* b = blocks[bi];
        int L = b->num_layers;
        b->conv_w_prims = calloc(L, sizeof(*b->conv_w_prims));
        b->conv_d_prims = calloc(L, sizeof(*b->conv_d_prims));
        b->deconv_w_prims = calloc(L, sizeof(*b->deconv_w_prims));
        b->deconv_d_prims = calloc(L, sizeof(*b->deconv_d_prims));
        b->prim_inited = calloc(L, sizeof(*b->prim_inited));
        // Initialize Adam moments for each layer
        for (int i = 0; i < L; ++i) {
            Layer* l = &b->layers[i];
            if (l->weights) {
                l->m_w = create_tensor(l->weights->shape, l->weights->dims);
                l->v_w = create_tensor(l->weights->shape, l->weights->dims);
            }
            if (l->biases) {
                l->m_b = create_tensor(l->biases->shape, l->biases->dims);
                l->v_b = create_tensor(l->biases->shape, l->biases->dims);
            }
        }
    }
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
        case LAYER_SIGMOID:
            next = copy_tensor(current);
            sigmoid_inplace(next);
            break;
        case LAYER_BATCHNORM: {
            // 저장된 input
            l->input = copy_tensor(current);
            // 채널별 평균/분산 계산 (training 모드)
            int N = current->shape[0], C = current->shape[1], H = current->shape[2], W = current->shape[3];
            int HW = H * W;
            Tensor* out = create_tensor(current->shape, current->dims);
            for (int c = 0; c < C; ++c) {
                // 1) mean
                float sum = 0;
                for (int n = 0; n < N; ++n)
                    for (int i = 0; i < HW; ++i)
                        sum += current->values[((n * C + c) * H * W) + i];
                float mean = sum / (N * HW);
                // 2) var
                float var = 0;
                for (int n = 0; n < N; ++n)
                    for (int i = 0; i < HW; ++i) {
                        float v = current->values[((n * C + c) * H * W) + i] - mean;
                        var += v * v;
                    }
                var /= (N * HW);
                // running 업데이트 (momentum=0.1)
                l->running_mean->values[c] = 0.9f * l->running_mean->values[c] + 0.1f * mean;
                l->running_var->values[c] = 0.9f * l->running_var->values[c] + 0.1f * var;
                l->batch_mean->values[c] = mean;
                l->batch_var->values[c] = var;
                // 3) normalize + scale+shift
                float inv_std = 1.0f / sqrtf(var + 1e-5f);
                for (int n = 0; n < N; ++n)
                    for (int i = 0; i < HW; ++i) {
                        int idx = (n * C + c) * H * W + i;
                        float normalized = (current->values[idx] - mean) * inv_std;
                        out->values[idx] = l->gamma->values[c] * normalized + l->beta->values[c];
                    }
            }
            next = out;
            break;
        }
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
        case LAYER_SIGMOID: {
            // dL/dx = dL/dy * σ'(x)
            Tensor* deriv = copy_tensor(l->output);  // σ(x)
            sigmoid_derivative_inplace(deriv);
            next_grad = tensor_elemwise_mul(current_grad, deriv);
            free_tensor(deriv);
            break;
        }
        case LAYER_RELU: {
            Tensor* deriv = copy_tensor(l->input);
            relu_derivative(deriv);
            next_grad = tensor_elemwise_mul(current_grad, deriv);
            free_tensor(deriv);
            break;
        }
        case LAYER_BATCHNORM: {
            // Full BatchNorm backward using saved batch_mean/var
            Tensor* dy = copy_tensor(current_grad);
            Tensor* x = l->input;
            int N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
            int M = N * H * W;

            next_grad = create_tensor(x->shape, x->dims);
            for (int c = 0; c < C; ++c) {
                float mean = l->batch_mean->values[c];
                float var = l->batch_var->values[c] + 1e-5f;
                float inv_std = 1.0f / sqrtf(var);

                // Compute gradients dgamma, dbeta
                float dgamma = 0.0f, dbeta = 0.0f;
                for (int n = 0; n < N; ++n) {
                    for (int h = 0; h < H; ++h) {
                        for (int w = 0; w < W; ++w) {
                            int idx = ((n * C + c) * H + h) * W + w;
                            float xv = x->values[idx];
                            float x_norm = (xv - mean) * inv_std;
                            float dyv = dy->values[idx];
                            dgamma += dyv * x_norm;
                            dbeta += dyv;
                        }
                    }
                }
                // accumulate to gamma/beta grads
                l->grad_gamma->values[c] += dgamma;
                l->grad_beta->values[c] += dbeta;

                // Precompute sums for dx
                float sum_dy = 0.0f, sum_dy_xnorm = 0.0f;
                for (int n = 0; n < N; ++n) {
                    for (int h = 0; h < H; ++h) {
                        for (int w = 0; w < W; ++w) {
                            int idx = ((n * C + c) * H + h) * W + w;
                            float xv = x->values[idx];
                            float x_norm = (xv - mean) * inv_std;
                            float dyv = dy->values[idx];
                            sum_dy += dyv;
                            sum_dy_xnorm += dyv * x_norm;
                        }
                    }
                }
                // Compute dx using full formula
                for (int n = 0; n < N; ++n) {
                    for (int h = 0; h < H; ++h) {
                        for (int w = 0; w < W; ++w) {
                            int idx = ((n * C + c) * H + h) * W + w;
                            float xv = x->values[idx];
                            float x_norm = (xv - mean) * inv_std;
                            float dyv = dy->values[idx];
                            float gamma = l->gamma->values[c];
                            float dx = (1.0f / M) * gamma * inv_std *
                                (M * dyv - sum_dy - x_norm * sum_dy_xnorm);
                            next_grad->values[idx] = dx;
                        }
                    }
                }
            }
            free_tensor(dy);
            break;
        }
        case LAYER_CONV2D: {
            Tensor* gw = tensor_conv_grad_weights(l->input, current_grad, l->weights, l->stride, l->padding, col_workspace_ptr);
            Tensor* gb = tensor_conv_grad_bias(current_grad);
            for (int j = 0; j < get_tensor_size(gw); ++j) l->grad_weights->values[j] += gw->values[j];
            for (int j = 0; j < get_tensor_size(gb); ++j) l->grad_biases->values[j] += gb->values[j];
            free_tensor(gw); free_tensor(gb);
            next_grad = tensor_conv_grad_input(current_grad, l->weights, l->stride, l->padding, *col_workspace_ptr);
            break;
        }
        case LAYER_TRANSPOSED_CONV2D: {
            Tensor* gw = tensor_conv_grad_weights(current_grad, l->input, l->weights, l->stride, l->padding, col_workspace_ptr);
            Tensor* gb = tensor_conv_grad_bias(current_grad);
            for (int j = 0; j < get_tensor_size(gw); ++j) l->grad_weights->values[j] += gw->values[j];
            for (int j = 0; j < get_tensor_size(gb); ++j) l->grad_biases->values[j] += gb->values[j];
            free_tensor(gw); free_tensor(gb);
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

static void clip_block_gradients(Block* b, float clip_val) {
    for (int i = 0; i < b->num_layers; ++i) {
        Layer* l = &b->layers[i];
        if (l->grad_weights) {
            int sz = get_tensor_size(l->grad_weights);
            for (int j = 0; j < sz; ++j) {
                if (l->grad_weights->values[j] > clip_val) l->grad_weights->values[j] = clip_val;
                if (l->grad_weights->values[j] < -clip_val) l->grad_weights->values[j] = -clip_val;
            }
        }
        if (l->grad_biases) {
            int sz = get_tensor_size(l->grad_biases);
            for (int j = 0; j < sz; ++j) {
                if (l->grad_biases->values[j] > clip_val) l->grad_biases->values[j] = clip_val;
                if (l->grad_biases->values[j] < -clip_val) l->grad_biases->values[j] = -clip_val;
            }
        }
    }
}

// 전체 모델에 대해 클리핑
void clip_gradients(UNetModel* m, float clip_val) {
    clip_block_gradients(&m->enc1, clip_val);
    clip_block_gradients(&m->enc2, clip_val);
    clip_block_gradients(&m->bottleneck, clip_val);
    clip_block_gradients(&m->dec_up1, clip_val);
    clip_block_gradients(&m->dec_conv1, clip_val);
    clip_block_gradients(&m->dec_up2, clip_val);
    clip_block_gradients(&m->dec_conv2, clip_val);
    clip_block_gradients(&m->final_conv, clip_val);
}