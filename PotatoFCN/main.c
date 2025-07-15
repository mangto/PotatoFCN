#define _CRT_SECURE_NO_WARNINGS
#include "cnn_unet.h"
#include "tensor.h"
#include "conv2d_primitive.h"  // init_dnnl_engine()
#include "utils.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>
#include <math.h>
#include <sys/stat.h>  // for stat()

// ------------------------------------------------------------
// 모델 I/O: save_model / load_model
// ------------------------------------------------------------
static bool file_exists(const char* path) {
    struct stat buf;
    return stat(path, &buf) == 0;
}

void save_model(const UNetModel* model, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) { perror("save_model fopen"); return; }
    // 1) for each block, write num_layers and for each layer its weights & biases
    Block* blocks[] = {
        &model->enc1, &model->enc2, &model->bottleneck,
        &model->dec_up1, &model->dec_conv1,
        &model->dec_up2, &model->dec_conv2,
        &model->final_conv
    };
    int nb = sizeof(blocks) / sizeof(blocks[0]);
    fwrite(&nb, sizeof(nb), 1, f);
    for (int bi = 0; bi < nb; ++bi) {
        Block* b = blocks[bi];
        fwrite(&b->num_layers, sizeof(b->num_layers), 1, f);
        for (int i = 0; i < b->num_layers; ++i) {
            Layer* l = &b->layers[i];
            fwrite(&l->type, sizeof(l->type), 1, f);
            // weights
            bool hasW = (l->weights != NULL);
            fwrite(&hasW, sizeof(hasW), 1, f);
            if (hasW) {
                fwrite(&l->weights->dims, sizeof(int), 1, f);
                fwrite(l->weights->shape, sizeof(int), l->weights->dims, f);
                int sz = get_tensor_size(l->weights);
                fwrite(l->weights->values, sizeof(float), sz, f);
            }
            // biases
            bool hasB = (l->biases != NULL);
            fwrite(&hasB, sizeof(hasB), 1, f);
            if (hasB) {
                fwrite(&l->biases->dims, sizeof(int), 1, f);
                fwrite(l->biases->shape, sizeof(int), l->biases->dims, f);
                int sz = get_tensor_size(l->biases);
                fwrite(l->biases->values, sizeof(float), sz, f);
            }
        }
    }
    fclose(f);
    printf("Model saved to '%s'\n", path);
}

void load_model(UNetModel* model, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror("load_model fopen"); return; }
    Block* blocks[] = {
        &model->enc1, &model->enc2, &model->bottleneck,
        &model->dec_up1, &model->dec_conv1,
        &model->dec_up2, &model->dec_conv2,
        &model->final_conv
    };
    int nb = 0;
    fread(&nb, sizeof(nb), 1, f);
    for (int bi = 0; bi < nb; ++bi) {
        Block* b = blocks[bi];
        int L = 0;
        fread(&L, sizeof(L), 1, f);
        assert(L == b->num_layers);
        for (int i = 0; i < L; ++i) {
            Layer* l = &b->layers[i];
            fread(&l->type, sizeof(l->type), 1, f);
            // weights
            bool hasW;
            fread(&hasW, sizeof(hasW), 1, f);
            if (hasW) {
                int dims; fread(&dims, sizeof(dims), 1, f);
                int shape[4];
                fread(shape, sizeof(int), dims, f);
                int sz = 1; for (int d = 0; d < dims; d++) sz *= shape[d];
                assert(l->weights);
                memcpy(l->weights->shape, shape, dims * sizeof(int));
                fread(l->weights->values, sizeof(float), sz, f);
            }
            // biases
            bool hasB;
            fread(&hasB, sizeof(hasB), 1, f);
            if (hasB) {
                int dims; fread(&dims, sizeof(dims), 1, f);
                int shape[1];
                fread(shape, sizeof(int), dims, f);
                int sz = shape[0];
                assert(l->biases);
                memcpy(l->biases->shape, shape, dims * sizeof(int));
                fread(l->biases->values, sizeof(float), sz, f);
            }
        }
    }
    fclose(f);
    printf("Model loaded from '%s'\n", path);
}

// ------------------------------------------------------------
// Updated print_progress: adds elapsed time
// ------------------------------------------------------------
void print_progress(int current, int total, float loss, int elapsed_sec) {
    const int bar_width = 50;
    float p = (float)current / total;
    int pos = (int)(bar_width * p);
    int h = elapsed_sec / 3600;
    int m = (elapsed_sec % 3600) / 60;
    int s = elapsed_sec % 60;

    printf("\r[");
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos)      putchar('=');
        else if (i == pos)putchar('>');
        else              putchar(' ');
    }
    printf("] %4d/%4d (%.1f%%) - Loss: %.6f - Time: %02d:%02d:%02d",
        current, total, p * 100.0f, loss, h, m, s);
    fflush(stdout);
}

#define M_PI 3.14159265358979

float get_lr(int step, int total_steps, float base_lr) {
    const int warmup = 500;
    if (step < warmup) {
        // 선형 워밍업: 0 → base_lr
        return base_lr * ((float)(step + 1) / (float)warmup);
    }
    // 워밍업 이후 코사인 디케이
    float t = (float)(step - warmup) / (float)(total_steps - warmup);
    return base_lr * 0.5f * (1.0f + cosf((float)M_PI * t));
}

//----------------------------------------------------------------------
// Main
//----------------------------------------------------------------------
int main() {
    srand((unsigned)time(NULL));

    // 1) Load data
    int num_samples, IMG_H, IMG_W;
    float* all_images, * all_masks;
    load_preprocessed_data("./preprocessed_data",
        &num_samples, &IMG_H, &IMG_W, &all_images, &all_masks);

    // 2) Build model
    UNetModel model;
    unet_build(&model);

    // Try to load existing
    const char* model_path = "unet_model_adam.bin";
    if (file_exists(model_path)) {
        init_dnnl_engine();
        load_model(&model, model_path);
    }
    else {
        init_dnnl_engine();
    }

    // 3) Training settings
    const int EPOCHS = 5;
    const int BATCH_SIZE = 8;
    Tensor* input_batch = create_tensor((int[]) { BATCH_SIZE, 1, IMG_H, IMG_W }, 4);
    Tensor* target_batch = create_tensor((int[]) { BATCH_SIZE, 1, IMG_H, IMG_W }, 4);
    int* indices = malloc(num_samples * sizeof(int));
    for (int i = 0; i < num_samples; ++i) indices[i] = i;

    int steps_per_epoch = (num_samples + BATCH_SIZE - 1) / BATCH_SIZE;
    int total_steps = EPOCHS * steps_per_epoch;
    int global_step = 0;


    printf("\n--- Starting UNet Training (batch size = %d) ---\n", BATCH_SIZE);
    for (int e = 0; e < EPOCHS; ++e) {
        // shuffle
        for (int i = num_samples - 1; i > 0; --i) {
            int j = rand() % (i + 1);
            int t = indices[i]; indices[i] = indices[j]; indices[j] = t;
        }
        printf("Epoch %d/%d\n", e + 1, EPOCHS);

        time_t epoch_start = time(NULL);
        float epoch_loss = 0.0f;
        int steps = (num_samples + BATCH_SIZE - 1) / BATCH_SIZE;

        for (int s = 0; s < steps; ++s) {
            // prepare batch...
            int base = s * BATCH_SIZE;
            int this_batch = BATCH_SIZE;
            if (base + BATCH_SIZE > num_samples)
                this_batch = num_samples - base;
            for (int b = 0; b < this_batch; ++b) {
                int idx = indices[base + b];
                memcpy(&input_batch->values[b * IMG_H * IMG_W],
                    &all_images[idx * IMG_H * IMG_W],
                    IMG_H * IMG_W * sizeof(float));
                memcpy(&target_batch->values[b * IMG_H * IMG_W],
                    &all_masks[idx * IMG_H * IMG_W],
                    IMG_H * IMG_W * sizeof(float));
            }
            if (this_batch < BATCH_SIZE) {
                int rem = (BATCH_SIZE - this_batch) * IMG_H * IMG_W;
                memset(&input_batch->values[this_batch * IMG_H * IMG_W], 0, rem * sizeof(float));
                memset(&target_batch->values[this_batch * IMG_H * IMG_W], 0, rem * sizeof(float));
            }
            float lr = get_lr(global_step, total_steps, 1e-4f);

            unet_zero_grads(&model);
            UNetIntermediates* im = unet_forward(&model, input_batch);
            float loss = mse_loss(im->pred_mask, target_batch);
            epoch_loss += loss;
            Tensor* grad = mse_loss_backward(im->pred_mask, target_batch);
            unet_backward(&model, im, grad);
            clip_gradients(&model, 1.0f);
            unet_update_adam(&model, lr, 0.9f, 0.999f, 1e-8f);
            free_tensor(grad);
            unet_free_intermediates(im);

            // update progress bar
            int elapsed = (int)(time(NULL) - epoch_start);
            float avg_loss = epoch_loss / (s + 1);
            print_progress(s + 1, steps_per_epoch, avg_loss, elapsed);
        }
        printf("\nEpoch %d complete — Avg Loss: %.6f\n\n",
            e + 1, epoch_loss / steps);

        // save after each epoch
        save_model(&model, model_path);
    }

    // cleanup
    unet_free(&model);
    free_tensor(input_batch);
    free_tensor(target_batch);
    free(all_images);
    free(all_masks);
    free(indices);

    printf("--- Training Finished ---\n");
    return 0;
}
