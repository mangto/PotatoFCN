#define _CRT_SECURE_NO_WARNINGS
#include "cnn_unet.h"
#include "tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Functions for loading data and calculating loss, defined at the top of the program for clarity.
// ... (load_preprocessed_data, mse_loss, mse_loss_backward, print_progress functions) ...
void load_preprocessed_data(const char* path, int* num_samples, int* h, int* w, float** images, float** masks) {
    char info_path[256], images_path[256], masks_path[256];
    sprintf(info_path, "%s/info.txt", path);
    sprintf(images_path, "%s/coco_images.bin", path);
    sprintf(masks_path, "%s/coco_masks.bin", path);

    FILE* f_info = fopen(info_path, "r");
    if (!f_info) { printf("Error: Could not open info file at %s\n", info_path); exit(1); }
    if (fscanf(f_info, "%d\n%d\n%d", num_samples, h, w) != 3) { printf("Error: Invalid format in info.txt\n"); exit(1); }
    fclose(f_info);

    long img_total_size = (long)(*num_samples) * (*h) * (*w);
    *images = (float*)malloc(img_total_size * sizeof(float));
    *masks = (float*)malloc(img_total_size * sizeof(float));
    if (!(*images) || !(*masks)) { printf("Error: Memory allocation failed for dataset.\n"); exit(1); }

    FILE* f_images = fopen(images_path, "rb");
    if (!f_images) { printf("Error: Could not open images file at %s\n", images_path); exit(1); }
    fread(*images, sizeof(float), img_total_size, f_images);
    fclose(f_images);

    FILE* f_masks = fopen(masks_path, "rb");
    if (!f_masks) { printf("Error: Could not open masks file at %s\n", masks_path); exit(1); }
    fread(*masks, sizeof(float), img_total_size, f_masks);
    fclose(f_masks);

    printf("Loaded %d samples of size %dx%d from '%s'\n", *num_samples, *h, *w, path);
}

float mse_loss(Tensor* pred, Tensor* target) {
    int size = get_tensor_size(pred);
    float loss = 0.0f;
    for (int i = 0; i < size; ++i) {
        float diff = pred->values[i] - target->values[i];
        loss += diff * diff;
    }
    return loss / size;
}

Tensor* mse_loss_backward(Tensor* pred, Tensor* target) {
    Tensor* grad = create_tensor(pred->shape, pred->dims);
    int size = get_tensor_size(pred);
    for (int i = 0; i < size; ++i) {
        grad->values[i] = 2.0f * (pred->values[i] - target->values[i]) / size;
    }
    return grad;
}

void print_progress(int current, int total, float loss) {
    int bar_width = 50;
    float progress = (float)current / total;
    int pos = bar_width * progress;

    printf("\r[");
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) printf("=");
        else if (i == pos) printf(">");
        else printf(" ");
    }
    printf("] %d/%d (%.1f%%) - Loss: %.6f", current, total, progress * 100.0f, loss);
    fflush(stdout);
}

int main() {
    srand((unsigned int)time(NULL));

    int num_samples, IMG_H, IMG_W;
    float* all_images_data, * all_masks_data;
    load_preprocessed_data("./preprocessed_data", &num_samples, &IMG_H, &IMG_W, &all_images_data, &all_masks_data);

    UNetModel model;
    unet_build(&model);

    int epochs = 5;
    float learning_rate = 0.01f;
    printf("\n--- Starting U-Net Training with COCO data ---\n");

    for (int e = 0; e < epochs; ++e) {
        printf("Epoch %d/%d\n", e + 1, epochs);
        float epoch_loss = 0.0f;
        for (int i = 0; i < num_samples; ++i) {
            Tensor* input_image = create_tensor((int[]) { 1, 1, IMG_H, IMG_W }, 4);
            Tensor* target_mask = create_tensor((int[]) { 1, 1, IMG_H, IMG_W }, 4);
            long data_size = (long)IMG_H * IMG_W * sizeof(float);
            memcpy(input_image->values, &all_images_data[(long)i * IMG_H * IMG_W], data_size);
            memcpy(target_mask->values, &all_masks_data[(long)i * IMG_H * IMG_W], data_size);

            // 1. Initialize gradients
            unet_zero_grads(&model);

            // 2. Forward pass
            UNetIntermediates* im = unet_forward(&model, input_image);

            // 3. Calculate loss and gradients
            float loss = mse_loss(im->pred_mask, target_mask);
            epoch_loss += loss;
            Tensor* grad = mse_loss_backward(im->pred_mask, target_mask);

            // 4. Backward pass
            unet_backward(&model, im, grad);

            // 5. Update parameters
            unet_update_params(&model, learning_rate);

            // --- [Memory Management] ---
            // 6. Free intermediate tensors
            free_tensor(grad);
            unet_free_intermediates(im); // unet_free_intermediates frees all tensors within the im struct
            free_tensor(input_image);
            free_tensor(target_mask);
            // --- [End of Memory Management] ---

            // Update progress bar
            print_progress(i + 1, num_samples, epoch_loss / (i + 1));
        }
        printf("\nEpoch %d/%d - Average Loss: %f\n\n", e + 1, epochs, epoch_loss / num_samples);
    }

    unet_free(&model); // Free all allocated memory for the model
    free(all_images_data);
    free(all_masks_data);
    printf("--- Training Finished ---\n");

    return 0;
}