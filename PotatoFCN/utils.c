#include "utils.h"
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Dataset loader: reads info.txt, images and mask binaries
void load_preprocessed_data(
    const char* path,
    int* num_samples,
    int* h,
    int* w,
    float** images,
    float** masks
) {
    char info_path[256];
    char images_path[256];
    char masks_path[256];
    // use secure _s variants to prevent buffer overflow
    sprintf_s(info_path, sizeof(info_path), "%s/info.txt", path);
    sprintf_s(images_path, sizeof(images_path), "%s/coco_images.bin", path);
    sprintf_s(masks_path, sizeof(masks_path), "%s/coco_masks.bin", path);

    FILE* f_info = NULL;
    if (fopen_s(&f_info, info_path, "r") != 0 || !f_info) {
        fprintf(stderr, "Error: Could not open %s\n", info_path);
        exit(1);
    }
    // read number of samples, height, width
    if (fscanf_s(f_info, "%d %d %d", num_samples, h, w) != 3) {
        fprintf(stderr, "Error: Invalid format in info.txt\n");
        fclose(f_info);
        exit(1);
    }
    fclose(f_info);

    long total = (long)(*num_samples) * (*h) * (*w);
    *images = (float*)malloc(total * sizeof(float));
    *masks = (float*)malloc(total * sizeof(float));
    if (!*images || !*masks) {
        fprintf(stderr, "Error: Memory allocation failed for dataset\n");
        exit(1);
    }

    FILE* f_im = NULL;
    if (fopen_s(&f_im, images_path, "rb") != 0 || !f_im) {
        fprintf(stderr, "Error: Could not open %s\n", images_path);
        exit(1);
    }
    fread(*images, sizeof(float), total, f_im);
    fclose(f_im);

    FILE* f_ma = NULL;
    if (fopen_s(&f_ma, masks_path, "rb") != 0 || !f_ma) {
        fprintf(stderr, "Error: Could not open %s\n", masks_path);
        exit(1);
    }
    fread(*masks, sizeof(float), total, f_ma);
    fclose(f_ma);

    printf("Loaded %d samples of size %dx%d from '%s'\n",
        *num_samples, *h, *w, path);
}

// Mean Squared Error loss
double mse_loss(const Tensor* pred, const Tensor* target) {
    int size = get_tensor_size(pred);
    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
        double d = pred->values[i] - target->values[i];
        sum += d * d;
    }
    return sum / size;
}

// Backward pass for MSE loss
Tensor* mse_loss_backward(const Tensor* pred, const Tensor* target) {
    int size = get_tensor_size(pred);
    Tensor* grad = create_tensor(pred->shape, pred->dims);
    float scale = 2.0f / size;
    for (int i = 0; i < size; ++i) {
        grad->values[i] = scale * (pred->values[i] - target->values[i]);
    }
    return grad;
}
