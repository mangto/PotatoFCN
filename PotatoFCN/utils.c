#include "utils.h"
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

float bce_loss(const Tensor* pred, const Tensor* target) {
    int N = get_tensor_size(pred);
    float loss = 0;
    for (int i = 0; i < N; ++i) {
        float p = fminf(fmaxf(pred->values[i], 1e-7f), 1 - 1e-7f);
        float y = target->values[i];
        loss += -(y * logf(p) + (1 - y) * logf(1 - p));
    }
    return loss / N;
}

Tensor* bce_loss_backward(const Tensor* pred, const Tensor* target) {
    int N = get_tensor_size(pred);
    Tensor* grad = create_tensor(pred->shape, pred->dims);
    for (int i = 0; i < N; ++i) {
        float p = fminf(fmaxf(pred->values[i], 1e-7f), 1 - 1e-7f);
        float y = target->values[i];
        grad->values[i] = (p - y) / (p * (1 - p) * N);
    }
    return grad;
}

float focal_loss(const Tensor* pred, const Tensor* target, float gamma) {
    int N = get_tensor_size(pred);
    float loss = 0.0f;
    for (int i = 0; i < N; ++i) {
        // clamp to avoid log(0)
        float p = fminf(fmaxf(pred->values[i], 1e-7f), 1.0f - 1e-7f);
        float y = target->values[i];
        // p_t = p if y==1 else (1-p)
        float p_t = y ? p : (1.0f - p);
        // focal weight = (1 - p_t)^gamma
        float w = powf(1.0f - p_t, gamma);
        // cross-entropy term
        float ce = y ? -logf(p) : -logf(1.0f - p);
        loss += w * ce;
    }
    return loss / N;
}

// --- Focal Loss backward ---
Tensor* focal_loss_backward(const Tensor* pred, const Tensor* target, float gamma) {
    int N = get_tensor_size(pred);
    Tensor* grad = create_tensor(pred->shape, pred->dims);
    for (int i = 0; i < N; ++i) {
        float p = fminf(fmaxf(pred->values[i], 1e-7f), 1.0f - 1e-7f);
        float y = target->values[i];
        float p_t = y ? p : (1.0f - p);
        float w = powf(1.0f - p_t, gamma);
        // d/dp of focal term:
        // for y=1: loss = w * (-log p)
        //    dw/dp = -¥ã(1-p)^(¥ã-1)
        //    dloss/dp = dw*(-log p) + w*(-1/p)
        // for y=0: similarly using 1-p and log(1-p)
        float dwdp = -gamma * powf(1.0f - p_t, gamma - 1.0f) * (y ? 1.0f : -1.0f);
        float dcedp = y ? (-1.0f / p) : (1.0f / (1.0f - p));
        float dloss_dp = dwdp * (y ? -logf(p) : -logf(1.0f - p)) + w * dcedp;
        // average over N
        grad->values[i] = dloss_dp / N;
    }
    return grad;
}