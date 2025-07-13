#include "cnn.h"
#include "tensor.h"
#include "math_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>

void load_mnist_data(const char* image_path, const char* label_path, Tensor** images, Tensor** labels, int num_samples) {
    FILE* f_images = fopen(image_path, "rb");
    FILE* f_labels = fopen(label_path, "rb");
    if (!f_images || !f_labels) {
        fprintf(stderr, "Error: Could not open MNIST data file: %s or %s.\n", image_path, label_path);
        fprintf(stderr, "Please ensure the 'mnist_data' directory exists and contains the .bin files.\n");
        exit(1);
    }

    *images = create_tensor((int[]) { num_samples, 784 }, 2);
    *labels = create_tensor((int[]) { num_samples, 10 }, 2);

    fread((*images)->values, sizeof(float), get_tensor_size(*images), f_images);
    fread((*labels)->values, sizeof(float), get_tensor_size(*labels), f_labels);

    fclose(f_images);
    fclose(f_labels);
}

float calculate_loss(Tensor* y_pred, Tensor* y_true) {
    assert(y_pred->shape[0] == y_true->shape[0]);
    int batch_size = y_pred->shape[0];
    int num_classes = y_pred->shape[1];
    float total_loss = 0.0f;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_classes; j++) {
            if (y_true->values[i * num_classes + j] == 1.0f) {
                total_loss -= logf(y_pred->values[i * num_classes + j] + 1e-9);
            }
        }
    }
    return total_loss / batch_size;
}

float calculate_accuracy(Model* model, Tensor* test_images, Tensor* test_labels) {
    int correct = 0;
    int num_samples = test_images->shape[0];
    for (int i = 0; i < num_samples; i++) {
        Tensor* img_flat = create_tensor_from_array(&test_images->values[i * 784], (int[]) { 1, 784 }, 2);


        Tensor* img = copy_tensor(img_flat);
        tensor_reshape(img, (int[]) { 1, 1, 28, 28 }, 4);

        Tensor* output = model_forward(model, img, 0);

        int predicted = 0;
        float max_p = -1.0f;
        for (int j = 0; j < 10; j++) {
            if (output->values[j] > max_p) {
                max_p = output->values[j];
                predicted = j;
            }
        }

        int actual = 0;
        for (int j = 0; j < 10; j++) {
            if (test_labels->values[i * 10 + j] == 1.0f) {
                actual = j;
                break;
            }
        }

        if (predicted == actual) correct++;

        free_tensor(img_flat);
        free_tensor(img);
        free_tensor(output);
    }
    return (float)correct / num_samples;
}

int main() {
    srand(time(NULL));

    Tensor* train_images, * train_labels, * test_images, * test_labels;
    printf("Loading MNIST data...\n");
    load_mnist_data("mnist_data/train_images.bin", "mnist_data/train_labels.bin", &train_images, &train_labels, 60000);
    load_mnist_data("mnist_data/test_images.bin", "mnist_data/test_labels.bin", &test_images, &test_labels, 10000);
    printf("Data loaded successfully.\n\n");

    // ==========================================================
    // ## Le-Net5 Style ##
    // ==========================================================
    Model model;
    model_init(&model);

    // Input: (Batch, 1, 28, 28)

    // Block 1: Conv -> ReLU -> Pool
    model_add_conv2d(&model, 1, 6, 5, 1, 2); // Padding=2 to keep 28x28
    model_add_activation(&model, LAYER_RELU);
    model_add_maxpool(&model, 2, 2); // Output: (Batch, 6, 14, 14)

    // Block 2: Conv -> ReLU -> Pool
    model_add_conv2d(&model, 6, 16, 5, 1, 0); // Output: (Batch, 16, 10, 10)
    model_add_activation(&model, LAYER_RELU);
    model_add_maxpool(&model, 2, 2); // Output: (Batch, 16, 5, 5)

    // Flatten: 4D feature map to 2D vector
    model_add_flatten(&model); // Output: (Batch, 16 * 5 * 5) = (Batch, 400)

    model_add_dense(&model, 400, 120);
    model_add_activation(&model, LAYER_RELU);
    model_add_dense(&model, 120, 84);
    model_add_activation(&model, LAYER_RELU);
    model_add_dense(&model, 84, 10);
    model_add_activation(&model, LAYER_SOFTMAX);

    printf("CNN model initialized successfully.\n\n");

    // Hyperparmater
    int epochs = 5;
    float learning_rate = 0.01f;
    int batch_size = 64;
    int num_batches = train_images->shape[0] / batch_size;

    printf("--- Starting Training ---\n");
    for (int e = 0; e < epochs; e++) {
        float epoch_loss = 0.0f;
        for (int b = 0; b < num_batches; b++) {
            Tensor* X_batch_flat = create_tensor_from_array(&train_images->values[b * batch_size * 784], (int[]) { batch_size, 784 }, 2);
            Tensor* X_batch = copy_tensor(X_batch_flat);
            tensor_reshape(X_batch, (int[]) { batch_size, 1, 28, 28 }, 4); // (N, C, H, W)

            Tensor* Y_batch = create_tensor_from_array(&train_labels->values[b * batch_size * 10], (int[]) { batch_size, 10 }, 2);

            Tensor* Y_pred = model_forward(&model, X_batch, 1);

            epoch_loss += calculate_loss(Y_pred, Y_batch);

            model_backward(&model, Y_pred, Y_batch);

            model_update_params(&model, learning_rate, batch_size);

            free_tensor(X_batch_flat);
            free_tensor(X_batch);
            free_tensor(Y_batch);
            free_tensor(Y_pred);

            printf("\rEpoch %d/%d, Batch %d/%d", e + 1, epochs, b + 1, num_batches);
            fflush(stdout);
        }


        float accuracy = calculate_accuracy(&model, test_images, test_labels);
        printf("\rEpoch %d/%d - Loss: %f, Accuracy: %.2f%%\n", e + 1, epochs, epoch_loss / num_batches, accuracy * 100.0f);
    }
    printf("--- Training Finished ---\n\n");

    float final_accuracy = calculate_accuracy(&model, test_images, test_labels);
    printf("Final test accuracy: %.2f%%\n", final_accuracy * 100.0f);

    model_free(&model);
    free_tensor(train_images);
    free_tensor(train_labels);
    free_tensor(test_images);
    free_tensor(test_labels);

    return 0;
}