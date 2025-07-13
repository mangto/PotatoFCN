#include "fnn.h"
#include "matrix.h"
#include "math_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void load_mnist_data(const char* image_path, const char* label_path, Matrix* images, Matrix* labels, int num_samples) {
    FILE* f_images = fopen(image_path, "rb");
    FILE* f_labels = fopen(label_path, "rb");
    if (!f_images || !f_labels) {
        fprintf(stderr, "Error: Cannot open MNIST data files. Check if 'mnist_data' directory exists and contains .bin files.\n");
        exit(1);
    }

    int image_shape[] = { num_samples, 784 };
    *images = create_matrix_from_array(NULL, image_shape, 2);

    int label_shape[] = { num_samples, 10 };
    *labels = create_matrix_from_array(NULL, label_shape, 2);

    fread(images->values, sizeof(float), num_samples * 784, f_images);
    fread(labels->values, sizeof(float), num_samples * 10, f_labels);

    fclose(f_images);
    fclose(f_labels);
}

float calculate_accuracy(FNN* fnn, Matrix* test_images, Matrix* test_labels) {
    int correct_predictions = 0;
    for (int i = 0; i < test_images->shape[0]; i++) {
        int shape[] = { 1, 784 };
        Matrix input = create_matrix_from_array(&test_images->values[i * 784], shape, 2);

        forward_fnn(fnn, &input);

        Matrix* output = &fnn->activations[fnn->num_layers - 1];

        int predicted_label = 0;
        float max_val = output->values[0];
        for (int j = 1; j < 10; j++) {
            if (output->values[j] > max_val) {
                max_val = output->values[j];
                predicted_label = j;
            }
        }

        int true_label = 0;
        for (int j = 0; j < 10; j++) {
            if (test_labels->values[i * 10 + j] == 1.0f) {
                true_label = j;
                break;
            }
        }

        if (predicted_label == true_label) {
            correct_predictions++;
        }
        free_matrix(&input);
    }
    return (float)correct_predictions / test_images->shape[0];
}

int main() {
    srand(time(NULL));

    Matrix train_images, train_labels, test_images, test_labels;
    printf("Loading MNIST data...\n");
    load_mnist_data("mnist_data/train_images.bin", "mnist_data/train_labels.bin", &train_images, &train_labels, 60000);
    load_mnist_data("mnist_data/test_images.bin", "mnist_data/test_labels.bin", &test_images, &test_labels, 10000);
    printf("Data loaded.\n\n");

    int layer_sizes[] = { 784, 128, 10 };
    int num_layers = sizeof(layer_sizes) / sizeof(int);
    FNN fnn;
    init_fnn(&fnn, num_layers, layer_sizes);
    printf("FNN initialized.\n");

    int epochs = 5;
    float learning_rate = 0.01f;
    int batch_size = 64;
    int num_batches = 60000 / batch_size;

    for (int e = 0; e < epochs; e++) {
        printf("Epoch %d/%d\n", e + 1, epochs);
        for (int b = 0; b < num_batches; b++) {
            int input_shape[] = { batch_size, 784 };
            Matrix input_batch = create_matrix_from_array(&train_images.values[b * batch_size * 784], input_shape, 2);

            int target_shape[] = { batch_size, 10 };
            Matrix target_batch = create_matrix_from_array(&train_labels.values[b * batch_size * 10], target_shape, 2);

            forward_fnn(&fnn, &input_batch);
            backward_fnn(&fnn, &input_batch, &target_batch, learning_rate);

            free_matrix(&input_batch);
            free_matrix(&target_batch);

            int bar_width = 50;
            float progress = (float)(b + 1) / num_batches;
            int pos = bar_width * progress;

            printf("\r[");
            for (int i = 0; i < bar_width; ++i) {
                if (i < pos) printf("=");
                else if (i == pos) printf(">");
                else printf(" ");
            }
            printf("] %d%%", (int)(progress * 100.0f));
            fflush(stdout);
        }
        printf("\n");
        float accuracy = calculate_accuracy(&fnn, &test_images, &test_labels);
        printf("Accuracy on test set: %.2f%%\n\n", accuracy * 100);
    }

    free_fnn(&fnn);
    free_matrix(&train_images);
    free_matrix(&train_labels);
    free_matrix(&test_images);
    free_matrix(&test_labels);

    return 0;
}