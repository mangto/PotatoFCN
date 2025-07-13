#include "utils.h"
#include <stdio.h>

void print_array_int(int* arr, int size) {
    printf("[");
    for (int i = 0; i < size; i++) {
        printf("%d", arr[i]);
        if (i < size - 1) printf(", ");
    }
    printf("]\n");
}

void print_array_float(float* arr, int size) {
    printf("[");
    for (int i = 0; i < size; i++) {
        printf("%.2f", arr[i]);
        if (i < size - 1) printf(", ");
    }
    printf("]\n");
}