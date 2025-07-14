#include "tensor.h"
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <float.h>
#include <stdio.h>
#include <immintrin.h>
#include <omp.h>

#include <cblas.h>

// --- Tensor creation and memory management ---
Tensor* create_tensor(const int* shape, int dims) {
    return create_tensor_from_array(NULL, shape, dims);
}

Tensor* create_tensor_from_array(const float* values, const int* shape, int dims) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    assert(t != NULL);

    t->dims = dims;
    t->shape = (int*)malloc(dims * sizeof(int));
    assert(t->shape != NULL);
    memcpy(t->shape, shape, dims * sizeof(int));

    int total_elements = get_tensor_size(t);
    t->values = (float*)malloc(total_elements * sizeof(float));
    assert(t->values != NULL);

    if (values != NULL) {
        memcpy(t->values, values, total_elements * sizeof(float));
    }
    else {
        memset(t->values, 0, total_elements * sizeof(float));
    }
    return t;
}

Tensor* copy_tensor(const Tensor* src) {
    return create_tensor_from_array(src->values, src->shape, src->dims);
}

void free_tensor(Tensor* tensor) {
    if (tensor == NULL) {
        return;
    }
    if (tensor->values) {
        free(tensor->values);
        tensor->values = NULL;
    }
    if (tensor->shape) {
        free(tensor->shape);
        tensor->shape = NULL;
    }
    free(tensor);
}

int get_tensor_size(const Tensor* t) {
    if (!t) return 0;
    int size = 1;
    for (int i = 0; i < t->dims; i++) {
        size *= t->shape[i];
    }
    return size;
}

void print_tensor(const Tensor* t, int print_values) {
    if (!t) {
        printf("NULL Tensor\n");
        return;
    }
    printf("Tensor (dims=%d): [", t->dims);
    for (int i = 0; i < t->dims; i++) {
        printf("%d%s", t->shape[i], i == t->dims - 1 ? "" : ", ");
    }
    printf("], Total elements: %d\n", get_tensor_size(t));
    if (print_values) {
        int size = get_tensor_size(t);
        for (int i = 0; i < size && i < 10; i++) { // Print first 10 values
            printf("%f ", t->values[i]);
        }
        if (size > 10) printf("...");
        printf("\n");
    }
}


// --- Basic operations ---
void tensor_update(Tensor* t, const Tensor* grad, float learning_rate, int batch_size) {
    int size = get_tensor_size(t);
    for (int i = 0; i < size; i++) {
        t->values[i] -= (learning_rate * grad->values[i]) / batch_size;
    }
}

void tensor_reshape(Tensor* t, int* new_shape, int new_dims) {
    int old_size = get_tensor_size(t);
    int new_size = 1;
    for (int i = 0; i < new_dims; i++) {
        new_size *= new_shape[i];
    }
    assert(old_size == new_size);

    free(t->shape);
    t->shape = (int*)malloc(new_dims * sizeof(int));
    memcpy(t->shape, new_shape, new_dims * sizeof(int));
    t->dims = new_dims;
}

Tensor* tensor_sub(const Tensor* t1, const Tensor* t2) {
    assert(get_tensor_size(t1) == get_tensor_size(t2));
    Tensor* result = copy_tensor(t1);
    int size = get_tensor_size(result);
    float* p_res = result->values;
    const float* p2 = t2->values;

    int i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 ymm1 = _mm256_loadu_ps(p_res + i);
        __m256 ymm2 = _mm256_loadu_ps(p2 + i);
        __m256 ymm_res = _mm256_sub_ps(ymm1, ymm2);
        _mm256_storeu_ps(p_res + i, ymm_res);
    }

    for (; i < size; i++) {
        p_res[i] -= p2[i];
    }
    return result;
}

Tensor* tensor_elemwise_mul(const Tensor* t1, const Tensor* t2) {
    assert(get_tensor_size(t1) == get_tensor_size(t2));
    Tensor* result = copy_tensor(t1);
    int size = get_tensor_size(result);
    float* p_res = result->values;
    const float* p2 = t2->values;

    int i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 ymm1 = _mm256_loadu_ps(p_res + i);
        __m256 ymm2 = _mm256_loadu_ps(p2 + i);
        __m256 ymm_res = _mm256_mul_ps(ymm1, ymm2);
        _mm256_storeu_ps(p_res + i, ymm_res);
    }

    for (; i < size; i++) {
        p_res[i] *= p2[i];
    }
    return result;
}

Tensor* tensor_dot(const Tensor* t1, const Tensor* t2) {
    assert(t1->dims == 2 && t2->dims == 2);
    assert(t1->shape[1] == t2->shape[0]);

    int m = t1->shape[0];
    int k = t1->shape[1];
    int n = t2->shape[1];

    Tensor* result = create_tensor((int[]) { m, n }, 2);

    // cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
    cblas_sgemm(CblasRowMajor, // 행 우선 순서
        CblasNoTrans,  // A는 Transpose 안함
        CblasNoTrans,  // B도 Transpose 안함
        m, n, k,       // M, N, K
        1.0f,          // alpha (곱셈 계수)
        t1->values, k, // A 행렬, lda (A의 열 개수)
        t2->values, n, // B 행렬, ldb (B의 열 개수)
        0.0f,          // beta (덧셈 계수, 0이면 덮어쓰기)
        result->values, n); // C 행렬, ldc (C의 열 개수)

    return result;
}

Tensor* tensor_transpose(const Tensor* t) {
    assert(t->dims == 2);
    Tensor* result = create_tensor((int[]) { t->shape[1], t->shape[0] }, 2);
    for (int i = 0; i < t->shape[0]; i++) {
        for (int j = 0; j < t->shape[1]; j++) {
            result->values[j * t->shape[0] + i] = t->values[i * t->shape[1] + j];
        }
    }
    return result;
}

Tensor* tensor_add_broadcast(const Tensor* main, const Tensor* broadcast) {
    assert(main->dims == 2 && broadcast->dims == 2);
    assert(main->shape[1] == broadcast->shape[1] && broadcast->shape[0] == 1);
    Tensor* result = copy_tensor(main);
    int M = main->shape[0];
    int N = main->shape[1];

    const float* p_br = broadcast->values;

    for (int i = 0; i < M; i++) {
        float* p_res_row = result->values + i * N;
        int j = 0;
        for (; j <= N - 8; j += 8) {
            __m256 ymm_main = _mm256_loadu_ps(p_res_row + j);
            __m256 ymm_br = _mm256_loadu_ps(p_br + j);
            __m256 ymm_res = _mm256_add_ps(ymm_main, ymm_br);
            _mm256_storeu_ps(p_res_row + j, ymm_res);
        }

        for (; j < N; j++) {
            p_res_row[j] += p_br[j];
        }
    }
    return result;
}

Tensor* tensor_sum_along_axis(const Tensor* t, int axis) {
    assert(axis == 0 && t->dims == 2);
    Tensor* result = create_tensor((int[]) { 1, t->shape[1] }, 2);
    for (int j = 0; j < t->shape[1]; j++) {
        float sum = 0.0f;
        for (int i = 0; i < t->shape[0]; i++) {
            sum += t->values[i * t->shape[1] + j];
        }
        result->values[j] = sum;
    }
    return result;
}

// --- CNN operations ---

static void im2col(const float* data_im, int channels, int height, int width,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    int pad_h, int pad_w, float* data_col) {
    int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int channels_col = channels * kernel_h * kernel_w;
#pragma omp parallel for
    for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % kernel_w;
        int h_offset = (c / kernel_w) % kernel_h;
        int c_im = c / (kernel_h * kernel_w);
        for (int h = 0; h < height_col; ++h) {
            for (int w = 0; w < width_col; ++w) {
                int h_pad = h * stride_h - pad_h + h_offset;
                int w_pad = w * stride_w - pad_w + w_offset;
                if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
                    data_col[(c * height_col + h) * width_col + w] = data_im[(c_im * height + h_pad) * width + w_pad];
                else
                    data_col[(c * height_col + h) * width_col + w] = 0;
            }
        }
    }
}

static void col2im(const float* data_col, int channels, int height, int width,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    int pad_h, int pad_w, float* data_im) {
    memset(data_im, 0, height * width * channels * sizeof(float));
    int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int channels_col = channels * kernel_h * kernel_w;
    for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % kernel_w;
        int h_offset = (c / kernel_w) % kernel_h;
        int c_im = c / (kernel_h * kernel_w);
        for (int h = 0; h < height_col; ++h) {
            for (int w = 0; w < width_col; ++w) {
                int h_pad = h * stride_h - pad_h + h_offset;
                int w_pad = w * stride_w - pad_w + w_offset;
                if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width) {
                    data_im[(c_im * height + h_pad) * width + w_pad] += data_col[(c * height_col + h) * width_col + w];
                }
            }
        }
    }
}



Tensor* tensor_conv2d(const Tensor* input, const Tensor* weights, const Tensor* biases,
    int stride, int padding, Tensor** col_buffer_ptr) {

    assert(input->dims == 4 && weights->dims == 4 && biases->dims == 1);
    assert(input->shape[1] == weights->shape[1]);

    int N = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
    int F = weights->shape[0], K = weights->shape[2];
    int H_out = (H - K + 2 * padding) / stride + 1;
    int W_out = (W - K + 2 * padding) / stride + 1;

    int M_gemm = F;
    int K_gemm = C * K * K;
    int N_gemm = H_out * W_out;

    long required_size = (long)K_gemm * N_gemm;
    if (*col_buffer_ptr == NULL || get_tensor_size(*col_buffer_ptr) < required_size) {
        if (*col_buffer_ptr) free_tensor(*col_buffer_ptr);
        *col_buffer_ptr = create_tensor((int[]) { required_size }, 1);
    }
    Tensor* col_buffer = *col_buffer_ptr;

    Tensor weights_reshaped = *weights;
    weights_reshaped.dims = 2;
    int new_shape[] = { M_gemm, K_gemm };
    weights_reshaped.shape = new_shape;

    Tensor* output = create_tensor((int[]) { N, F, H_out, W_out }, 4);

    for (int i = 0; i < N; ++i) {
        const float* input_image = input->values + i * C * H * W;
        im2col(input_image, C, H, W, K, K, stride, stride, padding, padding, col_buffer->values);

        Tensor col_buffer_reshaped;
        col_buffer_reshaped.dims = 2;
        int col_shape[] = { K_gemm, N_gemm };
        col_buffer_reshaped.shape = col_shape;
        col_buffer_reshaped.values = col_buffer->values;

        Tensor* output_gemm = tensor_dot(&weights_reshaped, &col_buffer_reshaped);
        assert(output_gemm->shape[0] == F && output_gemm->shape[1] == N_gemm);

        float* output_ptr = output->values + i * F * N_gemm;
        const float* gemm_ptr = output_gemm->values;

#pragma omp parallel for
        for (int f = 0; f < F; ++f) {
            const float bias_val = biases->values[f];
            float* out_channel_ptr = output_ptr + f * N_gemm;
            const float* gemm_channel_ptr = gemm_ptr + f * N_gemm;
            int j = 0;
            __m256 ymm_bias = _mm256_set1_ps(bias_val);
            for (; j <= N_gemm - 8; j += 8) {
                __m256 ymm_gemm = _mm256_loadu_ps(gemm_channel_ptr + j);
                __m256 ymm_res = _mm256_add_ps(ymm_gemm, ymm_bias);
                _mm256_storeu_ps(out_channel_ptr + j, ymm_res);
            }
            for (; j < N_gemm; ++j) {
                out_channel_ptr[j] = gemm_channel_ptr[j] + bias_val;
            }
        }

        free_tensor(output_gemm);
    }

    return output;
}

Tensor* tensor_maxpool(const Tensor* input, int pool_size, int stride, Tensor** max_indices_tensor) {
    assert(input->dims == 4);
    int N = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
    int H_out = (H - pool_size) / stride + 1;
    int W_out = (W - pool_size) / stride + 1;

    Tensor* output = create_tensor((int[]) { N, C, H_out, W_out }, 4);
    if (max_indices_tensor) {
        *max_indices_tensor = create_tensor((int[]) { N, C, H_out, W_out }, 4);
    }

    // 최적화 설명: N, C, H_out, W_out 4개 루프를 병렬화합니다.
    // 작업 부하가 균일할 것으로 예상되므로 schedule(static)을 사용합니다.
#pragma omp parallel for collapse(4) schedule(static)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int ho = 0; ho < H_out; ho++) {
                for (int wo = 0; wo < W_out; wo++) {
                    float max_val = -FLT_MAX;
                    int max_idx = -1;
                    for (int ph = 0; ph < pool_size; ph++) {
                        for (int pw = 0; pw < pool_size; pw++) {
                            int hi = ho * stride + ph;
                            int wi = wo * stride + pw;
                            int current_flat_idx = hi * W + wi;
                            float val = input->values[n * C * H * W + c * H * W + current_flat_idx];
                            if (val > max_val) {
                                max_val = val;
                                max_idx = current_flat_idx;
                            }
                        }
                    }
                    output->values[n * C * H_out * W_out + c * H_out * W_out + ho * W_out + wo] = max_val;
                    if (max_indices_tensor) {
                        (*max_indices_tensor)->values[n * C * H_out * W_out + c * H_out * W_out + ho * W_out + wo] = (float)max_idx;
                    }
                }
            }
        }
    }
    return output;
}

// --- CNN backward pass ---
Tensor* tensor_maxpool_backward(const Tensor* grad_output, const Tensor* original_input, const Tensor* max_indices) {
    Tensor* grad_input = create_tensor(original_input->shape, original_input->dims); // 0으로 초기화
    int N = grad_output->shape[0], C = grad_output->shape[1];
    int H_out = grad_output->shape[2], W_out = grad_output->shape[3];
    int H_in = original_input->shape[2], W_in = original_input->shape[3];

    // 최적화 설명: 4개의 외부 루프를 합쳐서 병렬화합니다.
#pragma omp parallel for collapse(4) schedule(static)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int ho = 0; ho < H_out; ho++) {
                for (int wo = 0; wo < W_out; wo++) {
                    float grad = grad_output->values[n * C * H_out * W_out + c * H_out * W_out + ho * W_out + wo];
                    int max_idx_flat = (int)max_indices->values[n * C * H_out * W_out + c * H_out * W_out + ho * W_out + wo];
                    int grad_input_idx = n * C * H_in * W_in + c * H_in * W_in + max_idx_flat;

                    // 주의: 이 연산은 원자적(atomic)이어야 할 수 있으나, max-pooling의 특성상
                    // 서로 다른 (ho, wo)가 같은 grad_input_idx에 접근할 확률이 거의 없어 일반적으로 안전합니다.
                    // 만약의 경우를 대비해 atomic을 사용할 수 있습니다.
#pragma omp atomic
                    grad_input->values[grad_input_idx] += grad;
                }
            }
        }
    }
    return grad_input;
}

Tensor* tensor_conv_grad_bias(const Tensor* grad_output) {
    int F = grad_output->shape[1];
    Tensor* grad_biases = create_tensor((int[]) { F }, 1);
    int N = grad_output->shape[0], H_out = grad_output->shape[2], W_out = grad_output->shape[3];

    // 최적화 설명: 각 필터(f)에 대한 계산은 독립적이므로 외부 루프를 병렬화합니다.
    // 이것이 이 함수 구조에 가장 적합한 병렬화 방식입니다.
#pragma omp parallel for schedule(static)
    for (int f = 0; f < F; f++) {
        float sum = 0.0f;
        // 이 내부 루프들은 각 스레드가 독립적으로 실행합니다.
        for (int n = 0; n < N; n++) {
            for (int h = 0; h < H_out; h++) {
                for (int w = 0; w < W_out; w++) {
                    sum += grad_output->values[n * F * H_out * W_out + f * H_out * W_out + h * W_out + w];
                }
            }
        }
        grad_biases->values[f] = sum;
    }
    return grad_biases;
}

Tensor* tensor_conv_grad_weights(const Tensor* input, const Tensor* grad_output, const Tensor* weights,
    int stride, int padding, Tensor** col_buffer_ptr) {

    int N = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
    int F = grad_output->shape[1], H_out = grad_output->shape[2], W_out = grad_output->shape[3];

    int K = weights->shape[2];

    int M_gemm = F;
    int K_gemm_gw = C * K * K;
    int N_gemm = H_out * W_out;

    long required_size = (long)K_gemm_gw * N_gemm;
    if (*col_buffer_ptr == NULL || get_tensor_size(*col_buffer_ptr) < required_size) {
        if (*col_buffer_ptr) free_tensor(*col_buffer_ptr);
        *col_buffer_ptr = create_tensor((int[]) { required_size }, 1);
    }
    Tensor* col_buffer = *col_buffer_ptr;

    Tensor* grad_weights_new = create_tensor((int[]) { F, C, K, K }, 4);

    for (int i = 0; i < N; ++i) {
        const float* input_image = input->values + i * C * H * W;
        const float* grad_output_image = grad_output->values + i * F * H_out * W_out;

        im2col(input_image, C, H, W, K, K, stride, stride, padding, padding, col_buffer->values);

        Tensor col_buffer_reshaped;
        col_buffer_reshaped.dims = 2;
        int col_shape[] = { K_gemm_gw, N_gemm };
        col_buffer_reshaped.shape = col_shape;
        col_buffer_reshaped.values = col_buffer->values;

        Tensor grad_output_reshaped;
        grad_output_reshaped.dims = 2;
        int grad_output_shape[] = { M_gemm, N_gemm };
        grad_output_reshaped.shape = grad_output_shape;
        grad_output_reshaped.values = (float*)grad_output_image;

        Tensor* col_buffer_T = tensor_transpose(&col_buffer_reshaped);
        Tensor* gw_gemm = tensor_dot(&grad_output_reshaped, col_buffer_T);

        for (int j = 0; j < get_tensor_size(gw_gemm); ++j) {
            grad_weights_new->values[j] += gw_gemm->values[j];
        }
        free_tensor(col_buffer_T);
        free_tensor(gw_gemm);
    }
    return grad_weights_new;
}

Tensor* tensor_conv_grad_input(const Tensor* grad_output, const Tensor* weights,
    int stride, int padding, Tensor* col_buffer) {
    int N = grad_output->shape[0];
    int F = grad_output->shape[1];
    int H_out = grad_output->shape[2];
    int W_out = grad_output->shape[3];

    int C = weights->shape[1];
    int K = weights->shape[2];

    int H = (H_out - 1) * stride + K - 2 * padding;
    int W = (W_out - 1) * stride + K - 2 * padding;

    Tensor* grad_input = create_tensor((int[]) { N, C, H, W }, 4);

    int M_gemm_gi = C * K * K;
    int N_gemm_gi = H_out * W_out;
    int K_gemm_gi = F;

    Tensor weights_T_proto = *weights;
    weights_T_proto.dims = 2;
    int w_shape[] = { K_gemm_gi, M_gemm_gi };
    weights_T_proto.shape = w_shape;
    Tensor* WT = tensor_transpose(&weights_T_proto);

    for (int i = 0; i < N; ++i) {
        const float* grad_output_image = grad_output->values + i * F * H_out * W_out;
        Tensor grad_output_reshaped;
        grad_output_reshaped.dims = 2;
        int grad_shape[] = { K_gemm_gi, N_gemm_gi };
        grad_output_reshaped.shape = grad_shape;
        grad_output_reshaped.values = (float*)grad_output_image;

        Tensor* dX_col = tensor_dot(WT, &grad_output_reshaped);

        float* grad_input_image = grad_input->values + i * C * H * W;
        col2im(dX_col->values, C, H, W, K, K, stride, stride, padding, padding, grad_input_image);
        free_tensor(dX_col);
    }

    free_tensor(WT);
    return grad_input;
}



Tensor* tensor_concatenate(Tensor* t1, Tensor* t2, int axis) {
    assert(axis == 1 && t1->dims == 4 && t2->dims == 4 && t1->shape[0] == t2->shape[0] && t1->shape[2] == t2->shape[2] && t1->shape[3] == t2->shape[3]);
    int N = t1->shape[0], C1 = t1->shape[1], C2 = t2->shape[1], H = t1->shape[2], W = t1->shape[3];
    int C_out = C1 + C2;
    size_t feature_map_size = (size_t)H * W;
    Tensor* output = create_tensor((int[]) { N, C_out, H, W }, 4);
    for (int n = 0; n < N; ++n) {
        memcpy(output->values + n * C_out * feature_map_size, t1->values + n * C1 * feature_map_size, C1 * feature_map_size * sizeof(float));
        memcpy(output->values + n * C_out * feature_map_size + C1 * feature_map_size, t2->values + n * C2 * feature_map_size, C2 * feature_map_size * sizeof(float));
    }
    return output;
}


void tensor_conv2d_backward(Tensor* grad_input, Tensor* grad_weights, Tensor* grad_biases, Tensor* grad_output, Tensor* input, Tensor* weights, int stride, int padding) {
    int N = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
    int F = weights->shape[0], K = weights->shape[2];
    int H_out = grad_output->shape[2], W_out = grad_output->shape[3];

    for (int f = 0; f < F; f++) {
        float sum = 0.0f;
        for (int n = 0; n < N; n++) for (int h = 0; h < H_out; h++) for (int w = 0; w < W_out; w++)
            sum += grad_output->values[n * F * H_out * W_out + f * H_out * W_out + h * W_out + w];
        grad_biases->values[f] += sum;
    }

    for (int n = 0; n < N; n++) for (int f = 0; f < F; f++) for (int ho = 0; ho < H_out; ho++) for (int wo = 0; wo < W_out; wo++) {
        float grad_out_val = grad_output->values[n * F * H_out * W_out + f * H_out * W_out + ho * W_out + wo];
        for (int c = 0; c < C; c++) for (int kh = 0; kh < K; kh++) for (int kw = 0; kw < K; kw++) {
            int hi = ho * stride + kh - padding, wi = wo * stride + kw - padding;
            if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                grad_weights->values[f * C * K * K + c * K * K + kh * K + kw] += input->values[n * C * H * W + c * H * W + hi * W + wi] * grad_out_val;
                grad_input->values[n * C * H * W + c * H * W + hi * W + wi] += weights->values[f * C * K * K + c * K * K + kh * K + kw] * grad_out_val;
            }
        }
    }
}

void tensor_transposed_conv2d_backward(Tensor* grad_input, Tensor* grad_weights, Tensor* grad_biases, Tensor* grad_output, Tensor* input, Tensor* weights, int stride) {
    int N = input->shape[0], C_in = input->shape[1], H_in = input->shape[2], W_in = input->shape[3];
    int C_out = weights->shape[1], K = weights->shape[2];
    int H_out = grad_output->shape[2], W_out = grad_output->shape[3];

    for (int c = 0; c < C_out; ++c) {
        float sum = 0;
        for (int n = 0; n < N; ++n) for (int h = 0; h < H_out; ++h) for (int w = 0; w < W_out; ++w)
            sum += grad_output->values[n * C_out * H_out * W_out + c * H_out * W_out + h * W_out + w];
        grad_biases->values[c] += sum;
    }

    for (int n = 0; n < N; ++n) for (int c_out = 0; c_out < C_out; ++c_out) for (int h_out = 0; h_out < H_out; ++h_out) for (int w_out = 0; w_out < W_out; ++w_out) {
        float grad_out_val = grad_output->values[n * C_out * H_out * W_out + c_out * H_out * W_out + h_out * W_out + w_out];
        for (int c_in = 0; c_in < C_in; ++c_in) for (int kh = 0; kh < K; ++kh) for (int kw = 0; kw < K; ++kw) {
            int h_in_idx = (h_out - kh), w_in_idx = (w_out - kw);
            if (h_in_idx >= 0 && h_in_idx % stride == 0 && w_in_idx >= 0 && w_in_idx % stride == 0) {
                h_in_idx /= stride; w_in_idx /= stride;
                if (h_in_idx < H_in && w_in_idx < W_in) {
                    grad_input->values[n * C_in * H_in * W_in + c_in * H_in * W_in + h_in_idx * W_in + w_in_idx] += weights->values[c_in * C_out * K * K + c_out * K * K + kh * K + kw] * grad_out_val;
                    grad_weights->values[c_in * C_out * K * K + c_out * K * K + kh * K + kw] += input->values[n * C_in * H_in * W_in + c_in * H_in * W_in + h_in_idx * W_in + w_in_idx] * grad_out_val;
                }
            }
        }
    }
}

void tensor_concatenate_backward(Tensor* grad_t1, Tensor* grad_t2, Tensor* grad_output, int axis, int c1_channels) {
    assert(axis == 1);
    int N = grad_output->shape[0], C_out = grad_output->shape[1], H = grad_output->shape[2], W = grad_output->shape[3];
    int C1 = c1_channels, C2 = C_out - C1;
    size_t feature_map_size = (size_t)H * W;
    for (int n = 0; n < N; ++n) {
        memcpy(grad_t1->values + n * C1 * feature_map_size, grad_output->values + n * C_out * feature_map_size, C1 * feature_map_size * sizeof(float));
        memcpy(grad_t2->values + n * C2 * feature_map_size, grad_output->values + n * C_out * feature_map_size + C1 * feature_map_size, C2 * feature_map_size * sizeof(float));
    }
}
Tensor* tensor_transposed_conv2d(Tensor* input, Tensor* weights, Tensor* biases, int stride) {
    assert(input->dims == 4 && weights->dims == 4 && input->shape[1] == weights->shape[0]);
    int N = input->shape[0], C_in = input->shape[1], H_in = input->shape[2], W_in = input->shape[3];
    int C_out = weights->shape[1], K = weights->shape[2];
    int H_out = (H_in - 1) * stride + K;
    int W_out = (W_in - 1) * stride + K;

    Tensor* output = create_tensor((int[]) { N, C_out, H_out, W_out }, 4);
    for (int n = 0; n < N; ++n) for (int c_out = 0; c_out < C_out; ++c_out) for (int h_out = 0; h_out < H_out; ++h_out) for (int w_out = 0; w_out < W_out; ++w_out) {
        float sum = 0.0f;
        for (int c_in = 0; c_in < C_in; ++c_in) for (int kh = 0; kh < K; ++kh) for (int kw = 0; kw < K; ++kw) {
            int h_in_idx = (h_out - kh), w_in_idx = (w_out - kw);
            if (h_in_idx >= 0 && h_in_idx % stride == 0 && w_in_idx >= 0 && w_in_idx % stride == 0) {
                h_in_idx /= stride; w_in_idx /= stride;
                if (h_in_idx < H_in && w_in_idx < W_in) {
                    sum += input->values[n * C_in * H_in * W_in + c_in * H_in * W_in + h_in_idx * W_in + w_in_idx] * weights->values[c_in * C_out * K * K + c_out * K * K + kh * K + kw];
                }
            }
        }
        output->values[n * C_out * H_out * W_out + c_out * H_out * W_out + h_out * W_out + w_out] = sum + biases->values[c_out];
    }
    return output;
}