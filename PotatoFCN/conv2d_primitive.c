// conv2d_primitive.c

#include "conv2d_primitive.h"
#include <stdio.h>
#include <stdlib.h>
#include <dnnl.h>

//-----------------------------------------------------------------------------
// dnnl_status_t → string
//-----------------------------------------------------------------------------
static const char* dnnl_status2str(dnnl_status_t s) {
    switch (s) {
    case dnnl_success:           return "dnnl_success";
    case dnnl_out_of_memory:     return "dnnl_out_of_memory";
    case dnnl_invalid_arguments: return "dnnl_invalid_arguments";
    case dnnl_unimplemented:     return "dnnl_unimplemented";
    case dnnl_runtime_error:     return "dnnl_runtime_error";
#ifdef dnnl_not_required
    case dnnl_not_required:      return "dnnl_not_required";
#endif
    default:                     return "unknown dnnl_status_t";
    }
}

//-----------------------------------------------------------------------------
// CHECK 매크로
//-----------------------------------------------------------------------------
#define CHECK(stmt)                                                         \
    do {                                                                     \
        dnnl_status_t _s = (stmt);                                           \
        if (_s != dnnl_success) {                                            \
            fprintf(stderr,                                                  \
                "[%s:%d] %s -> %s (%d)\n",                                  \
                __FILE__, __LINE__, #stmt,                                   \
                dnnl_status2str(_s), (int)_s);                               \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

//-----------------------------------------------------------------------------
// 전역 엔진/스트림
//-----------------------------------------------------------------------------
dnnl_engine_t eng = NULL;
dnnl_stream_t strm = NULL;

void init_dnnl_engine() {
    if (!eng)  CHECK(dnnl_engine_create(&eng, dnnl_cpu, 0));
    if (!strm) CHECK(dnnl_stream_create(&strm, eng, dnnl_stream_default_flags));
}

//-----------------------------------------------------------------------------
// primitive Init: blocked 포맷 + Conv→ReLU Fusion
//-----------------------------------------------------------------------------
void conv2d_primitive_init(
    Conv2DPrimitive* cp,
    int N, int C, int H, int W,
    int F, int K,
    int stride, int padding
) {
    init_dnnl_engine();

    // 모델 파라미터 저장
    cp->N = N; cp->C = C; cp->H = H; cp->W = W;
    cp->F = F; cp->K = K;
    cp->stride = stride;
    cp->padding = padding;

    // 출력 공간 계산
    int H_out = (H - K + 2 * padding) / stride + 1;
    int W_out = (W - K + 2 * padding) / stride + 1;

    // dims 배열
    dnnl_dims_t src_dims = { N, C,    H,    W };
    dnnl_dims_t weight_dims = { F, C,    K,    K };
    dnnl_dims_t bias_dims = { F };
    dnnl_dims_t dst_dims = { N, F,    H_out,W_out };
    dnnl_dims_t strides_dims = { stride, stride };
    dnnl_dims_t pad_dims = { padding, padding };

    // 메모리 디스크립터: 블록 포맷으로 변경
    // C채널을 16개씩 묶는 nChw16c, OIhw16o16i 블록 가중치
    CHECK(dnnl_memory_desc_init_by_tag(
        &cp->src_md, 4, src_dims, dnnl_f32, dnnl_nChw16c));
    CHECK(dnnl_memory_desc_init_by_tag(
        &cp->w_md, 4, weight_dims, dnnl_f32, dnnl_OIhw16o16i));
    CHECK(dnnl_memory_desc_init_by_tag(
        &cp->b_md, 1, bias_dims, dnnl_f32, dnnl_x));
    CHECK(dnnl_memory_desc_init_by_tag(
        &cp->dst_md, 4, dst_dims, dnnl_f32, dnnl_nChw16c));

    // Post‑ops: Conv + ReLU 합치기
    dnnl_post_ops_t ops;
    CHECK(dnnl_post_ops_create(&ops));
    CHECK(dnnl_post_ops_append_eltwise(
        ops, 1.0f,
        dnnl_eltwise_relu,
        0.0f, 0.0f));

    dnnl_primitive_attr_t attr;
    CHECK(dnnl_primitive_attr_create(&attr));
    CHECK(dnnl_primitive_attr_set_post_ops(attr, ops));
    dnnl_post_ops_destroy(ops);

    // Conv forward descriptor & primitive (attr 포함)
    dnnl_convolution_desc_t conv_desc;
    CHECK(dnnl_convolution_forward_desc_init(
        &conv_desc,
        dnnl_forward_inference,
        dnnl_convolution_direct,
        &cp->src_md, &cp->w_md, &cp->b_md, &cp->dst_md,
        strides_dims, pad_dims, pad_dims));

    CHECK(dnnl_primitive_desc_create(
        &cp->pd, &conv_desc, attr, eng, NULL));
    CHECK(dnnl_primitive_create(&cp->prim, cp->pd));
    dnnl_primitive_attr_destroy(attr);

    // 메모리 객체 생성 (데이터는 execute 시 바인딩)
    CHECK(dnnl_memory_create(&cp->src_mem, &cp->src_md, eng, NULL));
    CHECK(dnnl_memory_create(&cp->w_mem, &cp->w_md, eng, NULL));
    CHECK(dnnl_memory_create(&cp->b_mem, &cp->b_md, eng, NULL));
    CHECK(dnnl_memory_create(&cp->dst_mem, &cp->dst_md, eng, NULL));
}

//-----------------------------------------------------------------------------
// Execute: 데이터 핸들러만 바꿔가며 커널 실행
//-----------------------------------------------------------------------------
Tensor* conv2d_primitive_execute(
    Conv2DPrimitive* cp,
    const Tensor* input,
    const Tensor* weights,
    const Tensor* bias
) {
    // 데이터 바인딩
    CHECK(dnnl_memory_set_data_handle(cp->src_mem, (void*)input->values));
    CHECK(dnnl_memory_set_data_handle(cp->w_mem, (void*)weights->values));
    CHECK(dnnl_memory_set_data_handle(cp->b_mem, (void*)bias->values));

    // 출력 텐서 할당 및 바인딩
    int H_out = (cp->H - cp->K + 2 * cp->padding) / cp->stride + 1;
    int W_out = (cp->W - cp->K + 2 * cp->padding) / cp->stride + 1;
    Tensor* output = create_tensor(
        (int[]) {
        cp->N, cp->F, H_out, W_out
    }, 4);
    CHECK(dnnl_memory_set_data_handle(
        cp->dst_mem, (void*)output->values));

    // 실행
    dnnl_exec_arg_t args[] = {
        {DNNL_ARG_SRC,     cp->src_mem},
        {DNNL_ARG_WEIGHTS, cp->w_mem},
        {DNNL_ARG_BIAS,    cp->b_mem},
        {DNNL_ARG_DST,     cp->dst_mem},
    };
    CHECK(dnnl_primitive_execute(cp->prim, strm, 4, args));
    CHECK(dnnl_stream_wait(strm));

    return output;
}

//-----------------------------------------------------------------------------
// Destroy
//-----------------------------------------------------------------------------
void conv2d_primitive_destroy(Conv2DPrimitive* cp) {
    dnnl_primitive_destroy(cp->prim);
    dnnl_primitive_desc_destroy(cp->pd);
    dnnl_memory_destroy(cp->src_mem);
    dnnl_memory_destroy(cp->w_mem);
    dnnl_memory_destroy(cp->b_mem);
    dnnl_memory_destroy(cp->dst_mem);
}
