// conv_bwd_weights_primitive.c

#include "conv_bwd_weights_primitive.h"
#include <stdio.h>
#include <stdlib.h>
#include <dnnl.h>

// extern symbols from conv2d_primitive.c
extern dnnl_engine_t eng;
extern dnnl_stream_t strm;
extern const char* dnnl_status2str(dnnl_status_t);

#define CHECK(stmt) do {                                       \
    dnnl_status_t _s = (stmt);                                 \
    if (_s != dnnl_success) {                                  \
      fprintf(stderr, "[%s:%d] %s -> %s (%d)\n",               \
              __FILE__,__LINE__,#stmt,                         \
              dnnl_status2str(_s),(int)_s); exit(1);           \
    }                                                          \
} while(0)

void conv_bwd_weights_primitive_init(
    ConvBwdWeightsPrimitive* cp,
    int N, int C, int H, int W,
    int F, int K,
    int stride, int padding
) {
    cp->N = N; cp->C = C; cp->H = H; cp->W = W;
    cp->F = F; cp->K = K;
    cp->stride = stride;
    cp->padding = padding;

    int H_out = (H - K + 2 * padding) / stride + 1;
    int W_out = (W - K + 2 * padding) / stride + 1;

    dnnl_dims_t src_dims = { N, C, H,    W };
    dnnl_dims_t diff_weights_dims = { F, C, K,    K };
    dnnl_dims_t diff_dst_dims = { N, F, H_out,W_out };
    dnnl_dims_t strides_dims = { stride, stride };
    dnnl_dims_t pads = { padding, padding };

    // <<< PLAIN FORMATS: NCHW/OIHW >>>
    CHECK(dnnl_memory_desc_init_by_tag(
        &cp->src_md, 4, src_dims, dnnl_f32, dnnl_nchw));
    CHECK(dnnl_memory_desc_init_by_tag(
        &cp->diff_weights_md, 4, diff_weights_dims, dnnl_f32, dnnl_oihw));
    CHECK(dnnl_memory_desc_init_by_tag(
        &cp->diff_dst_md, 4, diff_dst_dims, dnnl_f32, dnnl_nchw));

    // backward?weights descriptor
    dnnl_convolution_desc_t bwd_w_desc;
    CHECK(dnnl_convolution_backward_weights_desc_init(
        &bwd_w_desc,
        dnnl_convolution_direct,
        &cp->src_md, &cp->diff_weights_md, NULL, &cp->diff_dst_md,
        strides_dims, pads, pads));

    CHECK(dnnl_primitive_desc_create(&cp->pd, &bwd_w_desc, NULL, eng, NULL));
    CHECK(dnnl_primitive_create(&cp->prim, cp->pd));

    // memory objects
    CHECK(dnnl_memory_create(
        &cp->src_mem, &cp->src_md, eng, NULL));
    CHECK(dnnl_memory_create(
        &cp->diff_weights_mem, &cp->diff_weights_md, eng, NULL));
    CHECK(dnnl_memory_create(
        &cp->diff_dst_mem, &cp->diff_dst_md, eng, NULL));
}

Tensor* conv_bwd_weights_primitive_execute(
    ConvBwdWeightsPrimitive* cp,
    const Tensor* input,       // [N,C,H,W] NCHW
    const Tensor* grad_output  // [N,F,H_out,W_out] NCHW
) {
    // bind Src & Diff_Dst
    CHECK(dnnl_memory_set_data_handle(
        cp->src_mem, (void*)input->values));
    CHECK(dnnl_memory_set_data_handle(
        cp->diff_dst_mem, (void*)grad_output->values));

    // allocate & bind Diff_Weights
    Tensor* grad_w = create_tensor(
        (int[]) {
        cp->F, cp->C, cp->K, cp->K
    }, 4);
    CHECK(dnnl_memory_set_data_handle(
        cp->diff_weights_mem, (void*)grad_w->values));

    dnnl_exec_arg_t args[] = {
        {DNNL_ARG_SRC,           cp->src_mem},
        {DNNL_ARG_DIFF_DST,      cp->diff_dst_mem},
        {DNNL_ARG_DIFF_WEIGHTS,  cp->diff_weights_mem},
    };
    CHECK(dnnl_primitive_execute(cp->prim, strm, 3, args));
    CHECK(dnnl_stream_wait(strm));
    return grad_w;
}

void conv_bwd_weights_primitive_destroy(ConvBwdWeightsPrimitive* cp) {
    dnnl_primitive_destroy(cp->prim);
    dnnl_primitive_desc_destroy(cp->pd);
    dnnl_memory_destroy(cp->src_mem);
    dnnl_memory_destroy(cp->diff_weights_mem);
    dnnl_memory_destroy(cp->diff_dst_mem);
}
