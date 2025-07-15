#include "conv_bwd_data_primitive.h"
#include <stdio.h>
#include <stdlib.h>
#include <dnnl.h>

// extern’d once in conv2d_primitive.c
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

void conv_bwd_data_primitive_init(
    ConvBwdDataPrimitive* cp,
    int N, int C, int H, int W,
    int F, int K,
    int stride, int padding
) {
    cp->N = N; cp->C = C; cp->H = H; cp->W = W;
    cp->F = F; cp->K = K;
    cp->stride = stride;
    cp->padding = padding;

    // compute output spatial dims
    int H_out = (H - K + 2 * padding) / stride + 1;
    int W_out = (W - K + 2 * padding) / stride + 1;

    dnnl_dims_t diff_src_dims = { N, C, H,    W };
    dnnl_dims_t w_dims = { F, C, K,    K };
    dnnl_dims_t diff_dst_dims = { N, F, H_out,W_out };
    dnnl_dims_t strides_dims = { stride, stride };
    dnnl_dims_t pads = { padding, padding };

    // <<< USE PLAIN NCHW/OIHW TAGS HERE >>>
    CHECK(dnnl_memory_desc_init_by_tag(
        &cp->diff_src_md, 4, diff_src_dims, dnnl_f32, dnnl_nchw));
    CHECK(dnnl_memory_desc_init_by_tag(
        &cp->w_md, 4, w_dims, dnnl_f32, dnnl_oihw));
    CHECK(dnnl_memory_desc_init_by_tag(
        &cp->diff_dst_md, 4, diff_dst_dims, dnnl_f32, dnnl_nchw));

    // backward‑data desc
    dnnl_convolution_desc_t bwd_desc;
    CHECK(dnnl_convolution_backward_data_desc_init(
        &bwd_desc,
        dnnl_convolution_direct,
        &cp->diff_src_md,
        &cp->w_md,
        &cp->diff_dst_md,
        strides_dims, pads, pads));

    CHECK(dnnl_primitive_desc_create(&cp->pd, &bwd_desc, NULL, eng, NULL));
    CHECK(dnnl_primitive_create(&cp->prim, cp->pd));

    // create memory objs (bind data at exec time)
    CHECK(dnnl_memory_create(
        &cp->diff_src_mem, &cp->diff_src_md, eng, NULL));
    CHECK(dnnl_memory_create(
        &cp->w_mem, &cp->w_md, eng, NULL));
    CHECK(dnnl_memory_create(
        &cp->diff_dst_mem, &cp->diff_dst_md, eng, NULL));
}

Tensor* conv_bwd_data_primitive_execute(
    ConvBwdDataPrimitive* cp,
    const Tensor* grad_output,  // shape [N,F,H_out,W_out] in NCHW
    const Tensor* weights       // shape [F,C,K,K] in OIHW
) {
    // bind pointers
    CHECK(dnnl_memory_set_data_handle(
        cp->diff_dst_mem, (void*)grad_output->values));
    CHECK(dnnl_memory_set_data_handle(
        cp->w_mem, (void*)weights->values));

    // allocate grad_input
    Tensor* grad_input = create_tensor(
        (int[]) {
        cp->N, cp->C, cp->H, cp->W
    }, 4);
    CHECK(dnnl_memory_set_data_handle(
        cp->diff_src_mem, (void*)grad_input->values));

    // exec
    dnnl_exec_arg_t args[] = {
        { DNNL_ARG_DIFF_DST,      cp->diff_dst_mem },
        { DNNL_ARG_WEIGHTS,       cp->w_mem        },
        { DNNL_ARG_DIFF_SRC,      cp->diff_src_mem },
    };
    CHECK(dnnl_primitive_execute(cp->prim, strm, 3, args));
    CHECK(dnnl_stream_wait(strm));
    return grad_input;
}

void conv_bwd_data_primitive_destroy(ConvBwdDataPrimitive* cp) {
    dnnl_primitive_destroy(cp->prim);
    dnnl_primitive_desc_destroy(cp->pd);
    dnnl_memory_destroy(cp->diff_src_mem);
    dnnl_memory_destroy(cp->w_mem);
    dnnl_memory_destroy(cp->diff_dst_mem);
}