#ifndef CONV_BWD_DATA_PRIMITIVE_H
#define CONV_BWD_DATA_PRIMITIVE_H

#include <dnnl.h>
#include "tensor.h"

typedef struct {
    dnnl_memory_desc_t diff_src_md;
    dnnl_memory_desc_t w_md;
    dnnl_memory_desc_t diff_dst_md;
    dnnl_primitive_desc_t pd;
    dnnl_primitive_t prim;
    dnnl_memory_t diff_src_mem;
    dnnl_memory_t w_mem;
    dnnl_memory_t diff_dst_mem;
    // shapes
    int N, C, H, W;
    int F, K, stride, padding;
} ConvBwdDataPrimitive;

void conv_bwd_data_primitive_init(
    ConvBwdDataPrimitive* cp,
    int N, int C, int H, int W,
    int F, int K,
    int stride, int padding
);

Tensor* conv_bwd_data_primitive_execute(
    ConvBwdDataPrimitive* cp,
    const Tensor* grad_output,
    const Tensor* weights
);

void conv_bwd_data_primitive_destroy(ConvBwdDataPrimitive* cp);

#endif
