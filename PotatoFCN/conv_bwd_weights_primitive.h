// conv_bwd_weights_primitive.h
#ifndef CONV_BWD_WEIGHTS_PRIMITIVE_H
#define CONV_BWD_WEIGHTS_PRIMITIVE_H

#include <dnnl.h>
#include "tensor.h"

typedef struct {
    dnnl_memory_desc_t src_md, diff_weights_md, diff_dst_md;
    dnnl_primitive_desc_t pd;
    dnnl_primitive_t prim;
    dnnl_memory_t src_mem, diff_weights_mem, diff_dst_mem;
    int N, C, H, W, F, K, stride, padding;
} ConvBwdWeightsPrimitive;

void conv_bwd_weights_primitive_init(
    ConvBwdWeightsPrimitive* cp,
    int N, int C, int H, int W,
    int F, int K,
    int stride, int padding
);

Tensor* conv_bwd_weights_primitive_execute(
    ConvBwdWeightsPrimitive* cp,
    const Tensor* input,
    const Tensor* grad_output
);

void conv_bwd_weights_primitive_destroy(ConvBwdWeightsPrimitive* cp);

#endif
