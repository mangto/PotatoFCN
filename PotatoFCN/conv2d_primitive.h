#ifndef CONV2D_PRIMITIVE_H
#define CONV2D_PRIMITIVE_H

#include <dnnl.h>
#include "tensor.h"

// A cached primitive for a single conv2d layer.
typedef struct {
    // oneDNN descriptors & primitive
    dnnl_memory_desc_t src_md;
    dnnl_memory_desc_t w_md;
    dnnl_memory_desc_t b_md;
    dnnl_memory_desc_t dst_md;
    dnnl_primitive_desc_t pd;
    dnnl_primitive_t prim;
    // oneDNN memory objects (we rebind the data handle each exec)
    dnnl_memory_t src_mem;
    dnnl_memory_t w_mem;
    dnnl_memory_t b_mem;
    dnnl_memory_t dst_mem;
    // layer parameters for output shape calc
    int N, C, H, W;
    int F, K;
    int stride, padding;
} Conv2DPrimitive;

// Initialize engine & stream (call before any primitives)
void init_dnnl_engine();

// Build & cache a conv2d primitive for fixed shapes
void conv2d_primitive_init(
    Conv2DPrimitive* cp,
    int N, int C, int H, int W,       // input shape
    int F, int K,                     // weight shape: [F,C,K,K]
    int stride, int padding           // conv params
);

// Execute the primitive with user tensors; returns a newly‐allocated output
Tensor* conv2d_primitive_execute(
    Conv2DPrimitive* cp,
    const Tensor* input,    // [N,C,H,W]
    const Tensor* weights,  // [F,C,K,K]
    const Tensor* bias      // [F]
);

// Destroy / free all oneDNN objects held in the primitive
void conv2d_primitive_destroy(Conv2DPrimitive* cp);

#endif // CONV2D_PRIMITIVE_H
