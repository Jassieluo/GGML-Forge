#pragma once

#include "ggml.h"
#include "ggml-backend.h"

namespace tts {
namespace ops {

struct ops_backend_interface {
    // Backend name prefix, e.g. "CUDA" or "SYCL"
    const char* backend_name_prefix;

    // Custom conv1d implementation
    bool (*compute_conv_1d)(
        ggml_backend_t backend,
        struct ggml_tensor* w,
        struct ggml_tensor* x,
        struct ggml_tensor* dst,
        int stride,
        int padding,
        int dilation
    );

    // Custom conv_transpose_1d implementation
    bool (*compute_conv_transpose_1d)(
        ggml_backend_t backend,
        struct ggml_tensor* w,
        struct ggml_tensor* x,
        struct ggml_tensor* dst,
        int stride,
        int padding,
        int dilation
    );

    // Custom softmax implementation
    bool (*compute_softmax)(
        ggml_backend_t backend,
        struct ggml_tensor* dst
    );
};

// Main lifecycle registry
void install_ops_hook(ggml_backend_t backend);
void uninstall_ops_hook(ggml_backend_t backend);

// Custom wrapper that handles 1D transposed convolution with padding natively on GPU
// and falls back to padding=0 + cropping on CPU.
struct ggml_tensor* ops_conv_transpose_1d(
    struct ggml_context* ctx,
    struct ggml_tensor* w,
    struct ggml_tensor* x,
    int stride,
    int padding,
    int dilation
);

// Backend registration (called by individual backend modules)
void register_ops_backend(const ops_backend_interface& iface);

} // namespace ops
} // namespace tts
