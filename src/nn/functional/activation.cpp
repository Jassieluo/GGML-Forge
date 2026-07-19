#include "nn/functional/activation.h"

#include "ops/ops.h"

namespace nn::functional {

struct ggml_tensor* mish(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    ggml_backend_t backend
) {
    return ggml_ops_mish(ctx, x, backend);
}

ggml_tensor* swiglu(ggml_context* ctx, ggml_tensor* input, int axis) {
    return ggml_ops_gated_activation(ctx, input, ggml_ops_gate_activation::silu, axis);
}

ggml_tensor* geglu(ggml_context* ctx, ggml_tensor* input, int axis) {
    return ggml_ops_gated_activation(ctx, input, ggml_ops_gate_activation::gelu, axis);
}

ggml_tensor* reglu(ggml_context* ctx, ggml_tensor* input, int axis) {
    return ggml_ops_gated_activation(ctx, input, ggml_ops_gate_activation::relu, axis);
}

struct ggml_tensor* gated_tanh_sigmoid(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    int channels,
    ggml_backend_t backend
) {
    return ggml_ops_gated_tanh_sigmoid(ctx, x, channels, backend);
}

} // namespace nn::functional
