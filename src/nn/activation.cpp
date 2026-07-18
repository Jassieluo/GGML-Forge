#include "nn/nn.h"

namespace nn::functional {

struct ggml_tensor* mish(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    ggml_backend_t backend
) {
    return ggml_ops_mish(ctx, x, backend);
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
