#include "nn/layers/feedforward.h"

#include "ops/ops.h"

namespace nn {

struct ggml_tensor* FeedForward::forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend) {
    ggml_backend_t b = backend ? backend : this->backend;
    struct ggml_tensor* h = w1.forward(ctx, x);
    switch (act_type) {
        case ActivationType::GELU:
            h = ggml_gelu(ctx, h);
            break;
        case ActivationType::GELU_ERF:
            h = ggml_gelu_erf(ctx, h);
            break;
        case ActivationType::RELU:
            h = ggml_relu(ctx, h);
            break;
        case ActivationType::LEAKY_RELU:
            h = ggml_leaky_relu(ctx, h, 0.1f, false);
            break;
        case ActivationType::MISH:
            h = ggml_ops_mish(ctx, h, b);
            break;
        case ActivationType::DOUBLE_SWISH:
            h = ggml_ops_double_swish(ctx, h, b);
            break;
        case ActivationType::SILU:
            h = ggml_silu(ctx, h);
            break;
    }
    return w2.forward(ctx, h);
}

} // namespace nn
