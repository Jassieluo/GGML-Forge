#include "nn/nn.h"

namespace nn {

// GLU
struct ggml_tensor* GLU::forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend) {
    return ggml_ops_glu(ctx, x, backend);
}

// FeedForward
FeedForward::FeedForward(
    struct ggml_tensor* w1_w, struct ggml_tensor* w1_b,
    struct ggml_tensor* w2_w, struct ggml_tensor* w2_b,
    ActivationType act
) : w1(w1_w, w1_b), w2(w2_w, w2_b), act_type(act) {}

struct ggml_tensor* FeedForward::forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend) {
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
            h = ggml_ops_mish(ctx, h, backend);
            break;
        case ActivationType::DOUBLE_SWISH:
            h = ggml_ops_double_swish(ctx, h, backend);
            break;
        case ActivationType::SILU:
            h = ggml_silu(ctx, h);
            break;
    }
    return w2.forward(ctx, h);
}

} // namespace nn
