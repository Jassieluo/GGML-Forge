#include "nn/nn.h"

namespace nn {

// LayerNorm
LayerNorm::LayerNorm(struct ggml_tensor* gamma, struct ggml_tensor* beta, float eps)
    : gamma(gamma), beta(beta), eps(eps) {}

struct ggml_tensor* LayerNorm::forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend) {
    return ggml_ops_layer_norm(ctx, x, gamma, beta, eps, backend);
}

// InstanceNorm
InstanceNorm::InstanceNorm(struct ggml_tensor* gamma, struct ggml_tensor* beta, float eps)
    : gamma(gamma), beta(beta), eps(eps) {}

struct ggml_tensor* InstanceNorm::forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend) {
    return ggml_ops_instance_norm(ctx, x, gamma, beta, eps, backend);
}

} // namespace nn
