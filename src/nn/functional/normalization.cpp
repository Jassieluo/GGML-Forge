#include "nn/functional/normalization.h"

#include "ops/ops.h"

namespace nn::functional {

ggml_tensor* layer_norm(
    ggml_context* ctx, ggml_tensor* input, ggml_tensor* gamma,
    ggml_tensor* beta, float epsilon, ggml_backend_t backend
) {
    return ggml_ops_layer_norm(ctx, input, gamma, beta, epsilon, backend);
}

} // namespace nn::functional
