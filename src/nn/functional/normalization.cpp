#include "nn/functional/normalization.h"

#include "ops/ops.h"

namespace nn::functional {

ggml_tensor* layer_norm(
    ggml_context* ctx, ggml_tensor* input, ggml_tensor* gamma,
    ggml_tensor* beta, float epsilon, ggml_backend_t backend
) {
    if (gamma && beta) {
        return ggml_ops_layer_norm(ctx, input, gamma, beta, epsilon, backend);
    }
    // The fused op requires both affine tensors; LayerNorm declares them
    // optional, so compose the affine-free (or single-tensor) form natively.
    ggml_tensor* normed = ggml_norm(ctx, input, epsilon);
    if (gamma) normed = ggml_mul(ctx, normed, gamma);
    if (beta) normed = ggml_add(ctx, normed, beta);
    return normed;
}

ggml_tensor* rms_norm(ggml_context* ctx, ggml_tensor* input, ggml_tensor* weight, float epsilon) {
    return ggml_ops_rms_norm(ctx, input, weight, epsilon);
}

ggml_tensor* group_norm(
    ggml_context* ctx, ggml_tensor* input, int groups,
    ggml_tensor* weight, ggml_tensor* bias, float epsilon
) {
    return ggml_ops_group_norm(ctx, input, groups, weight, bias, epsilon);
}

ggml_tensor* l2_normalize(ggml_context* ctx, ggml_tensor* input, int axis, float epsilon) {
    return ggml_ops_l2_normalize(ctx, input, axis, epsilon);
}

} // namespace nn::functional
