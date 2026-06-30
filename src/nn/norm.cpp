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

// AdaLN
struct ggml_tensor* AdaLN::forward(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* scale,
    struct ggml_tensor* shift,
    ggml_backend_t backend
) {
    return ggml_ops_ada_ln(ctx, x, scale, shift, eps, backend);
}

// AdaLayerNormZero
AdaLayerNormZero::AdaLayerNormZero(struct ggml_tensor* linear_w, struct ggml_tensor* linear_b, float eps)
    : linear(linear_w, linear_b), norm(nullptr, nullptr, eps), eps(eps) {}

AdaLayerNormZero::Output AdaLayerNormZero::forward(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* emb,
    ggml_backend_t backend
) {
    struct ggml_tensor* emb_silu = ggml_silu(ctx, emb);
    struct ggml_tensor* emb_proj = linear.forward(ctx, emb_silu);

    int64_t dim = emb_proj->ne[0] / 6;
    int64_t batch = emb_proj->ne[1];
    size_t element_size = ggml_element_size(emb_proj);
    size_t row_stride = emb_proj->nb[1];

    struct ggml_tensor* shift_msa = ggml_view_2d(ctx, emb_proj, dim, batch, row_stride, 0 * dim * element_size);
    struct ggml_tensor* scale_msa = ggml_view_2d(ctx, emb_proj, dim, batch, row_stride, 1 * dim * element_size);
    struct ggml_tensor* gate_msa  = ggml_view_2d(ctx, emb_proj, dim, batch, row_stride, 2 * dim * element_size);
    struct ggml_tensor* shift_mlp = ggml_view_2d(ctx, emb_proj, dim, batch, row_stride, 3 * dim * element_size);
    struct ggml_tensor* scale_mlp = ggml_view_2d(ctx, emb_proj, dim, batch, row_stride, 4 * dim * element_size);
    struct ggml_tensor* gate_mlp  = ggml_view_2d(ctx, emb_proj, dim, batch, row_stride, 5 * dim * element_size);

    struct ggml_tensor* x_modulated = ggml_ops_ada_ln(ctx, x, scale_msa, shift_msa, eps, backend);

    return { x_modulated, gate_msa, shift_mlp, scale_mlp, gate_mlp };
}

// Snake
struct ggml_tensor* Snake::forward(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    ggml_backend_t backend
) {
    return ggml_ops_snake(ctx, x, alpha, backend);
}

} // namespace nn
