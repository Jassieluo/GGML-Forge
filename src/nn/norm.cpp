#include "nn/nn.h"

namespace nn {

namespace functional {

struct ggml_tensor* layer_norm(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* gamma, struct ggml_tensor* beta, float eps, ggml_backend_t backend) {
    return ggml_ops_layer_norm(ctx, x, gamma, beta, eps, backend);
}

struct ggml_tensor* interpolate_nearest_2x(struct ggml_context* ctx, struct ggml_tensor* x) {
    int64_t C = x->ne[0];
    int64_t T = x->ne[1];
    struct ggml_tensor* x_3d = ggml_reshape_3d(ctx, x, C, 1, T);
    struct ggml_tensor* target = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, 2, T);
    struct ggml_tensor* repeated = ggml_repeat(ctx, x_3d, target);
    return ggml_cont(ctx, ggml_reshape_2d(ctx, repeated, C, T * 2));
}

} // namespace functional

struct ggml_tensor* LayerNorm::forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend) {
    ggml_backend_t b = backend ? backend : this->backend;
    return F::layer_norm(ctx, x, gamma.local_tensor(), beta.local_tensor(), eps, b);
}

// InstanceNorm

struct ggml_tensor* InstanceNorm::forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend) {
    ggml_backend_t b = backend ? backend : this->backend;
    return ggml_ops_instance_norm(ctx, x, gamma.local_tensor(), beta.local_tensor(), eps, b);
}

// AdaLN
struct ggml_tensor* AdaLN::forward(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* scale,
    struct ggml_tensor* shift,
    ggml_backend_t backend
) {
    ggml_backend_t b = backend ? backend : this->backend;
    return ggml_ops_ada_ln(ctx, x, scale, shift, eps, b);
}

// AdaLayerNormZero

AdaLayerNormZero::Output AdaLayerNormZero::forward(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* emb,
    ggml_backend_t backend
) {
    ggml_backend_t b = backend ? backend : this->backend;
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

    struct ggml_tensor* x_modulated = ggml_ops_ada_ln(ctx, x, scale_msa, shift_msa, eps, b);

    return { x_modulated, gate_msa, shift_mlp, scale_mlp, gate_mlp };
}

// Snake
struct ggml_tensor* Snake::forward(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    ggml_backend_t backend
) {
    ggml_backend_t b = backend ? backend : this->backend;
    return ggml_ops_snake(ctx, x, alpha, b);
}

} // namespace nn
