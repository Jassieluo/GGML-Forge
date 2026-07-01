#include "ops/ops.h"
#include "ggml-impl.h"

struct ggml_tensor* ggml_ops_snake_beta(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* alpha,
    struct ggml_tensor* beta,
    ggml_backend_t backend
) {
    // 1. Check if the backend registers a custom builder
    ggml_ops_ext::ops_op_builder_t builder = ggml_ops_ext::find_ops_builder(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_SNAKE_BETA);
    if (builder) {
        struct ggml_tensor* srcs[] = { x, alpha, beta };
        return builder(ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_SNAKE_BETA, srcs, 3, nullptr, 0, backend);
    }

    // 2. Otherwise check if it supports direct handler execution (virtual node)
    if (ggml_ops_backend_supports_op(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_SNAKE_BETA)) {
        struct ggml_tensor* srcs[] = { x, alpha, beta };
        struct ggml_tensor* result = ggml_ops_ext::ops_new_virtual_node(ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_SNAKE_BETA, x->type, ggml_n_dims(x), x->ne, 3, srcs);
        return result;
    }

    // 3. Default fallback: CPU execution using standard GGML node composition
    struct ggml_tensor* alpha_f32 = ggml_ops_ext::force_w_f32(ctx, alpha);
    struct ggml_tensor* beta_f32 = ggml_ops_ext::force_w_f32(ctx, beta);

    int64_t C = x->ne[1];
    struct ggml_tensor* alpha_2d = ggml_reshape_2d(ctx, alpha_f32, 1, C);
    struct ggml_tensor* alpha_repeated = ggml_repeat(ctx, alpha_2d, x);

    struct ggml_tensor* beta_2d = ggml_reshape_2d(ctx, beta_f32, 1, C);
    struct ggml_tensor* beta_repeated = ggml_repeat(ctx, beta_2d, x);

    struct ggml_tensor* x_alpha = ggml_mul(ctx, x, alpha_repeated);
    struct ggml_tensor* sin_val = ggml_sin(ctx, x_alpha);
    struct ggml_tensor* sin_sq = ggml_mul(ctx, sin_val, sin_val);
    struct ggml_tensor* term = ggml_div(ctx, sin_sq, beta_repeated);
    return ggml_add(ctx, x, term);
}
