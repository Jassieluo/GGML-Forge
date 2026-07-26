#include "ops/ops.h"
#include "ggml-impl.h"
#include <cfloat>

struct ggml_tensor* ggml_ops_snake_beta(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* alpha,
    struct ggml_tensor* beta,
    ggml_backend_t backend
) {
    if (!x || !alpha || !beta) return nullptr;
    struct ggml_tensor* srcs[] = { x, alpha, beta };
    // 2. Otherwise check if it supports direct handler execution (virtual node)
    if (ggml_ops_backend_supports_op(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_SNAKE_BETA, srcs, 3)) {
        struct ggml_tensor* result = ggml_ops_ext::ops_new_virtual_node(ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_SNAKE_BETA, x->type, ggml_n_dims(x), x->ne, 3, srcs);
        return result;
    }

    // 3. Default fallback: y = x + sin^2(alpha*x) * sgn(beta)/max(|beta|, 1e-6),
    // composed from native GGML nodes and computed in F32 (ggml_mul/ggml_div do
    // not accept mixed float widths). Clamping |beta| away from zero mirrors
    // the kernel, which treats |beta| < 1e-6 as identity; beta == 0 yields
    // sgn == 0 and therefore an exact identity here as well.
    struct ggml_tensor* x_f32 = ggml_ops_ext::force_w_f32(ctx, x);
    struct ggml_tensor* alpha_f32 = ggml_ops_ext::force_w_f32(ctx, alpha);
    struct ggml_tensor* beta_f32 = ggml_ops_ext::force_w_f32(ctx, beta);
    if (!x_f32 || !alpha_f32 || !beta_f32) return nullptr;

    int64_t C = x->ne[1];
    struct ggml_tensor* alpha_2d = ggml_reshape_2d(ctx, alpha_f32, 1, C);
    struct ggml_tensor* alpha_repeated = ggml_repeat(ctx, alpha_2d, x_f32);

    struct ggml_tensor* inv_beta = ggml_div(
        ctx, ggml_sgn(ctx, beta_f32),
        ggml_clamp(ctx, ggml_abs(ctx, beta_f32), 1e-6f, FLT_MAX));
    struct ggml_tensor* inv_beta_2d = ggml_reshape_2d(ctx, inv_beta, 1, C);
    struct ggml_tensor* inv_beta_repeated = ggml_repeat(ctx, inv_beta_2d, x_f32);

    struct ggml_tensor* x_alpha = ggml_mul(ctx, x_f32, alpha_repeated);
    struct ggml_tensor* sin_val = ggml_sin(ctx, x_alpha);
    struct ggml_tensor* sin_sq = ggml_mul(ctx, sin_val, sin_val);
    struct ggml_tensor* term = ggml_mul(ctx, sin_sq, inv_beta_repeated);
    struct ggml_tensor* result = ggml_add(ctx, x_f32, term);
    return x->type == GGML_TYPE_F32 ? result
                                    : ggml_cont(ctx, ggml_cast(ctx, result, x->type));
}
