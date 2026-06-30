#include "ops/ops.h"
#include "ggml-impl.h"
#include <cmath>
#include <cstring>

struct ggml_tensor* ggml_ops_snake(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    float alpha,
    ggml_backend_t backend
) {
    // 1. Check if the backend registers a custom builder
    ggml_ops_ext::ops_op_builder_t builder = ggml_ops_ext::find_ops_builder(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_SNAKE);
    if (builder) {
        struct ggml_tensor* srcs[] = { x };
        int32_t params[1];
        std::memcpy(params, &alpha, sizeof(float));
        return builder(ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_SNAKE, srcs, 1, params, 1, backend);
    }

    // 2. Otherwise check if it supports direct handler execution (virtual node)
    if (ggml_ops_backend_supports_op(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_SNAKE)) {
        struct ggml_tensor* srcs[] = { x };
        struct ggml_tensor* result = ggml_ops_ext::ops_new_virtual_node(
            ctx,
            ggml_ops_ext::GGML_OP_OPS_VIRT_SNAKE,
            x->type,
            ggml_n_dims(x),
            x->ne,
            1,
            srcs
        );
        float params[] = { alpha };
        ggml_set_op_params(result, params, sizeof(params));
        return result;
    }

    // 3. Default fallback (cast to F32 for CPU calculation, then cast back if needed)
    if (std::abs(alpha) < 1e-6f) {
        return x;
    }

    struct ggml_tensor* x_f32 = (x->type == GGML_TYPE_F32) ? x : ggml_cast(ctx, x, GGML_TYPE_F32);
    
    // Formula: x + (sin(alpha * x)^2) / alpha
    struct ggml_tensor* ax = ggml_scale(ctx, x_f32, alpha);
    struct ggml_tensor* sin_ax = ggml_sin(ctx, ax);
    struct ggml_tensor* sin_sq = ggml_mul(ctx, sin_ax, sin_ax);
    struct ggml_tensor* scaled = ggml_scale(ctx, sin_sq, 1.0f / alpha);
    struct ggml_tensor* res_f32 = ggml_add(ctx, x_f32, scaled);

    return (x->type == GGML_TYPE_F32) ? res_f32 : ggml_cast(ctx, res_f32, x->type);
}
