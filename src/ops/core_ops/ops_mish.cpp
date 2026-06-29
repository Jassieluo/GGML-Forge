#include "ops/ops.h"
#include "ggml-impl.h"

struct ggml_tensor* ggml_ops_mish(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    ggml_backend_t backend
) {
    // 1. Check if the backend registers a custom builder
    ggml_ops_ext::ops_op_builder_t builder = ggml_ops_ext::find_ops_builder(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_MISH);
    if (builder) {
        struct ggml_tensor* srcs[] = { x };
        return builder(ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_MISH, srcs, 1, nullptr, 0, backend);
    }

    // 2. Otherwise check if it supports direct handler execution (virtual node)
    if (ggml_ops_backend_supports_op(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_MISH)) {
        struct ggml_tensor* srcs[] = { x };
        struct ggml_tensor* result = ggml_ops_ext::ops_new_virtual_node(
            ctx,
            ggml_ops_ext::GGML_OP_OPS_VIRT_MISH,
            x->type,
            ggml_n_dims(x),
            x->ne,
            1,
            srcs
        );
        return result;
    }

    // 3. Default fallback (cast to F32 for CPU calculation, then cast back if needed)
    struct ggml_tensor* x_f32 = (x->type == GGML_TYPE_F32) ? x : ggml_cast(ctx, x, GGML_TYPE_F32);
    struct ggml_tensor* exp_x = ggml_exp(ctx, x_f32);
    struct ggml_tensor* ones = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ones = ggml_fill(ctx, ones, 1.0f);
    struct ggml_tensor* sp = ggml_log(ctx, ggml_add(ctx, exp_x, ones));
    struct ggml_tensor* res_f32 = ggml_mul(ctx, x_f32, ggml_tanh(ctx, sp));
    return (x->type == GGML_TYPE_F32) ? res_f32 : ggml_cast(ctx, res_f32, x->type);
}
