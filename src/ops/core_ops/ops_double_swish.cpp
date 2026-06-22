#include "ops/ops.h"
#include "ggml-impl.h"

struct ggml_tensor* ggml_ops_double_swish(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    ggml_backend_t backend
) {
    // Ensure x is FP32
    struct ggml_tensor* x_f32 = (x->type == GGML_TYPE_F32) ? x : ggml_cast(ctx, x, GGML_TYPE_F32);

    // 1. Check if the backend registers a custom builder
    ggml_ops_ext::ops_op_builder_t builder = ggml_ops_ext::find_ops_builder(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_DOUBLE_SWISH);
    if (builder) {
        struct ggml_tensor* srcs[] = { x_f32 };
        return builder(ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_DOUBLE_SWISH, srcs, 1, nullptr, 0, backend);
    }

    // 2. Otherwise check if it supports direct handler execution (virtual node)
    if (ggml_ops_backend_supports_op(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_DOUBLE_SWISH)) {
        struct ggml_tensor* srcs[] = { x_f32 };
        struct ggml_tensor* result = ggml_ops_ext::ops_new_virtual_node(
            ctx,
            ggml_ops_ext::GGML_OP_OPS_VIRT_DOUBLE_SWISH,
            GGML_TYPE_F32,
            ggml_n_dims(x_f32),
            x_f32->ne,
            1,
            srcs
        );
        return result;
    }

    // 3. Default fallback (mathematically identical: x * sigmoid(x - 1.0f))
    struct ggml_tensor* ones = ggml_new_tensor(ctx, GGML_TYPE_F32, ggml_n_dims(x_f32), x_f32->ne);
    ones = ggml_fill(ctx, ones, 1.0f);
    struct ggml_tensor* x_minus_1 = ggml_sub(ctx, x_f32, ones);
    struct ggml_tensor* sig = ggml_sigmoid(ctx, x_minus_1);
    return ggml_mul(ctx, x_f32, sig);
}
