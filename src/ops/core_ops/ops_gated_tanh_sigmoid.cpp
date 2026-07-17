#include "ops/ops.h"
#include "ggml-impl.h"

struct ggml_tensor* ggml_ops_gated_tanh_sigmoid(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    int hidden_channels,
    ggml_backend_t backend
) {
    struct ggml_tensor* srcs[] = { x };
    int32_t params[] = { hidden_channels };
    // 2. Otherwise check if it supports direct handler execution (virtual node)
    if (ggml_ops_backend_supports_op(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_GATED_TANH_SIGMOID,
                                     srcs, 1, params, sizeof(params))) {
        const int64_t ne[4] = { hidden_channels, x->ne[1], x->ne[2], x->ne[3] };
        struct ggml_tensor* result = ggml_ops_ext::ops_new_virtual_node(
            ctx,
            ggml_ops_ext::GGML_OP_OPS_VIRT_GATED_TANH_SIGMOID,
            x->type,
            ggml_n_dims(x),
            ne,
            1,
            srcs
        );
        ggml_set_op_params(result, params, sizeof(params));
        return result;
    }

    // 3. Default fallback (cast to F32 for CPU calculation, then cast back if needed)
    struct ggml_tensor* x_f32 = (x->type == GGML_TYPE_F32) ? x : ggml_cast(ctx, x, GGML_TYPE_F32);
    int64_t C = hidden_channels;
    int64_t rows = x_f32->ne[1] * x_f32->ne[2] * x_f32->ne[3];
    struct ggml_tensor* x_2d = ggml_reshape_2d(ctx, x_f32, x_f32->ne[0], rows);
    size_t stride = x_2d->ne[0] * sizeof(float);

    struct ggml_tensor* t_act = ggml_view_2d(ctx, x_2d, C, rows, stride, 0);
    struct ggml_tensor* s_act = ggml_view_2d(ctx, x_2d, C, rows, stride, C * sizeof(float));

    struct ggml_tensor* tanh_part = ggml_tanh(ctx, ggml_cont(ctx, t_act));
    struct ggml_tensor* sigm_part = ggml_sigmoid(ctx, ggml_cont(ctx, s_act));
    struct ggml_tensor* mul_part = ggml_mul(ctx, tanh_part, sigm_part);

    struct ggml_tensor* res_f32 = ggml_reshape_4d(ctx, mul_part, C, x_f32->ne[1], x_f32->ne[2], x_f32->ne[3]);
    return (x->type == GGML_TYPE_F32) ? res_f32 : ggml_cast(ctx, res_f32, x->type);
}
