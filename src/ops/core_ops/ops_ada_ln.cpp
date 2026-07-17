#include "ops/ops.h"
#include "ggml-impl.h"
#include <cstring>

struct ggml_tensor* ggml_ops_ada_ln(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* scale,
    struct ggml_tensor* shift,
    float eps,
    ggml_backend_t backend
) {
    struct ggml_tensor* srcs[] = { x, scale, shift };
    int32_t params[1];
    std::memcpy(params, &eps, sizeof(float));
    // 2. Otherwise check if it supports direct handler execution (virtual node)
    bool supports = ggml_ops_backend_supports_op(
        backend, ggml_ops_ext::GGML_OP_OPS_VIRT_ADA_LN, srcs, 3, params, sizeof(params));

    if (supports) {
        struct ggml_tensor* result = ggml_ops_ext::ops_new_virtual_node(
            ctx,
            ggml_ops_ext::GGML_OP_OPS_VIRT_ADA_LN,
            x->type,
            ggml_n_dims(x),
            x->ne,
            3,
            srcs
        );
        float params[] = { eps };
        ggml_set_op_params(result, params, sizeof(params));
        return result;
    }

    // 3. Default fallback (implemented using primitive GGML nodes on target backend)
    // Formula: y = LayerNorm(x) * (1 + scale) + shift
    struct ggml_tensor* x_f32 = (x->type == GGML_TYPE_F32) ? x : ggml_cast(ctx, x, GGML_TYPE_F32);
    struct ggml_tensor* scale_f32 = (scale->type == GGML_TYPE_F32) ? scale : ggml_cast(ctx, scale, GGML_TYPE_F32);
    struct ggml_tensor* shift_f32 = (shift->type == GGML_TYPE_F32) ? shift : ggml_cast(ctx, shift, GGML_TYPE_F32);

    // Standard LayerNorm (mean=0, var=1) without affine parameters
    struct ggml_tensor* norm = ggml_norm(ctx, x_f32, eps);

    // Reshape scale and shift if they don't have sequence dim.
    struct ggml_tensor* scale_reshaped = scale_f32;
    struct ggml_tensor* shift_reshaped = shift_f32;
    if (ggml_n_dims(scale_f32) == 2) {
        scale_reshaped = ggml_reshape_3d(ctx, scale_f32, scale_f32->ne[0], 1, scale_f32->ne[1]);
    }
    if (ggml_n_dims(shift_f32) == 2) {
        shift_reshaped = ggml_reshape_3d(ctx, shift_f32, shift_f32->ne[0], 1, shift_f32->ne[1]);
    }

    // scale_plus_1 = 1 + scale
    struct ggml_tensor* ones = ggml_new_tensor(ctx, GGML_TYPE_F32, ggml_n_dims(scale_reshaped), scale_reshaped->ne);
    ones = ggml_fill(ctx, ones, 1.0f);
    struct ggml_tensor* scale_plus_1 = ggml_add(ctx, scale_reshaped, ones);

    // scaled = norm * scale_plus_1
    struct ggml_tensor* scaled = ggml_mul(ctx, norm, scale_plus_1);

    // output = scaled + shift
    struct ggml_tensor* res_f32 = ggml_add(ctx, scaled, shift_reshaped);

    return (x->type == GGML_TYPE_F32) ? res_f32 : ggml_cast(ctx, res_f32, x->type);
}
