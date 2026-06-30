#include "ops/ops.h"
#include "ggml-impl.h"
#include <cstring>

struct ggml_tensor* ggml_ops_layer_norm(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* gamma,
    struct ggml_tensor* beta,
    float eps,
    ggml_backend_t backend
) {
    // 1. Check if the backend registers a custom builder
    ggml_ops_ext::ops_op_builder_t builder = ggml_ops_ext::find_ops_builder(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_LAYER_NORM);
    if (builder) {
        struct ggml_tensor* srcs[] = { x, gamma, beta };
        int32_t params[1];
        std::memcpy(params, &eps, sizeof(float));
        return builder(ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_LAYER_NORM, srcs, 3, params, 1, backend);
    }

    // 2. Otherwise check if it supports direct handler execution (virtual node)
    bool supports = ggml_ops_backend_supports_op(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_LAYER_NORM);

    if (supports) {
        struct ggml_tensor* srcs[] = { x, gamma, beta };
        struct ggml_tensor* result = ggml_ops_ext::ops_new_virtual_node(
            ctx,
            ggml_ops_ext::GGML_OP_OPS_VIRT_LAYER_NORM,
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

    // 3. Default fallback (cast to F32 for CPU calculation, then cast back if needed)
    struct ggml_tensor* x_f32 = (x->type == GGML_TYPE_F32) ? x : ggml_cast(ctx, x, GGML_TYPE_F32);
    struct ggml_tensor* gamma_f32 = (gamma->type == GGML_TYPE_F32) ? gamma : ggml_cast(ctx, gamma, GGML_TYPE_F32);
    struct ggml_tensor* beta_f32 = (beta->type == GGML_TYPE_F32) ? beta : ggml_cast(ctx, beta, GGML_TYPE_F32);

    struct ggml_tensor* norm = ggml_norm(ctx, x_f32, eps);
    struct ggml_tensor* g_reshaped = (ggml_n_dims(gamma_f32) == 1) ? ggml_reshape_2d(ctx, gamma_f32, gamma_f32->ne[0], 1) : gamma_f32;
    struct ggml_tensor* b_reshaped = (ggml_n_dims(beta_f32) == 1) ? ggml_reshape_2d(ctx, beta_f32, beta_f32->ne[0], 1) : beta_f32;
    struct ggml_tensor* scaled = ggml_mul(ctx, norm, g_reshaped);
    struct ggml_tensor* res_f32 = ggml_add(ctx, scaled, b_reshaped);
    return (x->type == GGML_TYPE_F32) ? res_f32 : ggml_cast(ctx, res_f32, x->type);
}
