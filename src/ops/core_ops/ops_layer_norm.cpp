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
    // The contract requires gamma and beta, and the fallback dereferences them.
    if (!x || !gamma || !beta) return nullptr;
    struct ggml_tensor* srcs[] = { x, gamma, beta };
    int32_t params[1];
    std::memcpy(params, &eps, sizeof(float));
    // 2. Otherwise check if it supports direct handler execution (virtual node)
    bool supports = ggml_ops_backend_supports_op(
        backend, ggml_ops_ext::GGML_OP_OPS_VIRT_LAYER_NORM, srcs, 3, params, sizeof(params));

    if (supports) {
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
