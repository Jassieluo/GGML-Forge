#include "ops/ops.h"
#include "ggml-impl.h"

struct ggml_tensor* ggml_ops_instance_norm(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* gamma,
    struct ggml_tensor* beta,
    float eps,
    ggml_backend_t backend
) {
    struct ggml_tensor* srcs[] = { x, gamma, beta };
    int32_t params[1];
    std::memcpy(&params[0], &eps, sizeof(float));
    // 2. Check if backend registers virtual node support
    if (ggml_ops_backend_supports_op(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_INSTANCE_NORM,
                                     srcs, 3, params, sizeof(params))) {

        int64_t ne[GGML_MAX_DIMS] = { x->ne[0], x->ne[1], x->ne[2], x->ne[3] };
        struct ggml_tensor* result = ggml_ops_ext::ops_new_virtual_node(
            ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_INSTANCE_NORM, x->type, ggml_n_dims(x), ne, 3, srcs);
        
        std::memcpy(result->op_params, &eps, sizeof(float));
        return result;
    }

    // 3. Fallback: execute standard GGML InstanceNorm subgraph
    struct ggml_tensor* x_f32 = x->type == GGML_TYPE_F32 ? x : ggml_cast(ctx, x, GGML_TYPE_F32);
    int64_t C = x_f32->ne[1];
    struct ggml_tensor* normalized = ggml_norm(ctx, x_f32, eps);
    struct ggml_tensor* out = normalized;

    if (gamma) {
        struct ggml_tensor* gamma_reshaped = ggml_reshape_2d(ctx, gamma, 1, C);
        struct ggml_tensor* gamma_f32 = gamma_reshaped->type == GGML_TYPE_F32 ? gamma_reshaped : ggml_cast(ctx, gamma_reshaped, GGML_TYPE_F32);
        out = ggml_mul(ctx, out, gamma_f32);
    }
    if (beta) {
        struct ggml_tensor* beta_reshaped = ggml_reshape_2d(ctx, beta, 1, C);
        struct ggml_tensor* beta_f32 = beta_reshaped->type == GGML_TYPE_F32 ? beta_reshaped : ggml_cast(ctx, beta_reshaped, GGML_TYPE_F32);
        out = ggml_add(ctx, out, beta_f32);
    }

    if (out->type != x->type) {
        out = ggml_cast(ctx, out, x->type);
    }

    return out;
}
