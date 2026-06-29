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
    // 1. Check if backend registers a custom builder
    ggml_ops_ext::ops_op_builder_t builder = ggml_ops_ext::find_ops_builder(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_INSTANCE_NORM);
    if (builder) {
        struct ggml_tensor* srcs[] = { x, gamma, beta };
        int32_t params[1];
        std::memcpy(&params[0], &eps, sizeof(float));
        return builder(ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_INSTANCE_NORM, srcs, 3, params, 1, backend);
    }

    // 2. Check if backend registers virtual node support
    if (ggml_ops_backend_supports_op(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_INSTANCE_NORM)) {
        struct ggml_tensor* srcs[] = { x, gamma, beta };
        int32_t params[1];
        std::memcpy(&params[0], &eps, sizeof(float));

        int64_t ne[GGML_MAX_DIMS] = { x->ne[0], x->ne[1], x->ne[2], x->ne[3] };
        struct ggml_tensor* result = ggml_ops_ext::ops_new_virtual_node(
            ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_INSTANCE_NORM, x->type, ggml_n_dims(x), ne, 3, srcs);
        
        std::memcpy(result->op_params, &eps, sizeof(float));
        return result;
    }

    // 3. Fallback: execute standard GGML InstanceNorm subgraph
    int64_t C = x->ne[1];
    struct ggml_tensor* normalized = ggml_norm(ctx, x, eps);
    struct ggml_tensor* out = normalized;

    if (gamma) {
        struct ggml_tensor* gamma_reshaped = ggml_reshape_2d(ctx, gamma, 1, C);
        out = ggml_mul(ctx, out, gamma_reshaped);
    }
    if (beta) {
        struct ggml_tensor* beta_reshaped = ggml_reshape_2d(ctx, beta, 1, C);
        out = ggml_add(ctx, out, beta_reshaped);
    }

    return out;
}
