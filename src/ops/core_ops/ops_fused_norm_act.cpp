#include "ops/ops.h"
#include "ggml-impl.h"
#include <cstring>

struct ggml_tensor* ggml_ops_fused_norm_act(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* gamma,
    struct ggml_tensor* beta,
    struct ggml_tensor* residual,
    float eps,
    ggml_ops_gate_activation activation,
    ggml_backend_t backend
) {
    if (!x || !gamma || !beta) return nullptr;
    ggml_ops_ext::ops_fused_norm_act_params params;
    params.eps = eps;
    params.activation = static_cast<int32_t>(activation);
    struct ggml_tensor* srcs[] = { x, gamma, beta, residual };
    const int n_srcs = residual ? 4 : 3;

    if (ggml_ops_backend_supports_op(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_FUSED_NORM_ACT,
                                     srcs, n_srcs, &params, sizeof(params))) {
        struct ggml_tensor* result = ggml_ops_ext::ops_new_virtual_node(
            ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_FUSED_NORM_ACT, x->type, ggml_n_dims(x), x->ne,
            n_srcs, srcs);
        ggml_set_op_params(result, &params, sizeof(params));
        return result;
    }

    // Fallback composed from native GGML nodes.
    struct ggml_tensor* input = residual ? ggml_add(ctx, x, residual) : x;
    struct ggml_tensor* normed = ggml_ops_layer_norm(ctx, input, gamma, beta, eps, backend);
    if (!normed) return nullptr;
    switch (activation) {
        case ggml_ops_gate_activation::silu: return ggml_silu(ctx, normed);
        case ggml_ops_gate_activation::gelu: return ggml_gelu(ctx, normed);
        case ggml_ops_gate_activation::relu: return ggml_relu(ctx, normed);
        case ggml_ops_gate_activation::identity: return normed;
    }
    return nullptr;
}
