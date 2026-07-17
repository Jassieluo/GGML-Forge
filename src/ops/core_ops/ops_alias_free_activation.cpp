#include "ops/ops.h"

struct ggml_tensor* ggml_ops_alias_free_activation(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* up_filter,
    struct ggml_tensor* down_filter,
    struct ggml_tensor* alpha,
    struct ggml_tensor* beta,
    ggml_backend_t backend
) {
    struct ggml_tensor* srcs[] = { x, up_filter, down_filter, alpha, beta };
    if (!ggml_ops_backend_supports_op(
            backend, ggml_ops_ext::GGML_OP_OPS_VIRT_ALIAS_FREE_ACTIVATION, srcs, 5)) {
        return nullptr;
    }
    return ggml_ops_ext::ops_new_virtual_node(
        ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_ALIAS_FREE_ACTIVATION,
        x->type, ggml_n_dims(x), x->ne, 5, srcs);
}
