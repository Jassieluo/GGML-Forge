#include "ops/ops.h"
#include "ggml-impl.h"

struct ggml_tensor* ggml_ops_soft_max(
    struct ggml_context* ctx,
    struct ggml_tensor* a,
    struct ggml_tensor* mask,
    float scale,
    float max_bias,
    ggml_backend_t backend
) {
    // 1. Check if the backend registers a custom builder
    ggml_ops_ext::ops_op_builder_t builder = ggml_ops_ext::find_ops_builder(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_SOFTMAX);
    if (builder) {
        struct ggml_tensor* srcs[] = { a, mask, nullptr };
        int n_srcs = mask ? 2 : 1;
        union {
            float f;
            int32_t i;
        } u_scale, u_bias;
        u_scale.f = scale;
        u_bias.f = max_bias;
        int32_t params[] = { u_scale.i, u_bias.i };
        return builder(ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_SOFTMAX, srcs, n_srcs, params, 2, backend);
    }

    // 2. Otherwise check if it supports direct handler execution (virtual node)
    if (ggml_ops_backend_supports_op(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_SOFTMAX)) {
        struct ggml_tensor* srcs[] = { a, mask, nullptr };
        int n_srcs = mask ? 2 : 1;
        struct ggml_tensor* result = ggml_ops_ext::ops_new_virtual_node(ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_SOFTMAX, GGML_TYPE_F32, ggml_n_dims(a), a->ne, n_srcs, srcs);
        
        float params[] = { scale, max_bias };
        ggml_set_op_params(result, params, sizeof(params));
        return result;
    }

    // 3. Default fallback
    if (mask) {
        return ggml_soft_max_ext(ctx, a, mask, scale, max_bias);
    } else {
        return ggml_soft_max(ctx, a);
    }
}
