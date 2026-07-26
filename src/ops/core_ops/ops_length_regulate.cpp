#include "ops/ops.h"
#include "ggml-impl.h"
#include <cstring>

struct ggml_tensor* ggml_ops_length_regulate(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* durations,
    int32_t total,
    ggml_backend_t backend
) {
    if (!x || !durations || total < 1) return nullptr;
    ggml_ops_ext::ops_length_regulate_params params;
    params.total = total;
    struct ggml_tensor* srcs[] = { x, durations };

    if (!ggml_ops_backend_supports_op(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_LENGTH_REGULATE,
                                      srcs, 2, &params, sizeof(params))) {
        // Kernel-required: variable-count repetition has no native composition
        // (ggml repeat only broadcasts uniformly).
        return nullptr;
    }

    const int64_t ne[2] = { x->ne[0], total };
    struct ggml_tensor* result = ggml_ops_ext::ops_new_virtual_node(
        ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_LENGTH_REGULATE, GGML_TYPE_F32, 2, ne, 2, srcs);
    ggml_set_op_params(result, &params, sizeof(params));
    return result;
}
