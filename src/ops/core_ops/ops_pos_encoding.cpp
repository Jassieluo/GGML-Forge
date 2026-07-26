#include "ops/ops.h"
#include "ggml-impl.h"
#include <cstring>

struct ggml_tensor* ggml_ops_pos_encoding(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* position,
    float base,
    int32_t offset,
    ggml_backend_t backend
) {
    if (!x) return nullptr;
    ggml_ops_ext::ops_pos_encoding_params params;
    params.base = base;
    params.offset = offset;
    struct ggml_tensor* srcs[] = { x, position };
    const int n_srcs = position ? 2 : 1;

    if (!ggml_ops_backend_supports_op(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_POS_ENCODING,
                                      srcs, n_srcs, &params, sizeof(params))) {
        // Kernel-required: composing the sinusoid table from native nodes would
        // need arange/pow chains that no current caller wants on a hot path.
        return nullptr;
    }

    struct ggml_tensor* result = ggml_ops_ext::ops_new_virtual_node(
        ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_POS_ENCODING, x->type, ggml_n_dims(x), x->ne,
        n_srcs, srcs);
    ggml_set_op_params(result, &params, sizeof(params));
    return result;
}
