#include "ops/ops.h"
#include "ggml-impl.h"
#include <cstring>

struct ggml_tensor* ggml_ops_sample_dist(
    struct ggml_context* ctx,
    struct ggml_tensor* logits,
    struct ggml_tensor* uniform,
    int32_t top_k,
    float top_p,
    float temperature,
    ggml_backend_t backend
) {
    if (!logits || !uniform) return nullptr;
    ggml_ops_ext::ops_sample_dist_params params;
    params.top_k = top_k;
    params.top_p = top_p;
    params.temperature = temperature;
    struct ggml_tensor* srcs[] = { logits, uniform };

    if (!ggml_ops_backend_supports_op(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_SAMPLE_DIST,
                                      srcs, 2, &params, sizeof(params))) {
        // Kernel-required: sampling has no sensible native-node composition.
        return nullptr;
    }

    const int64_t ne[4] = { 1, 1, 1, 1 };
    struct ggml_tensor* result = ggml_ops_ext::ops_new_virtual_node(
        ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_SAMPLE_DIST, GGML_TYPE_I32, 1, ne, 2, srcs);
    ggml_set_op_params(result, &params, sizeof(params));
    return result;
}
