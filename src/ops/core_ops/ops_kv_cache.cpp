#include "ops/ops.h"

struct ggml_tensor* ggml_ops_kv_cache_update(
    struct ggml_context* ctx,
    struct ggml_tensor* cache_k,
    struct ggml_tensor* cache_v,
    struct ggml_tensor* new_k,
    struct ggml_tensor* new_v,
    struct ggml_tensor* position,
    ggml_backend_t backend
) {
    struct ggml_tensor* srcs[] = { cache_k, cache_v, new_k, new_v, position };
    if (!ggml_ops_backend_supports_op(
            backend, ggml_ops_ext::GGML_OP_OPS_VIRT_KV_CACHE_UPDATE,
            srcs, 5, nullptr, 0)) return nullptr;
    const int64_t ne[] = { 1 };
    return ggml_ops_ext::ops_new_virtual_node(
        ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_KV_CACHE_UPDATE,
        GGML_TYPE_I32, 1, ne, 5, srcs);
}
