#include "nn/functional/attention.h"

#include "ops/ops.h"

namespace nn::functional {

ggml_tensor* attention(
    ggml_context* ctx, ggml_tensor* query, ggml_tensor* key, ggml_tensor* value,
    ggml_tensor* mask, ggml_tensor* weights, float scale, int sliding_window,
    ggml_backend_t backend
) {
    return ggml_ops_attention(
        ctx, query, key, value, mask, weights, scale, sliding_window, backend);
}

ggml_tensor* relative_position_keys(
    ggml_context* ctx, ggml_tensor* query, ggml_tensor* embedding,
    float scale, int window, ggml_backend_t backend
) {
    return ggml_ops_relative_pe_keys(ctx, query, embedding, scale, window, backend);
}

ggml_tensor* relative_position_values(
    ggml_context* ctx, ggml_tensor* weights, ggml_tensor* embedding,
    ggml_tensor* attention_output, int window, ggml_backend_t backend
) {
    return ggml_ops_relative_pe_values(
        ctx, weights, embedding, attention_output, window, backend);
}

} // namespace nn::functional
