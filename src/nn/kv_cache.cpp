#include "nn/nn.h"

#include <stdexcept>

namespace nn {

ggml_tensor* KVCache::prefill_attention(
    Context& context,
    int layer,
    ggml_tensor* q,
    ggml_tensor* new_k,
    ggml_tensor* new_v,
    float scale,
    ggml_backend_t backend,
    ggml_tensor* mask,
    ggml_cgraph* graph
) {
    if (!k || !v || !backend) throw std::logic_error("KVCache must be allocated before prefill");
    if (layer < 0 || layer >= n_layers) throw std::out_of_range("KVCache layer is out of range");
    const int64_t length = new_k->ne[1];
    if (length <= 0 || length > max_len || new_v->ne[1] != length) {
        throw std::invalid_argument("KVCache prefill length is invalid");
    }
    ggml_context* native = context.native_handle();
    ggml_tensor* layer_k = ggml_view_3d(
        native, k, head_dim, length, n_heads, k->nb[1], k->nb[2],
        static_cast<size_t>(layer) * k->nb[3]);
    ggml_tensor* layer_v = ggml_view_3d(
        native, v, head_dim, length, n_heads, v->nb[1], v->nb[2],
        static_cast<size_t>(layer) * v->nb[3]);
    ggml_tensor* copy_k = ggml_cpy(native, new_k, layer_k);
    ggml_tensor* copy_v = ggml_cpy(native, new_v, layer_v);
    if (graph) {
        ggml_build_forward_expand(graph, copy_k);
        ggml_build_forward_expand(graph, copy_v);
    }
    ggml_tensor* output = ggml_ops_attention(
        native, q, layer_k, layer_v, mask, nullptr, scale, -1, backend);
    if (!output) throw std::runtime_error("backend does not support KV cache prefill attention");
    return output;
}

ggml_tensor* KVCache::decode_attention(
    Context& context,
    int layer,
    ggml_tensor* q,
    ggml_tensor* new_k,
    ggml_tensor* new_v,
    ggml_tensor* position,
    ggml_tensor* valid_length,
    float scale,
    ggml_backend_t backend,
    ggml_tensor* mask
) {
    if (!k || !v || !backend) throw std::logic_error("KVCache must be allocated before decode");
    if (layer < 0 || layer >= n_layers) throw std::out_of_range("KVCache layer is out of range");
    if (!q || !new_k || !new_v || !position || !valid_length) {
        throw std::invalid_argument("KVCache decode tensors cannot be null");
    }

    ggml_context* native = context.native_handle();
    const size_t k_offset = static_cast<size_t>(layer) * k->nb[3];
    const size_t v_offset = static_cast<size_t>(layer) * v->nb[3];
    ggml_tensor* layer_k = ggml_view_3d(
        native, k, head_dim, max_len, n_heads, k->nb[1], k->nb[2], k_offset);
    ggml_tensor* layer_v = ggml_view_3d(
        native, v, head_dim, max_len, n_heads, v->nb[1], v->nb[2], v_offset);
    ggml_tensor* update = ggml_ops_kv_cache_update(
        native, layer_k, layer_v, new_k, new_v, position, backend);
    if (!update) throw std::runtime_error("backend does not support KV cache update");
    ggml_tensor* output = ggml_ops_attention(
        native, q, layer_k, layer_v, mask, nullptr, scale, -1, backend,
        valid_length, update);
    if (!output) throw std::runtime_error("backend does not support runtime-length KV attention");
    return output;
}

} // namespace nn
