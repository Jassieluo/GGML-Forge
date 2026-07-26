#pragma once

#include "nn/core/context.h"

#include <memory>
#include <utility>

namespace nn {

struct KVCacheConfig {
    ggml_type key_type = GGML_TYPE_F16;
    ggml_type value_type = GGML_TYPE_F16;
};

class KVCache {
public:
    int head_dim = 0;
    int max_len = 512;
    int n_heads = 0;
    int n_layers = 0;
    KVCacheConfig config;

    KVCache() = default;
    KVCache(const KVCache&) = delete;
    KVCache& operator=(const KVCache&) = delete;

    KVCache(KVCache&& other) noexcept
        : head_dim(other.head_dim), max_len(other.max_len), n_heads(other.n_heads),
          n_layers(other.n_layers), config(other.config), k(other.k), v(other.v),
          context_(std::move(other.context_)), buffer_(std::exchange(other.buffer_, nullptr)) {
        other.k = nullptr;
        other.v = nullptr;
    }

    KVCache& operator=(KVCache&& other) noexcept {
        if (this != &other) {
            release();
            head_dim = other.head_dim;
            max_len = other.max_len;
            n_heads = other.n_heads;
            n_layers = other.n_layers;
            config = other.config;
            k = other.k;
            v = other.v;
            context_ = std::move(other.context_);
            buffer_ = std::exchange(other.buffer_, nullptr);
            other.k = nullptr;
            other.v = nullptr;
        }
        return *this;
    }

    ~KVCache() { release(); }

    bool allocate(
        ggml_backend_t backend,
        int dimension,
        int capacity,
        int heads,
        int layers,
        KVCacheConfig cache_config = {}) {
        release();
        head_dim = dimension;
        max_len = capacity;
        n_heads = heads;
        n_layers = layers;
        config = cache_config;

        const auto valid_type = [dimension](ggml_type type) {
            return (type == GGML_TYPE_F32 || type == GGML_TYPE_F16 ||
                    type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0) &&
                   dimension % ggml_blck_size(type) == 0;
        };
        if (!valid_type(config.key_type) || !valid_type(config.value_type)) return false;

        try {
            context_ = std::make_unique<Context>(64 * 1024);
            k = context_->empty("kv_cache.key", {dimension, capacity, heads, layers}, config.key_type);
            v = context_->empty("kv_cache.value", {dimension, capacity, heads, layers}, config.value_type);
        } catch (const std::exception&) {
            release();
            return false;
        }

        buffer_ = ggml_backend_alloc_ctx_tensors(context_->native_handle(), backend);
        if (!buffer_) {
            release();
            return false;
        }
        return true;
    }

    ggml_tensor* decode_attention(
        Context& context, int layer, ggml_tensor* query, ggml_tensor* new_key,
        ggml_tensor* new_value, ggml_tensor* position, ggml_tensor* valid_length,
        float scale, ggml_backend_t backend, ggml_tensor* mask = nullptr);

    ggml_tensor* prefill_attention(
        Context& context, int layer, ggml_tensor* query, ggml_tensor* new_key,
        ggml_tensor* new_value, float scale, ggml_backend_t backend,
        ggml_tensor* mask = nullptr);

private:
    ggml_tensor* k = nullptr;
    ggml_tensor* v = nullptr;
    std::unique_ptr<Context> context_;
    ggml_backend_buffer_t buffer_ = nullptr;

    void release() noexcept {
        if (buffer_) ggml_backend_buffer_free(buffer_);
        buffer_ = nullptr;
        context_.reset();
        k = nullptr;
        v = nullptr;
    }
};

} // namespace nn
