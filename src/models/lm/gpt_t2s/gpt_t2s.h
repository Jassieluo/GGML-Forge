#pragma once

#include "models/gguf_model.h"
#include "nn/nn.h"

namespace gpt_sovits {

// T2S Autoregressive GPT Graph Builder
struct T2SModel : public GGUFModel {
    struct ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buffer = nullptr;
    struct ggml_tensor* kv_k = nullptr;
    struct ggml_tensor* kv_v = nullptr;
    
    // Pre-allocated static input placeholders (zero allocation during inference loop)
    nn::Buffer text_ids;
    nn::Buffer audio_ids;
    nn::Buffer token;
    nn::Buffer bert_features;
    
    struct ggml_context* custom_ctx = nullptr;
    ggml_backend_buffer_t custom_buffer = nullptr;
    
    int n_layers = 24;

    ~T2SModel() override {
        if (kv_buffer) ggml_backend_buffer_free(kv_buffer);
        if (kv_ctx) ggml_free(kv_ctx);
        if (custom_buffer) ggml_backend_buffer_free(custom_buffer);
        if (custom_ctx) ggml_free(custom_ctx);
    }

    void on_read_metadata(struct gguf_context* ctx_gguf) override;
    bool load(const std::string& path, ggml_backend_t backend);
    
    // Autoregressive token-by-token decoding with KV Cache
    std::vector<int32_t> forward(
        struct ggml_context* ctx_graph, 
        const std::vector<int32_t>& prompt_phones,
        const std::vector<int32_t>& target_phones,
        const std::vector<int32_t>& prompt_semantics,
        struct ggml_tensor* bert_features,
        const std::vector<int32_t>& target_word2ph,
        int max_len,
        ggml_backend_t backend
    );
};

} // namespace gpt_sovits
