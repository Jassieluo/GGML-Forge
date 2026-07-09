#pragma once

#include "models/gguf_model.h"
#include "nn/nn.h"

namespace gpt_sovits {

// T2S Autoregressive GPT Graph Builder
struct T2SModel : public GGUFModel, public nn::Module {
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

    // Submodules as class members
    nn::Embedding word_embeddings;
    nn::Embedding audio_embeddings;
    nn::Linear bert_proj;
    nn::Linear predict;

    struct TransformerBlock : public nn::Module {
        nn::KVHeadAttention self_attn;
        nn::LayerNorm ln1;
        nn::LayerNorm ln2;
        nn::FeedForward ffn;

        TransformerBlock() {
            register_module("self_attn", &self_attn);
            register_module("ln1", &ln1);
            register_module("ln2", &ln2);
            register_module("ffn", &ffn);
        }
    };

    std::vector<TransformerBlock> layers;
    std::unordered_map<std::string, std::string> default_name_map;

    T2SModel();

    ~T2SModel() override {
        if (kv_buffer) ggml_backend_buffer_free(kv_buffer);
        if (kv_ctx) ggml_free(kv_ctx);
        if (custom_buffer) ggml_backend_buffer_free(custom_buffer);
        if (custom_ctx) ggml_free(custom_ctx);
    }

    void on_read_metadata(struct gguf_context* ctx_gguf) override;
    bool load(const std::string& path, ggml_backend_t backend);
    void init_default_name_map();
    
    // Autoregressive token-by-token decoding with KV Cache
    std::vector<int32_t> forward(
        struct ggml_context* ctx_graph, 
        const std::vector<int32_t>& prompt_phones,
        const std::vector<int32_t>& target_phones,
        const std::vector<int32_t>& prompt_semantics,
        struct ggml_tensor* bert_features,
        const std::vector<int32_t>& target_word2ph,
        int max_len,
        ggml_backend_t backend,
        ggml_gallocr_t galloc = nullptr
    );
};

} // namespace gpt_sovits
