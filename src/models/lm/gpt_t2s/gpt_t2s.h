#pragma once

#include "models/debug.h"
#include "models/model_profile.h"
#include "nn/nn.h"
#include <random>
#include <string>

struct gguf_context;

namespace gpt_sovits {

// T2S Autoregressive GPT Graph Builder
struct T2SModel : public nn::Module {
    // Reused autoregressive inputs, owned and created by the NN context.
    std::unique_ptr<nn::Context> input_context;
    struct ggml_tensor* text_ids_input = nullptr;
    struct ggml_tensor* audio_ids_input = nullptr;
    struct ggml_tensor* token_input = nullptr;
    struct ggml_tensor* bert_features_input = nullptr;
    
    int n_layers = 0;
    int n_heads = 0;
    int head_dim = 0;
    int family = 0;
    int metadata_hidden_dim = 0;
    int version = 0;
    std::string version_string;

    // Submodules as class members
    nn::Embedding word_embeddings;
    nn::Embedding audio_embeddings;
    nn::Linear bert_proj;
    nn::Linear predict;
    nn::Parameter text_position_alpha = nn::Parameter::optional();
    nn::Parameter audio_position_alpha = nn::Parameter::optional();

    nn::TransformerDecoder decoder;
    std::unordered_map<std::string, std::string> default_name_map;

    float text_alpha = 1.0f;
    float audio_alpha = 1.0f;
    std::vector<float> text_position_cache;
    std::vector<float> audio_position_cache;
    nn::AttentionCacheConfig attention_cache_config;

    ggml_backend_buffer_t input_buffer = nullptr;

    T2SModel();

    ~T2SModel() override {
        if (input_buffer) ggml_backend_buffer_free(input_buffer);
    }

    bool read_metadata(const struct gguf_context* ctx_gguf);
    bool load(const std::string& path, ggml_backend_t backend);
    void init_default_name_map(int layer_count = 24);
    
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
        std::mt19937& rng,
        ggml_gallocr_t galloc = nullptr
    );

    struct ggml_tensor* build_decoding_step(
        nn::Context& step_context,
        nn::AttentionCache& attention_cache,
        struct ggml_cgraph* cgraph,
        int total_decoded,
        int text_len,
        int audio_len,
        const std::vector<int32_t>& text_ids_vec,
        const std::vector<int32_t>& current_audio_ids,
        std::vector<float>& temp_bert,
        std::vector<float>& text_pe_data,
        std::vector<float>& audio_pe_data,
        std::vector<float>& mask_data,
        int32_t& last_token,
        struct ggml_tensor* bert_features,
        ggml_backend_t backend
    );

    int32_t sample_next_token(
        std::vector<float>& host_logits,
        const std::vector<int32_t>& current_audio_ids,
        int total_decoded,
        std::mt19937& rng
    );
};

} // namespace gpt_sovits
