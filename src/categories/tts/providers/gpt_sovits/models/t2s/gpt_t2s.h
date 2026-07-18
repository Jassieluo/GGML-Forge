#pragma once

#include "providers/gpt_sovits/models/debug.h"
#include "providers/gpt_sovits/models/model_profile.h"
#include "nn/nn.h"
#include <random>
#include <string>

struct gguf_context;

namespace gpt_sovits {

class T2SInputs {
public:
    explicit T2SInputs(ggml_backend_t backend);
    ~T2SInputs();
    T2SInputs(const T2SInputs&) = delete;
    T2SInputs& operator=(const T2SInputs&) = delete;

    nn::Context context{4 * 1024 * 1024};
    ggml_tensor* text_ids = nullptr;
    ggml_tensor* audio_ids = nullptr;
    ggml_tensor* token = nullptr;
    ggml_tensor* bert_features = nullptr;

private:
    ggml_backend_buffer_t buffer_ = nullptr;
};

// T2S Autoregressive GPT Graph Builder
struct T2SModel : public nn::Module<T2SModel> {
    int n_layers = 0;
    int n_heads = 0;
    int head_dim = 0;
    int family = 0;
    int metadata_hidden_dim = 0;
    int version = 0;
    std::string version_string;

    // Submodules as class members
    nn::Embedding& word_embeddings = submodule<nn::Embedding>("word_embeddings");
    nn::Embedding& audio_embeddings = submodule<nn::Embedding>("audio_embeddings");
    nn::Linear& bert_proj = submodule<nn::Linear>("bert_proj");
    nn::Linear& predict = submodule<nn::Linear>("predict");
    nn::Parameter& text_position_alpha = parameter(
        "text_position_alpha", nn::Parameter::optional(std::nullopt, nn::Parameter::Usage::scalar));
    nn::Parameter& audio_position_alpha = parameter(
        "audio_position_alpha", nn::Parameter::optional(std::nullopt, nn::Parameter::Usage::scalar));

    nn::TransformerDecoder& decoder = submodule<nn::TransformerDecoder>(
        "decoder", 24, 16, 32, nn::ActivationType::RELU, 1e-5f);

    float text_alpha = 1.0f;
    float audio_alpha = 1.0f;
    std::vector<float> text_position_cache;
    std::vector<float> audio_position_cache;
    nn::KVCacheConfig attention_cache_config;

    T2SModel();

    bool read_metadata(const struct gguf_context* ctx_gguf);
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
        ggml_backend_t backend,
        std::mt19937& rng,
        ggml_gallocr_t galloc = nullptr
    );

    struct ggml_tensor* prefill(
        nn::Context& step_context,
        T2SInputs& inputs,
        nn::KVCache& attention_cache,
        struct ggml_cgraph* cgraph,
        int text_len,
        int audio_len,
        const std::vector<int32_t>& text_ids_vec,
        const std::vector<int32_t>& current_audio_ids,
        std::vector<float>& temp_bert,
        std::vector<float>& text_pe_data,
        std::vector<float>& audio_pe_data,
        std::vector<float>& mask_data,
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
