#pragma once

#include "models/debug.h"
#include "models/model_profile.h"
#include "nn/nn.h"

struct gguf_context;

namespace gpt_sovits {

// RoBERTa BERT Graph Builder
struct BertModel : public nn::Module {
    nn::TransformerEncoder encoder;
    nn::Embedding word_embeddings;
    nn::Embedding position_embeddings;
    nn::Embedding token_type_embeddings;
    nn::LayerNorm embeddings_ln;
    int n_heads = 0;

    BertModel() : encoder(22, 16, 64, nn::ActivationType::GELU, 1e-12f, false),
                  word_embeddings(nullptr),
                  position_embeddings(nullptr),
                  token_type_embeddings(nullptr),
                  embeddings_ln(nullptr, nullptr, 1e-12f) {
        register_module("encoder", &encoder);
        register_module("word_embeddings", &word_embeddings);
        register_module("position_embeddings", &position_embeddings);
        register_module("token_type_embeddings", &token_type_embeddings);
        register_module("embeddings_ln", &embeddings_ln);
    }

    bool load(const std::string& path, ggml_backend_t backend);
    bool read_metadata(const struct gguf_context* ctx_gguf);
    struct ggml_tensor* forward(struct ggml_context* ctx_graph, const std::vector<int32_t>& input_ids, ggml_backend_t backend, ggml_gallocr_t galloc = nullptr);
};

} // namespace gpt_sovits
