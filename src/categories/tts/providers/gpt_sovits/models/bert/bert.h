#pragma once

#include "providers/gpt_sovits/models/debug.h"
#include "providers/gpt_sovits/models/model_profile.h"
#include "nn/nn.h"

struct gguf_context;

namespace gpt_sovits {

// RoBERTa BERT Graph Builder
struct BertModel : public nn::Module<BertModel> {
    nn::TransformerEncoder& encoder = submodule<nn::TransformerEncoder>(
        "encoder", 22, 16, 64, nn::ActivationType::GELU, 1e-12f, false);
    nn::Embedding& word_embeddings = submodule<nn::Embedding>("word_embeddings");
    nn::Embedding& position_embeddings = submodule<nn::Embedding>("position_embeddings");
    nn::Embedding& token_type_embeddings = submodule<nn::Embedding>("token_type_embeddings");
    nn::LayerNorm& embeddings_ln = submodule<nn::LayerNorm>("embeddings_ln");
    int n_heads = 0;

    BertModel() { embeddings_ln.eps = 1e-12f; }

    bool load(const std::string& path, ggml_backend_t backend);
    bool read_metadata(const struct gguf_context* ctx_gguf);
    struct ggml_tensor* forward(
        nn::Context& context,
        struct ggml_tensor* input_ids,
        struct ggml_tensor* position_ids,
        struct ggml_tensor* token_type_ids,
        ggml_backend_t backend);
};

class BertRunner {
public:
    BertRunner(BertModel& model, ggml_backend_t backend, ggml_gallocr_t allocator = nullptr)
        : model_(model), backend_(backend), allocator_(allocator) {}
    ggml_tensor* forward(ggml_context* output_context, const std::vector<int32_t>& input_ids);

private:
    BertModel& model_;
    ggml_backend_t backend_ = nullptr;
    ggml_gallocr_t allocator_ = nullptr;
};

} // namespace gpt_sovits
