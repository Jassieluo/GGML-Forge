#pragma once

#include "models/gguf_model.h"
#include "nn/nn.h"

namespace gpt_sovits {

// RoBERTa BERT Graph Builder
struct BertModel : public GGUFModel {
    struct ggml_context* custom_ctx = nullptr;
    ggml_backend_buffer_t custom_buffer = nullptr;

    // Pre-allocated static input placeholders
    nn::Buffer input_ids;
    nn::Buffer position_ids;
    nn::Buffer token_type_ids;

    ~BertModel() override {
        if (custom_ctx) {
            ggml_free(custom_ctx);
        }
        if (custom_buffer) {
            ggml_backend_buffer_free(custom_buffer);
        }
    }

    bool load(const std::string& path, ggml_backend_t backend);
    void on_read_metadata(struct gguf_context* ctx_gguf) override;
    struct ggml_tensor* forward(struct ggml_context* ctx_graph, const std::vector<int32_t>& input_ids, ggml_backend_t backend);
};

} // namespace gpt_sovits
