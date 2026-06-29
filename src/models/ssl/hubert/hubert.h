#pragma once

#include "models/gguf_model.h"
#include "nn/nn.h"

namespace gpt_sovits {

// CNHuBERT Graph Builder
struct HubertModel : public GGUFModel {
    struct ggml_tensor* pos_conv_w = nullptr;
    std::vector<uint8_t> pos_conv_w_data;
    struct ggml_context* custom_ctx = nullptr;
    ggml_backend_buffer_t custom_buffer = nullptr;
    ggml_backend_buffer_t pos_conv_w_buffer = nullptr;
    
    // Pre-allocated static input placeholder
    nn::Buffer input_audio;
    
    ~HubertModel() override {
        if (custom_ctx) {
            ggml_free(custom_ctx);
        }
        if (custom_buffer) {
            ggml_backend_buffer_free(custom_buffer);
        }
        if (pos_conv_w_buffer) {
            ggml_backend_buffer_free(pos_conv_w_buffer);
        }
    }

    void on_read_metadata(struct gguf_context* ctx_gguf) override;
    bool load(const std::string& path, ggml_backend_t backend);
    struct ggml_tensor* forward(struct ggml_context* ctx_graph, struct ggml_tensor* input_audio, ggml_backend_t backend);
};

} // namespace gpt_sovits
