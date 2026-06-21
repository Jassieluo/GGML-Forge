#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"  // GGML's native GGUF loader header

#ifndef GPT_SOVITS_DEBUG_ENABLED
#define GPT_SOVITS_DEBUG_ENABLED() (std::getenv("GPT_SOVITS_DEBUG") != nullptr)
#endif

namespace gpt_sovits {

struct TensorShape {
    int64_t ne[4];
};

// Base Model Weights Struct
struct GGUFModel {
    struct ggml_context* ctx = nullptr;
    ggml_backend_buffer_t backend_buffer = nullptr;
    std::vector<uint8_t> weight_data; // holds the tensor binary data if loaded in memory
    std::unordered_map<std::string, struct ggml_tensor*> tensors;
    
    int n_heads = 8;
    int head_dim = 64;
    
    virtual ~GGUFModel() {
        if (ctx) {
            ggml_free(ctx);
        }
        if (backend_buffer) {
            ggml_backend_buffer_free(backend_buffer);
        }
    }

    // Lifecycle hooks for model-specific GGUF loading customization
    virtual void on_read_metadata(struct gguf_context* ctx_gguf) {}
    virtual void on_prepare_tensor(struct ggml_tensor* tensor, const std::string& name) {}
    virtual bool on_upload_tensor(
        struct ggml_tensor* t_backend,
        const void* raw_data,
        size_t size,
        enum ggml_type type,
        const std::string& name
    ) {
        return false; // Default: standard copy by generic loader
    }
    
    struct ggml_tensor* get_tensor(const std::string& name) const {
        auto it = tensors.find(name);
        if (it != tensors.end()) {
            return it->second;
        }
        return nullptr;
    }
};

// Helper to load GGUF files and extract weights into ggml tensors
bool load_gguf_model(const std::string& path, GGUFModel& model, ggml_backend_t backend);

} // namespace gpt_sovits
