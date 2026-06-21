#pragma once

#include "include/tts.h"
#include "src/models/model.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

// Internal implementation of tts_model
struct tts_model {
    std::string arch_name; // Loaded from GGUF e.g. "vits", "gpt-t2s"
    
    // GGUF loading and registry
    struct ggml_context* ctx = nullptr;
    ggml_backend_buffer_t backend_buffer = nullptr;
    std::vector<uint8_t> weight_data; // raw binary file contents
    std::unordered_map<std::string, struct ggml_tensor*> tensors;
    
    // Vocabulary and tokens mappings
    std::unordered_map<std::string, int32_t> token_to_id;
    std::vector<std::string> id_to_token;

    // Decoupled architecture implementation
    std::unique_ptr<tts::ModelArch> arch;

    ~tts_model() {
        if (ctx) {
            ggml_free(ctx);
        }
        if (backend_buffer) {
            ggml_backend_buffer_free(backend_buffer);
        }
    }
};

// Internal implementation of tts_context
struct tts_context {
    const tts_model* model = nullptr;
    
    // Computation state and thread configurations
    uint32_t n_threads = 4;
    
    // GGML Backend and memory allocators
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    
    // Audio synthesis buffer
    std::vector<float> audio_output;

    ~tts_context() {
        if (gallocr) {
            ggml_gallocr_free(gallocr);
        }
        if (backend) {
            ggml_backend_free(backend);
        }
    }
};
