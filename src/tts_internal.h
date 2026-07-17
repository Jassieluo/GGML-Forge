#pragma once

#include "include/tts.h"
#include "src/pipelines/tts_pipeline.h"
#include "src/runtime.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>

// Forward declarations
namespace tts {
    class ITTSPipeline;
}

struct tts_runtime {
    tts::RuntimeContext context;
    std::mutex policy_mutex;
};

// Internal implementation of tts_model
struct tts_model {
    std::string arch_name; // Loaded from GGUF e.g. "gpt-sovits"
    tts::ModelConfig config;
    std::shared_ptr<const tts::RuntimeContext> runtime;
    
    // GGUF loading and registry
    struct ggml_context* ctx = nullptr;
    ggml_backend_buffer_t backend_buffer = nullptr;
    std::vector<uint8_t> weight_data; // raw binary file contents
    std::unordered_map<std::string, struct ggml_tensor*> tensors;
    
    // Vocabulary and tokens mappings
    std::unordered_map<std::string, int32_t> token_to_id;
    std::vector<std::string> id_to_token;

    // Decoupled pipeline implementation
    std::shared_ptr<tts::ITTSPipeline> pipeline;

    ~tts_model() {
        pipeline.reset();
        if (backend_buffer) {
            ggml_backend_buffer_free(backend_buffer);
        }
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

// Internal implementation of tts_session
struct tts_session {
    std::shared_ptr<tts::ITTSPipeline> pipeline;
    std::unique_ptr<tts::ITTSSession> session;
    
    // Audio synthesis buffer
    std::vector<float> audio_output;
};
