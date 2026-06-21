#include "include/tts.h"
#include "src/tts_internal.h"
#include "src/tts_frontend.h"
#include "src/models/model.h"

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cstring>

static int g_log_level = 0; // Default to info

void tts_set_log_level(int level) {
    g_log_level = level;
}

tts_model_ptr tts_load_model_from_file(const char* path, struct tts_model_params params) {
    if (!path) return nullptr;

    std::cout << "[TTS Core] Loading model from: " << path << std::endl;

    auto model = std::make_unique<tts_model>();

    // Placeholder: In a real implementation, we use GGUF library to read metadata and tensors.
    // E.g.:
    // struct gguf_init_params gguf_params = { /*...*/ };
    // struct gguf_context * gguf_ctx = gguf_init_from_file(path, gguf_params);
    // model->arch_name = gguf_get_val_str(gguf_ctx, "general.architecture");
    
    // For now, let's pretend we parsed GGUF and got "vits" as architecture
    model->arch_name = "vits";

    // 1. Resolve model architecture using factory
    model->arch = tts::create_model_arch(model->arch_name);
    if (!model->arch) {
        std::cerr << "[TTS Core] Error: Unsupported model architecture: " << model->arch_name << std::endl;
        return nullptr;
    }

    // 2. Load vocabulary / token mapping (represented as GGUF KV metadata)
    // E.g. token_to_id and id_to_token mapping
    
    return model.release();
}

void tts_free_model(tts_model_ptr model) {
    if (model) {
        delete model;
    }
}

tts_context_ptr tts_new_context_with_model(tts_model_ptr model, struct tts_context_params params) {
    if (!model) return nullptr;

    auto ctx = std::make_unique<tts_context>();
    ctx->model = model;
    ctx->n_threads = params.n_threads;

    std::cout << "[TTS Core] Created new inference context with " << ctx->n_threads << " threads." << std::endl;

    // 1. Initialize backend (e.g. CPU or CUDA)
    // E.g.:
    // ctx->backend = ggml_backend_cpu_init();
    
    return ctx.release();
}

void tts_free_context(tts_context_ptr ctx) {
    if (ctx) {
        delete ctx;
    }
}

const float* tts_synthesize(
    tts_context_ptr ctx,
    const char*     text,
    const char*     lang,
    float           speed,
    int32_t*        out_samples_count
) {
    if (!ctx || !text || !out_samples_count) {
        if (out_samples_count) *out_samples_count = 0;
        return nullptr;
    }

    std::cout << "[TTS Core] Synthesizing text: \"" << text << "\" (" << lang << ") at speed " << speed << std::endl;

    // 1. Run Text Processing / Frontend to convert text to IDs
    // E.g.:
    // std::vector<int32_t> input_ids;
    // tts::Frontend frontend("dict_path");
    // frontend.text_to_ids(text, lang, ctx->model->token_to_id, input_ids);

    // 2. Build the GGML computational graph for execution
    // struct ggml_context* graph_ctx = ggml_init(...);
    // std::unordered_map<std::string, struct ggml_tensor*> graph_inputs;
    // ... create input tensors ...
    // struct ggml_cgraph* gf = ctx->model->arch->build_graph(graph_ctx, graph_inputs);

    // 3. Allocate tensor memories and execute on backend
    // ggml_backend_graph_compute(ctx->backend, gf);

    // 4. Read synthesized waveform back from graph output tensor
    // struct ggml_tensor* output_tensor = ...
    
    // Mock audio data: 1 second of 22050Hz sine wave for testing
    ctx->audio_output.clear();
    ctx->audio_output.resize(22050);
    for (size_t i = 0; i < ctx->audio_output.size(); ++i) {
        ctx->audio_output[i] = 0.5f * std::sin(2.0f * 3.14159f * 440.0f * i / 22050.0f); // 440Hz Sine
    }

    *out_samples_count = static_cast<int32_t>(ctx->audio_output.size());
    return ctx->audio_output.data();
}
