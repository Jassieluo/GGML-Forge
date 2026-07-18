#include "categories/llm/llm.h"

#include <iostream>

namespace {

bool print_piece(const char* text, size_t length, void*) {
    std::cout.write(text, static_cast<std::streamsize>(length));
    std::cout.flush();
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf> [prompt]\n";
        return 2;
    }

    llm_runtime_params runtime_params = llm_runtime_default_params();
    runtime_params.n_ctx = 4096;
    runtime_params.n_gpu_layers = 99;
    llm_runtime_ptr runtime = llm_runtime_create(runtime_params);
    llm_model_ptr model = runtime ? llm_load_model(runtime, argv[1]) : nullptr;
    llm_session_ptr session = model ? llm_create_session(model) : nullptr;
    if (!session) {
        std::cerr << "Failed to initialize the LLM.\n";
        llm_free_session(session);
        llm_free_model(model);
        llm_runtime_free(runtime);
        return 1;
    }

    std::cerr << "Provider: " << llm_model_get_provider(model) << '\n';
    llm_generation_params generation = llm_generation_default_params();
    generation.max_tokens = 128;
    const char* prompt = argc > 2 ? argv[2] : "Write a short greeting from GGML-Forge:";
    const bool ok = llm_generate(session, prompt, generation, print_piece, nullptr);
    std::cout << '\n';

    llm_free_session(session);
    llm_free_model(model);
    llm_runtime_free(runtime);
    return ok ? 0 : 1;
}
