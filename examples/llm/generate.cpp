#include "categories/llm/llm.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

struct TimedOutput {
    std::chrono::steady_clock::time_point started;
    std::chrono::steady_clock::time_point first;
    bool received = false;
};

bool print_piece(const char* text, size_t length, void* data) {
    auto* timing = static_cast<TimedOutput*>(data);
    if (!timing->received) {
        timing->received = true;
        timing->first = std::chrono::steady_clock::now();
    }
    std::cout.write(text, static_cast<std::streamsize>(length));
    std::cout.flush();
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.gguf> [prompt] [device] [repeat]\n";
        return 2;
    }

    llm_runtime_params runtime_params = llm_runtime_default_params();
    runtime_params.n_ctx = 4096;
    runtime_params.n_gpu_layers = 99;
    if (argc > 3) runtime_params.device = argv[3];
    const auto load_started = std::chrono::steady_clock::now();
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
    const auto load_finished = std::chrono::steady_clock::now();
    std::cerr << "Load + context: "
              << std::chrono::duration<double>(load_finished - load_started).count()
              << " s\n";
    llm_generation_params generation = llm_generation_default_params();
    generation.max_tokens = 128;
    const char* prompt = argc > 2 ? argv[2] : "Write a short greeting from GGML-Forge:";
    const int repeat = argc > 4 ? std::max(1, std::atoi(argv[4])) : 1;
    bool ok = true;
    for (int iteration = 0; iteration < repeat; ++iteration) {
        TimedOutput timing;
        timing.started = std::chrono::steady_clock::now();
        ok = llm_generate(session, prompt, generation, print_piece, &timing) && ok;
        const auto finished = std::chrono::steady_clock::now();
        std::cout << '\n';
        std::cerr << "Run " << (iteration + 1) << ": TTFT="
                  << (timing.received
                          ? std::chrono::duration<double>(timing.first - timing.started).count()
                          : -1.0)
                  << " s, total="
                  << std::chrono::duration<double>(finished - timing.started).count()
                  << " s\n";
    }

    llm_free_session(session);
    llm_free_model(model);
    llm_runtime_free(runtime);
    return ok ? 0 : 1;
}
