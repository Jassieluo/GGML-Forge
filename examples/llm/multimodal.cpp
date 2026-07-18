#include "categories/llm/llm.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

bool print_piece(const char* text, size_t length, void*) {
    std::cout.write(text, static_cast<std::streamsize>(length));
    std::cout.flush();
    return true;
}

std::vector<unsigned char> read_file(const char* path) {
    std::ifstream input(path, std::ios::binary);
    return input
        ? std::vector<unsigned char>(std::istreambuf_iterator<char>(input), {})
        : std::vector<unsigned char>{};
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4 || argc > 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.gguf> <mmproj.gguf> <image> [question]\n";
        return 2;
    }

    const std::vector<unsigned char> image = read_file(argv[3]);
    if (image.empty()) {
        std::cerr << "Failed to read image: " << argv[3] << '\n';
        return 1;
    }

    llm_runtime_params runtime_params = llm_runtime_default_params();
    runtime_params.n_ctx = 8192;
    runtime_params.n_batch = 1024;
    runtime_params.n_gpu_layers = 99;
    llm_runtime_ptr runtime = llm_runtime_create(runtime_params);
    llm_model_params model_params = llm_model_default_params();
    model_params.model = argv[1];
    model_params.mmproj = argv[2];
    llm_model_ptr model = runtime
        ? llm_load_model_with_params(runtime, &model_params)
        : nullptr;
    llm_session_ptr session = model ? llm_create_session(model) : nullptr;
    const llm_capabilities capabilities = llm_model_get_capabilities(model);
    if (!session || !capabilities.vision) {
        std::cerr << "Failed to initialize a vision-capable LLM.\n";
        llm_free_session(session);
        llm_free_model(model);
        llm_runtime_free(runtime);
        return 1;
    }

    const char prefix[] = "User: ";
    const char* question = argc > 4
        ? argv[4]
        : "\nDescribe the main subject and visible colors in one short sentence.\nAssistant:";
    const llm_content_part parts[] = {
        {LLM_CONTENT_TEXT, prefix, sizeof(prefix) - 1, "text/plain"},
        {LLM_CONTENT_IMAGE, image.data(), image.size(), nullptr},
        {LLM_CONTENT_TEXT, question, std::char_traits<char>::length(question), "text/plain"},
    };
    llm_generation_params generation = llm_generation_default_params();
    generation.max_tokens = 128;
    const bool ok = llm_generate_content(
        session, parts, sizeof(parts) / sizeof(parts[0]), generation, print_piece, nullptr);
    std::cout << '\n';

    llm_free_session(session);
    llm_free_model(model);
    llm_runtime_free(runtime);
    return ok ? 0 : 1;
}
