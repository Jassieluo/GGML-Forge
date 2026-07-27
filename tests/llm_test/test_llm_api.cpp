#include "categories/llm/llm.h"
#include "llm_provider.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

class FakeSession final : public llm::ILLMSession {
public:
    bool reset() override { return true; }
    bool generate(const llm::GenerationRequest&, llm::TextSink sink) override {
        return sink("generated", 9);
    }
    bool generate_chat(
        const std::vector<llm::ChatMessage>& messages,
        const llm::GenerationRequest& request,
        llm::TextSink sink) override {
        if (messages.empty() || request.max_tokens != 17) return false;
        if (messages.back().parts.empty()) return sink(messages.back().content.data(), messages.back().content.size());
        const auto& part = messages.back().parts.front();
        return sink(static_cast<const char*>(part.data), part.size);
    }
};

class FakeModel final : public llm::ILLMModel {
public:
    std::unique_ptr<llm::ILLMSession> create_session() override { return std::make_unique<FakeSession>(); }
    llm::Capabilities capabilities() const override { return {true, true}; }
    bool format_chat(const std::vector<llm::ChatMessage>&, std::string&) const override { return true; }
};

class FakeProvider final : public llm::ILLMProvider {
public:
    const char* name() const override { return "fake"; }
    std::shared_ptr<llm::ILLMModel> load(const llm::ModelConfig&, const llm::RuntimeConfig&) const override {
        return std::make_shared<FakeModel>();
    }
};

bool append_text(const char* text, size_t length, void* data) {
    static_cast<std::string*>(data)->append(text, length);
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
    const llm_runtime_params defaults = llm_runtime_default_params();
    const llm_generation_params generation_defaults = llm_generation_default_params();
    const llm_model_params model_defaults = llm_model_default_params();
    if (defaults.n_ctx == 0 || defaults.n_batch == 0 || defaults.n_threads == 0 ||
        !defaults.device || std::string(defaults.device) != "auto" ||
        generation_defaults.max_tokens <= 0 || model_defaults.model || model_defaults.mmproj) {
        std::cerr << "invalid LLM defaults\n";
        return 1;
    }
    if (llm_load_model(nullptr, "missing.gguf") || llm_create_session(nullptr) ||
        llm_session_reset(nullptr)) {
        std::cerr << "LLM null-handle contract failed\n";
        return 1;
    }
    const llm_capabilities empty = llm_model_get_capabilities(nullptr);
    if (empty.vision || empty.audio) {
        std::cerr << "null LLM model reported capabilities\n";
        return 1;
    }
    if (argc == 1) {
        llm::ProviderRegistry::get().register_provider("llama.cpp", [] {
            return std::make_unique<FakeProvider>();
        });
        llm_runtime_ptr fake_runtime = llm_runtime_create(defaults);
        if (!fake_runtime || std::string(llm_runtime_get_device(fake_runtime)) != "auto") {
            std::cerr << "LLM runtime device contract failed\n";
            return 1;
        }
        llm_model_ptr fake_model = llm_load_model(fake_runtime, "fake.gguf");
        llm_session_ptr fake_session = llm_create_session(fake_model);
        llm_generation_params fake_generation = generation_defaults;
        fake_generation.max_tokens = 17;
        std::string fake_output;
        const llm_chat_message text_messages[] = {{"user", "chat contract"}};
        if (!llm_generate_chat(fake_session, text_messages, 1, fake_generation,
                append_text, &fake_output) || fake_output != "chat contract") {
            std::cerr << "text chat contract failed\n";
            return 1;
        }
        fake_output.clear();
        const char content[] = "content contract";
        const llm_content_part content_parts[] = {
            {LLM_CONTENT_TEXT, content, sizeof(content) - 1, "text/plain"},
        };
        const llm_chat_content_message content_messages[] = {{"user", content_parts, 1}};
        if (!llm_generate_chat_content(fake_session, content_messages, 1, fake_generation,
                append_text, &fake_output) || fake_output != content) {
            std::cerr << "multimodal chat contract failed\n";
            return 1;
        }
        llm_free_session(fake_session);
        llm_free_model(fake_model);
        llm_runtime_free(fake_runtime);
        std::cout << "LLM API lifecycle and chat contract checks passed\n";
        return 0;
    }
    if (argc != 4) {
        std::cerr << "usage: test_llm_api <model.gguf> <mmproj.gguf> <image>\n";
        return 2;
    }

    llm_runtime_params runtime_params = defaults;
    runtime_params.n_ctx = 8192;
    runtime_params.n_batch = 1024;
    runtime_params.n_threads = 8;
    runtime_params.n_gpu_layers = 99;
    llm_runtime_ptr runtime = llm_runtime_create(runtime_params);
    llm_model_params model_params{argv[1], argv[2]};
    llm_model_ptr model = llm_load_model_with_params(runtime, &model_params);
    if (!model) {
        std::cerr << "failed to load multimodal model\n";
        llm_runtime_free(runtime);
        return 1;
    }
    const llm_capabilities capabilities = llm_model_get_capabilities(model);
    if (!capabilities.vision) {
        std::cerr << "loaded mmproj does not report vision support\n";
        llm_free_model(model);
        llm_runtime_free(runtime);
        return 1;
    }
    llm_session_ptr session = llm_create_session(model);
    const auto image = read_file(argv[3]);
    if (!session || image.empty()) {
        std::cerr << "failed to create session or read image\n";
        llm_free_session(session);
        llm_free_model(model);
        llm_runtime_free(runtime);
        return 1;
    }

    const char prefix[] = "<|im_start|>user\n";
    const char question[] =
        "\nDescribe the main subject and visible colors in this image in one short sentence."
        "<|im_end|>\n<|im_start|>assistant\n";
    const llm_content_part parts[] = {
        {LLM_CONTENT_TEXT, prefix, sizeof(prefix) - 1, "text/plain"},
        {LLM_CONTENT_IMAGE, image.data(), image.size(), "image/jpeg"},
        {LLM_CONTENT_TEXT, question, sizeof(question) - 1, "text/plain"},
    };
    llm_generation_params generation = generation_defaults;
    generation.max_tokens = 64;
    generation.temperature = 0.0f;
    std::string output;
    const bool ok = llm_generate_content(
        session, parts, 3, generation, append_text, &output);
    std::cout << '\n';

    llm_free_session(session);
    llm_free_model(model);
    llm_runtime_free(runtime);
    if (!ok || output.empty()) {
        std::cerr << "multimodal generation failed\n";
        return 1;
    }
    std::cout << "Multimodal LLM generation passed\n";
    return 0;
}
