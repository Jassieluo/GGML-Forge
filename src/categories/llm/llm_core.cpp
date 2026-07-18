#include "categories/llm/llm.h"
#include "llm_internal.h"

#include <algorithm>
#include <memory>
#include <string>
#include <thread>

namespace {

llm_model_ptr load_model(llm_runtime_ptr runtime, const llm_model_params& params) {
    if (!runtime || !params.model || params.model[0] == '\0') return nullptr;
    auto provider = llm::ProviderRegistry::get().create("llama.cpp");
    if (!provider) return nullptr;
    llm::ModelConfig config;
    config.model = params.model;
    config.mmproj = params.mmproj ? params.mmproj : "";
    auto implementation = provider->load(config, runtime->config);
    if (!implementation) return nullptr;
    auto model = std::make_unique<llm_model>();
    model->provider_name = provider->name();
    model->implementation = std::move(implementation);
    return model.release();
}

void apply_generation(llm::GenerationRequest& request, const llm_generation_params& params) {
    request.max_tokens = std::max(0, params.max_tokens);
    request.temperature = params.temperature;
    request.top_k = params.top_k;
    request.top_p = params.top_p;
    request.seed = params.seed;
}

} // namespace

struct llm_runtime_params llm_runtime_default_params(void) {
    const auto threads = std::thread::hardware_concurrency();
    return {4096u, 512u, threads > 0 ? threads : 4u, 0};
}

llm_runtime_ptr llm_runtime_create(struct llm_runtime_params params) {
    auto runtime = std::make_unique<llm_runtime>();
    runtime->config.n_ctx = params.n_ctx > 0 ? params.n_ctx : 4096u;
    runtime->config.n_batch = params.n_batch > 0 ? params.n_batch : 512u;
    runtime->config.n_threads = params.n_threads > 0 ? params.n_threads : 1u;
    runtime->config.n_gpu_layers = params.n_gpu_layers;
    return runtime.release();
}

void llm_runtime_free(llm_runtime_ptr runtime) {
    delete runtime;
}

llm_model_ptr llm_load_model(llm_runtime_ptr runtime, const char* path) {
    const llm_model_params params{path, nullptr};
    return load_model(runtime, params);
}

llm_model_params llm_model_default_params(void) {
    return {nullptr, nullptr};
}

llm_model_ptr llm_load_model_with_params(llm_runtime_ptr runtime, const llm_model_params* params) {
    return params ? load_model(runtime, *params) : nullptr;
}

void llm_free_model(llm_model_ptr model) {
    delete model;
}

const char* llm_model_get_provider(llm_model_ptr model) {
    return model ? model->provider_name.c_str() : nullptr;
}

llm_capabilities llm_model_get_capabilities(llm_model_ptr model) {
    if (!model || !model->implementation) return {};
    const auto capabilities = model->implementation->capabilities();
    return {capabilities.vision, capabilities.audio};
}

llm_session_ptr llm_create_session(llm_model_ptr model) {
    if (!model || !model->implementation) return nullptr;
    auto session = std::make_unique<llm_session>();
    session->model = model->implementation;
    session->implementation = session->model->create_session();
    return session->implementation ? session.release() : nullptr;
}

void llm_free_session(llm_session_ptr session) {
    delete session;
}

bool llm_session_reset(llm_session_ptr session) {
    return session && session->implementation && session->implementation->reset();
}

struct llm_generation_params llm_generation_default_params(void) {
    return {128, 0.8f, 40, 0.95f, 0xFFFFFFFFu};
}

bool llm_generate(
    llm_session_ptr session,
    const char* prompt,
    struct llm_generation_params params,
    llm_text_callback callback,
    void* user_data
) {
    if (!session || !session->implementation || !prompt || !callback) return false;
    llm::GenerationRequest request;
    request.prompt = prompt;
    apply_generation(request, params);
    return session->implementation->generate(request, [callback, user_data](const char* text, size_t size) {
        return callback(text, size, user_data);
    });
}

bool llm_generate_content(
    llm_session_ptr session,
    const llm_content_part* parts,
    size_t part_count,
    llm_generation_params params,
    llm_text_callback callback,
    void* user_data
) {
    if (!session || !session->implementation || !parts || part_count == 0 || !callback) return false;
    llm::GenerationRequest request;
    request.content.reserve(part_count);
    for (size_t i = 0; i < part_count; ++i) {
        if (!parts[i].data || parts[i].size == 0) return false;
        llm::ContentPart::Type type;
        switch (parts[i].type) {
            case LLM_CONTENT_TEXT: type = llm::ContentPart::Type::Text; break;
            case LLM_CONTENT_IMAGE: type = llm::ContentPart::Type::Image; break;
            case LLM_CONTENT_AUDIO: type = llm::ContentPart::Type::Audio; break;
            default: return false;
        }
        request.content.push_back({type, parts[i].data, parts[i].size});
    }
    apply_generation(request, params);
    return session->implementation->generate(request, [callback, user_data](const char* text, size_t size) {
        return callback(text, size, user_data);
    });
}
