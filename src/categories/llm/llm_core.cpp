#include "categories/llm/llm.h"
#include "llm_internal.h"

#include <algorithm>
#include <memory>
#include <thread>

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
    if (!runtime || !path || path[0] == '\0') return nullptr;
    auto provider = llm::ProviderRegistry::get().create("llama.cpp");
    if (!provider) return nullptr;
    auto implementation = provider->load(path, runtime->config);
    if (!implementation) return nullptr;
    auto model = std::make_unique<llm_model>();
    model->provider_name = provider->name();
    model->implementation = std::move(implementation);
    return model.release();
}

void llm_free_model(llm_model_ptr model) {
    delete model;
}

const char* llm_model_get_provider(llm_model_ptr model) {
    return model ? model->provider_name.c_str() : nullptr;
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
    request.max_tokens = std::max(0, params.max_tokens);
    request.temperature = params.temperature;
    request.top_k = params.top_k;
    request.top_p = params.top_p;
    request.seed = params.seed;
    return session->implementation->generate(request, [callback, user_data](const char* text, size_t size) {
        return callback(text, size, user_data);
    });
}
