#include "categories/visual_perception/image_classification.h"
#include "classification_internal.h"

#include "gguf.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <thread>

namespace {

std::string read_gguf_architecture(const char* path) {
    gguf_init_params params{};
    params.no_alloc = true;
    gguf_context* metadata = gguf_init_from_file(path, params);
    if (!metadata) return {};
    std::string architecture;
    const int key = gguf_find_key(metadata, "general.architecture");
    if (key >= 0 && gguf_get_kv_type(metadata, key) == GGUF_TYPE_STRING) {
        architecture = gguf_get_val_str(metadata, key);
    }
    gguf_free(metadata);
    return architecture;
}

bool image_valid(const classification_image* image) {
    return image && image->data && image->width > 0 && image->height > 0 &&
           (image->channels == 1 || image->channels == 3 || image->channels == 4);
}

bool copy_result(const visual_perception::classification::Result& source,
                 classification_result* output) {
    output->scores = nullptr;
    output->score_count = 0;
    if (source.scores.empty()) return true;
    const size_t bytes = source.scores.size() * sizeof(classification_score);
    auto* scores = static_cast<classification_score*>(std::malloc(bytes));
    if (!scores) return false;
    std::memcpy(scores, source.scores.data(), bytes);
    output->scores = scores;
    output->score_count = source.scores.size();
    return true;
}

} // namespace

classification_runtime_params classification_runtime_default_params(void) {
    const auto threads = std::thread::hardware_concurrency();
    return {"auto", threads > 0 ? threads : 4u};
}

classification_runtime_ptr classification_runtime_create(
    classification_runtime_params params) {
    auto runtime = std::make_unique<classification_runtime>();
    runtime->config.device = params.device && params.device[0] ? params.device : "auto";
    runtime->config.n_threads = params.n_threads > 0 ? params.n_threads : 1u;
    return runtime.release();
}

void classification_runtime_free(classification_runtime_ptr runtime) { delete runtime; }

const char* classification_runtime_get_device(classification_runtime_ptr runtime) {
    return runtime ? runtime->config.device.c_str() : nullptr;
}

uint32_t classification_runtime_get_thread_count(classification_runtime_ptr runtime) {
    return runtime ? runtime->config.n_threads : 0;
}

classification_model_ptr classification_load_model(
    classification_runtime_ptr runtime, const char* path) {
    if (!runtime || !path || !path[0]) return nullptr;
    const std::string architecture = read_gguf_architecture(path);
    if (architecture.empty()) return nullptr;
    auto provider = visual_perception::classification::ProviderRegistry::get().create_for(architecture);
    if (!provider) return nullptr;
    auto implementation = provider->load(path, runtime->config);
    if (!implementation) return nullptr;
    auto model = std::make_unique<classification_model>();
    model->provider_name = provider->name();
    model->implementation = std::move(implementation);
    return model.release();
}

void classification_free_model(classification_model_ptr model) { delete model; }

const char* classification_model_get_provider(classification_model_ptr model) {
    return model ? model->provider_name.c_str() : nullptr;
}

classification_capabilities classification_model_get_capabilities(
    classification_model_ptr model) {
    return model && model->implementation
        ? model->implementation->capabilities() : classification_capabilities{};
}

const char* classification_model_get_label(
    classification_model_ptr model, int32_t class_id) {
    return model && model->implementation ? model->implementation->label(class_id) : nullptr;
}

classification_session_ptr classification_create_session(classification_model_ptr model) {
    if (!model || !model->implementation) return nullptr;
    auto session = std::make_unique<classification_session>();
    session->model = model->implementation;
    session->implementation = session->model->create_session();
    return session->implementation ? session.release() : nullptr;
}

void classification_free_session(classification_session_ptr session) { delete session; }

classification_request_params classification_request_default_params(void) { return {5}; }

bool classification_classify(
    classification_session_ptr session, const classification_image* image,
    classification_request_params params, classification_result* result) {
    if (result) *result = {nullptr, 0};
    if (!session || !session->implementation || !result || !image_valid(image) ||
        params.top_k == 0) return false;
    visual_perception::classification::Request request;
    request.image = image;
    request.top_k = params.top_k;
    visual_perception::classification::Result provider_result;
    if (!session->implementation->classify(request, provider_result)) return false;
    return copy_result(provider_result, result);
}

void classification_free_result(classification_result* result) {
    if (!result) return;
    std::free(result->scores);
    result->scores = nullptr;
    result->score_count = 0;
}
