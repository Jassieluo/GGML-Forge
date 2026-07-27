#include "categories/visual_perception/semantic_segmentation.h"
#include "segmentation_internal.h"

#include "gguf.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <thread>

namespace {

// The provider is chosen by the model file itself: GGUF `general.architecture`
// names the network family and providers register for the architectures they
// implement.
std::string read_gguf_architecture(const char* path) {
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = nullptr;
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

bool image_valid(const segmentation_image* image) {
    return image && image->data && image->width > 0 && image->height > 0 &&
           (image->channels == 1 || image->channels == 3 || image->channels == 4);
}

} // namespace

struct segmentation_runtime_params segmentation_runtime_default_params(void) {
    const auto threads = std::thread::hardware_concurrency();
    return {"auto", threads > 0 ? threads : 4u};
}

segmentation_runtime_ptr segmentation_runtime_create(struct segmentation_runtime_params params) {
    auto runtime = std::make_unique<segmentation_runtime>();
    runtime->config.device = params.device && params.device[0] ? params.device : "auto";
    runtime->config.n_threads = params.n_threads > 0 ? params.n_threads : 1u;
    return runtime.release();
}

void segmentation_runtime_free(segmentation_runtime_ptr runtime) { delete runtime; }

const char* segmentation_runtime_get_device(segmentation_runtime_ptr runtime) {
    return runtime ? runtime->config.device.c_str() : nullptr;
}

uint32_t segmentation_runtime_get_thread_count(segmentation_runtime_ptr runtime) {
    return runtime ? runtime->config.n_threads : 0;
}

segmentation_model_ptr segmentation_load_model(segmentation_runtime_ptr runtime, const char* path) {
    if (!runtime || !path || !path[0]) return nullptr;
    const std::string architecture = read_gguf_architecture(path);
    if (architecture.empty()) {
        std::cerr << "[Segmentation Core] Error: Not a GGUF model with general.architecture: "
                  << path << std::endl;
        return nullptr;
    }
    auto provider = segmentation::ProviderRegistry::get().create_for(architecture);
    if (!provider) {
        std::cerr << "[Segmentation Core] Error: Unsupported architecture: " << architecture
                  << std::endl;
        return nullptr;
    }
    auto implementation = provider->load(path, runtime->config);
    if (!implementation) {
        std::cerr << "[Segmentation Core] Error: Failed to load provider: " << provider->name()
                  << std::endl;
        return nullptr;
    }
    auto model = std::make_unique<segmentation_model>();
    model->provider_name = provider->name();
    model->implementation = std::move(implementation);
    return model.release();
}

void segmentation_free_model(segmentation_model_ptr model) { delete model; }

const char* segmentation_model_get_provider(segmentation_model_ptr model) {
    return model ? model->provider_name.c_str() : nullptr;
}

struct segmentation_capabilities segmentation_model_get_capabilities(segmentation_model_ptr model) {
    return model && model->implementation
        ? model->implementation->capabilities()
        : segmentation_capabilities{};
}

const char* segmentation_model_get_label(segmentation_model_ptr model, int32_t class_id) {
    return model && model->implementation ? model->implementation->label(class_id) : nullptr;
}

bool segmentation_model_get_color(segmentation_model_ptr model, int32_t class_id, uint8_t rgb[3]) {
    return model && model->implementation && rgb &&
           model->implementation->color(class_id, rgb);
}

segmentation_session_ptr segmentation_create_session(segmentation_model_ptr model) {
    if (!model || !model->implementation) return nullptr;
    auto session = std::make_unique<segmentation_session>();
    session->model = model->implementation;
    session->implementation = session->model->create_session();
    return session->implementation ? session.release() : nullptr;
}

void segmentation_free_session(segmentation_session_ptr session) { delete session; }

struct segmentation_request_params segmentation_request_default_params(void) {
    return {false};
}

bool segmentation_segment(
    segmentation_session_ptr session,
    const struct segmentation_image* image,
    struct segmentation_request_params params,
    struct segmentation_result* result
) {
    if (result) *result = {0, 0, nullptr, nullptr};
    if (!session || !session->implementation || !result || !image_valid(image)) return false;
    if (params.want_confidence && !session->model->capabilities().confidence) return false;

    segmentation::Request request;
    request.image = image;
    request.want_confidence = params.want_confidence;

    segmentation::Result provider_result;
    if (!session->implementation->segment(request, provider_result)) return false;
    const size_t count = static_cast<size_t>(provider_result.width) * provider_result.height;
    if (count == 0 || provider_result.class_map.size() != count) return false;
    if (params.want_confidence && provider_result.confidence.size() != count) return false;

    auto* class_map = static_cast<int32_t*>(std::malloc(count * sizeof(int32_t)));
    if (!class_map) return false;
    std::memcpy(class_map, provider_result.class_map.data(), count * sizeof(int32_t));
    float* confidence = nullptr;
    if (params.want_confidence) {
        confidence = static_cast<float*>(std::malloc(count * sizeof(float)));
        if (!confidence) {
            std::free(class_map);
            return false;
        }
        std::memcpy(confidence, provider_result.confidence.data(), count * sizeof(float));
    }
    result->width = provider_result.width;
    result->height = provider_result.height;
    result->class_map = class_map;
    result->confidence = confidence;
    return true;
}

void segmentation_free_result(struct segmentation_result* result) {
    if (!result) return;
    std::free(result->class_map);
    std::free(result->confidence);
    result->class_map = nullptr;
    result->confidence = nullptr;
    result->width = 0;
    result->height = 0;
}
