#include "categories/object_detection/object_detection.h"
#include "detection_internal.h"

#include "gguf.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
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

bool image_valid(const detection_image* image) {
    return image && image->data && image->width > 0 && image->height > 0 &&
           (image->channels == 1 || image->channels == 3 || image->channels == 4);
}

bool task_supported(const detection_capabilities& capabilities, detection_task task) {
    switch (task) {
        case DETECTION_TASK_BOXES:          return capabilities.boxes;
        case DETECTION_TASK_ORIENTED_BOXES: return capabilities.oriented_boxes;
        case DETECTION_TASK_INSTANCE_MASKS: return capabilities.instance_masks;
        case DETECTION_TASK_KEYPOINTS:      return capabilities.keypoints;
    }
    return false;
}

// Deep-copies the provider result so everything reachable from
// detection_result is owned by it and released by detection_free_result.
bool copy_result(const detection::Result& source, detection_result* out) {
    out->instances = nullptr;
    out->instance_count = 0;
    if (source.instances.empty()) return true;
    auto* instances = static_cast<detection_instance*>(
        std::calloc(source.instances.size(), sizeof(detection_instance)));
    if (!instances) return false;
    size_t count = 0;
    for (const detection_instance& instance : source.instances) {
        detection_instance copy = instance;
        copy.mask = nullptr;
        copy.keypoints = nullptr;
        if (instance.mask && instance.mask_width > 0 && instance.mask_height > 0) {
            const size_t bytes = static_cast<size_t>(instance.mask_width) * instance.mask_height;
            auto* mask = static_cast<uint8_t*>(std::malloc(bytes));
            if (!mask) {
                detection_result partial{instances, count};
                detection_free_result(&partial);
                return false;
            }
            std::memcpy(mask, instance.mask, bytes);
            copy.mask = mask;
        }
        if (instance.keypoints && instance.keypoint_count > 0) {
            const size_t bytes = instance.keypoint_count * sizeof(detection_keypoint);
            auto* keypoints = static_cast<detection_keypoint*>(std::malloc(bytes));
            if (!keypoints) {
                std::free(const_cast<uint8_t*>(copy.mask));
                detection_result partial{instances, count};
                detection_free_result(&partial);
                return false;
            }
            std::memcpy(keypoints, instance.keypoints, bytes);
            copy.keypoints = keypoints;
        }
        instances[count++] = copy;
    }
    out->instances = instances;
    out->instance_count = count;
    return true;
}

} // namespace

struct detection_runtime_params detection_runtime_default_params(void) {
    const auto threads = std::thread::hardware_concurrency();
    return {"auto", threads > 0 ? threads : 4u};
}

detection_runtime_ptr detection_runtime_create(struct detection_runtime_params params) {
    auto runtime = std::make_unique<detection_runtime>();
    runtime->config.device = params.device && params.device[0] ? params.device : "auto";
    runtime->config.n_threads = params.n_threads > 0 ? params.n_threads : 1u;
    return runtime.release();
}

void detection_runtime_free(detection_runtime_ptr runtime) { delete runtime; }

const char* detection_runtime_get_device(detection_runtime_ptr runtime) {
    return runtime ? runtime->config.device.c_str() : nullptr;
}

uint32_t detection_runtime_get_thread_count(detection_runtime_ptr runtime) {
    return runtime ? runtime->config.n_threads : 0;
}

detection_model_ptr detection_load_model(detection_runtime_ptr runtime, const char* path) {
    if (!runtime || !path || !path[0]) return nullptr;
    const std::string architecture = read_gguf_architecture(path);
    if (architecture.empty()) {
        std::cerr << "[Detection Core] Error: Not a GGUF model with general.architecture: "
                  << path << std::endl;
        return nullptr;
    }
    auto provider = detection::ProviderRegistry::get().create_for(architecture);
    if (!provider) {
        std::cerr << "[Detection Core] Error: Unsupported architecture: " << architecture
                  << std::endl;
        return nullptr;
    }
    auto implementation = provider->load(path, runtime->config);
    if (!implementation) {
        std::cerr << "[Detection Core] Error: Failed to load provider: " << provider->name()
                  << std::endl;
        return nullptr;
    }
    auto model = std::make_unique<detection_model>();
    model->provider_name = provider->name();
    model->implementation = std::move(implementation);
    return model.release();
}

void detection_free_model(detection_model_ptr model) { delete model; }

const char* detection_model_get_provider(detection_model_ptr model) {
    return model ? model->provider_name.c_str() : nullptr;
}

struct detection_capabilities detection_model_get_capabilities(detection_model_ptr model) {
    return model && model->implementation
        ? model->implementation->capabilities()
        : detection_capabilities{};
}

const char* detection_model_get_label(detection_model_ptr model, int32_t class_id) {
    return model && model->implementation ? model->implementation->label(class_id) : nullptr;
}

detection_session_ptr detection_create_session(detection_model_ptr model) {
    if (!model || !model->implementation) return nullptr;
    auto session = std::make_unique<detection_session>();
    session->model = model->implementation;
    session->implementation = session->model->create_session();
    return session->implementation ? session.release() : nullptr;
}

void detection_free_session(detection_session_ptr session) { delete session; }

struct detection_request_params detection_request_default_params(void) {
    return {DETECTION_TASK_BOXES, 0.25f, 0.45f, 300};
}

bool detection_detect(
    detection_session_ptr session,
    const struct detection_image* image,
    struct detection_request_params params,
    struct detection_result* result
) {
    if (result) *result = {nullptr, 0};
    if (!session || !session->implementation || !result || !image_valid(image) ||
        params.score_threshold < 0.0f || params.score_threshold > 1.0f ||
        params.iou_threshold <= 0.0f || params.iou_threshold > 1.0f ||
        params.max_instances < 1) return false;
    if (!task_supported(session->model->capabilities(), params.task)) return false;

    detection::Request request;
    request.image = image;
    request.task = params.task;
    request.score_threshold = params.score_threshold;
    request.iou_threshold = params.iou_threshold;
    request.max_instances = params.max_instances;

    detection::Result provider_result;
    if (!session->implementation->detect(request, provider_result)) return false;
    return copy_result(provider_result, result);
}

void detection_free_result(struct detection_result* result) {
    if (!result || !result->instances) return;
    for (size_t i = 0; i < result->instance_count; ++i) {
        std::free(const_cast<uint8_t*>(result->instances[i].mask));
        std::free(const_cast<detection_keypoint*>(result->instances[i].keypoints));
    }
    std::free(result->instances);
    result->instances = nullptr;
    result->instance_count = 0;
}
