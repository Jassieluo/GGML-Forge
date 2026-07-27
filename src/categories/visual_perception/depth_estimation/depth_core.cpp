#include "categories/visual_perception/depth_estimation.h"
#include "depth_internal.h"

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

bool image_valid(const depth_image* image) {
    return image && image->data && image->width > 0 && image->height > 0 &&
           (image->channels == 1 || image->channels == 3 || image->channels == 4);
}

} // namespace

struct depth_runtime_params depth_runtime_default_params(void) {
    const auto threads = std::thread::hardware_concurrency();
    return {"auto", threads > 0 ? threads : 4u};
}

depth_runtime_ptr depth_runtime_create(struct depth_runtime_params params) {
    auto runtime = std::make_unique<depth_runtime>();
    runtime->config.device = params.device && params.device[0] ? params.device : "auto";
    runtime->config.n_threads = params.n_threads > 0 ? params.n_threads : 1u;
    return runtime.release();
}

void depth_runtime_free(depth_runtime_ptr runtime) { delete runtime; }

const char* depth_runtime_get_device(depth_runtime_ptr runtime) {
    return runtime ? runtime->config.device.c_str() : nullptr;
}

uint32_t depth_runtime_get_thread_count(depth_runtime_ptr runtime) {
    return runtime ? runtime->config.n_threads : 0;
}

depth_model_ptr depth_load_model(depth_runtime_ptr runtime, const char* path) {
    if (!runtime || !path || !path[0]) return nullptr;
    const std::string architecture = read_gguf_architecture(path);
    if (architecture.empty()) {
        std::cerr << "[Depth Core] Error: Not a GGUF model with general.architecture: "
                  << path << std::endl;
        return nullptr;
    }
    auto provider = depth::ProviderRegistry::get().create_for(architecture);
    if (!provider) {
        std::cerr << "[Depth Core] Error: Unsupported architecture: " << architecture << std::endl;
        return nullptr;
    }
    auto implementation = provider->load(path, runtime->config);
    if (!implementation) {
        std::cerr << "[Depth Core] Error: Failed to load provider: " << provider->name()
                  << std::endl;
        return nullptr;
    }
    auto model = std::make_unique<depth_model>();
    model->provider_name = provider->name();
    model->implementation = std::move(implementation);
    return model.release();
}

void depth_free_model(depth_model_ptr model) { delete model; }

const char* depth_model_get_provider(depth_model_ptr model) {
    return model ? model->provider_name.c_str() : nullptr;
}

struct depth_capabilities depth_model_get_capabilities(depth_model_ptr model) {
    return model && model->implementation
        ? model->implementation->capabilities()
        : depth_capabilities{};
}

depth_session_ptr depth_create_session(depth_model_ptr model) {
    if (!model || !model->implementation) return nullptr;
    auto session = std::make_unique<depth_session>();
    session->model = model->implementation;
    session->implementation = session->model->create_session();
    return session->implementation ? session.release() : nullptr;
}

void depth_free_session(depth_session_ptr session) { delete session; }

struct depth_request_params depth_request_default_params(void) {
    return {DEPTH_TASK_MONOCULAR};
}

bool depth_estimate(
    depth_session_ptr session,
    const struct depth_image* image,
    const struct depth_image* right,
    struct depth_request_params params,
    struct depth_map* map
) {
    if (map) *map = {0, 0, DEPTH_MAP_RELATIVE, nullptr};
    if (!session || !session->implementation || !map || !image_valid(image)) return false;
    if (params.task == DEPTH_TASK_MONOCULAR) {
        if (right) return false;
    } else if (params.task == DEPTH_TASK_STEREO) {
        // A rectified pair shares dimensions by construction.
        if (!image_valid(right) || right->width != image->width ||
            right->height != image->height) return false;
    } else {
        return false;
    }
    const depth_capabilities capabilities = session->model->capabilities();
    if (params.task == DEPTH_TASK_MONOCULAR ? !capabilities.monocular : !capabilities.stereo) {
        return false;
    }

    depth::Request request;
    request.image = image;
    request.right = right;
    request.task = params.task;

    depth::Result result;
    if (!session->implementation->estimate(request, result)) return false;
    const size_t count = static_cast<size_t>(result.width) * result.height;
    if (count == 0 || result.values.size() != count) return false;
    auto* values = static_cast<float*>(std::malloc(count * sizeof(float)));
    if (!values) return false;
    std::memcpy(values, result.values.data(), count * sizeof(float));
    map->width = result.width;
    map->height = result.height;
    map->kind = result.kind;
    map->data = values;
    return true;
}

void depth_free_map(struct depth_map* map) {
    if (!map) return;
    std::free(map->data);
    map->data = nullptr;
    map->width = 0;
    map->height = 0;
}
