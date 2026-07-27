#include "providers/instance_provider.h"
#include "providers/yolo/v8/yolo_v8.h"
#include "providers/yolo/yolo_common.h"

#include "ggml-backend.h"
#include "gguf.h"
#include "nn/core/executor.h"
#include "nn/io/gguf.h"
#include "nn/io/load.h"
#include "ops/cpu.h"
#include "ops/ops.h"

#include <algorithm>
#include <array>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

extern "C" void ggml_ops_ext_cpu_init();
#ifdef GGML_USE_CUDA
extern "C" void ggml_ops_ext_cuda_init();
#endif
#ifdef GGML_USE_SYCL
extern "C" void ggml_ops_ext_sycl_init();
#endif

namespace visual_perception::instance::yolo {
namespace {

void initialize_backends() {
    static std::once_flag flag;
    std::call_once(flag, [] {
        ggml_backend_load_all();
        ggml_ops_ext_cpu_init();
#ifdef GGML_USE_CUDA
        ggml_ops_ext_cuda_init();
#endif
#ifdef GGML_USE_SYCL
        ggml_ops_ext_sycl_init();
#endif
    });
}

class OpsHookGuard {
public:
    OpsHookGuard() { ggml_ops_ext::acquire_ops_hook(); }
    ~OpsHookGuard() {
        if (active_) ggml_ops_ext::release_ops_hook();
    }
    void transfer_to_model() { active_ = false; }

private:
    bool active_ = true;
};

ggml_backend_dev_t select_device(const std::string& requested) {
    if (requested.empty() || requested == "auto") {
        for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
            ggml_backend_dev_t device = ggml_backend_dev_get(index);
            if (!device) continue;
            const auto type = ggml_backend_dev_type(device);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                type == GGML_BACKEND_DEVICE_TYPE_IGPU) return device;
        }
    }
    if (requested.empty() || requested == "auto" || requested == "cpu") {
        for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
            ggml_backend_dev_t device = ggml_backend_dev_get(index);
            if (device && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                return device;
            }
        }
        return nullptr;
    }
    return ggml_backend_dev_by_name(requested.c_str());
}

bool read_string(const gguf_context* metadata, const char* name, std::string& value) {
    const int64_t key = gguf_find_key(metadata, name);
    if (key < 0 || gguf_get_kv_type(metadata, key) != GGUF_TYPE_STRING) return false;
    const char* text = gguf_get_val_str(metadata, key);
    if (!text) return false;
    value = text;
    return true;
}

bool read_uint32(const gguf_context* metadata, const char* name, uint32_t& value) {
    const int64_t key = gguf_find_key(metadata, name);
    if (key < 0 || gguf_get_kv_type(metadata, key) != GGUF_TYPE_UINT32) return false;
    value = gguf_get_val_u32(metadata, key);
    return true;
}

bool read_bool(const gguf_context* metadata, const char* name, bool& value) {
    const int64_t key = gguf_find_key(metadata, name);
    if (key < 0 || gguf_get_kv_type(metadata, key) != GGUF_TYPE_BOOL) return false;
    value = gguf_get_val_bool(metadata, key);
    return true;
}

bool read_uint32_array(const gguf_context* metadata, const char* name,
                       std::array<uint32_t, 3>& value) {
    const int64_t key = gguf_find_key(metadata, name);
    if (key < 0 || gguf_get_kv_type(metadata, key) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_n(metadata, key) != value.size()) return false;
    const void* data = gguf_get_arr_data(metadata, key);
    if (!data) return false;
    switch (gguf_get_arr_type(metadata, key)) {
        case GGUF_TYPE_UINT8: {
            const auto* items = static_cast<const uint8_t*>(data);
            std::copy(items, items + value.size(), value.begin()); return true;
        }
        case GGUF_TYPE_UINT16: {
            const auto* items = static_cast<const uint16_t*>(data);
            std::copy(items, items + value.size(), value.begin()); return true;
        }
        case GGUF_TYPE_UINT32: {
            const auto* items = static_cast<const uint32_t*>(data);
            std::copy(items, items + value.size(), value.begin()); return true;
        }
        case GGUF_TYPE_INT32: {
            const auto* items = static_cast<const int32_t*>(data);
            for (size_t index = 0; index < value.size(); ++index) {
                if (items[index] <= 0) return false;
                value[index] = static_cast<uint32_t>(items[index]);
            }
            return true;
        }
        default: return false;
    }
}

bool read_labels(const gguf_context* metadata, uint32_t class_count,
                 std::vector<std::string>& labels) {
    labels.clear();
    const int64_t key = gguf_find_key(metadata, "instance.labels");
    if (key < 0) return true;
    if (gguf_get_kv_type(metadata, key) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(metadata, key) != GGUF_TYPE_STRING ||
        gguf_get_arr_n(metadata, key) != class_count) return false;
    labels.reserve(class_count);
    for (uint32_t index = 0; index < class_count; ++index) {
        const char* label = gguf_get_arr_str(metadata, key, index);
        if (!label) return false;
        labels.emplace_back(label);
    }
    return true;
}

bool read_model_config(const gguf_context* metadata, ModelConfig& config,
                       std::string& error) {
    config = {};
    std::string architecture;
    std::string normalization;
    bool boxes = false;
    bool oriented = false;
    bool masks = false;
    bool keypoints = false;
    if (!read_string(metadata, "general.architecture", architecture) ||
        (architecture != "yolo_v8" && architecture != "yolo_v8_seg" &&
         architecture != "yolo_v8_pose" && architecture != "yolo_v8_obb") ||
        !read_string(metadata, "yolo.version", config.version) || config.version != "v8" ||
        !read_uint32(metadata, "instance.input.width", config.input_width) ||
        !read_uint32(metadata, "instance.input.height", config.input_height) ||
        !read_uint32(metadata, "instance.class_count", config.class_count) ||
        !read_uint32(metadata, "yolo.reg_max", config.reg_max) ||
        !read_string(metadata, "instance.input.normalization", normalization) ||
        normalization != "zero_to_one" ||
        !read_bool(metadata, "instance.task.boxes", boxes) || !boxes ||
        !read_bool(metadata, "instance.task.oriented_boxes", oriented) ||
        !read_bool(metadata, "instance.task.masks", masks) ||
        !read_bool(metadata, "instance.task.keypoints", keypoints) ||
        !read_uint32_array(metadata, "yolo.strides", config.strides) ||
        !read_labels(metadata, config.class_count, config.labels)) {
        error = "invalid YOLOv8 instance-perception metadata";
        return false;
    }
    config.instance_masks = masks;
    config.keypoints = keypoints;
    config.oriented_boxes = oriented;
    if ((architecture == "yolo_v8_seg") != config.instance_masks ||
        (architecture == "yolo_v8_pose") != config.keypoints ||
        (architecture == "yolo_v8_obb") != config.oriented_boxes ||
        static_cast<int>(config.instance_masks) + static_cast<int>(config.keypoints) +
                static_cast<int>(config.oriented_boxes) > 1) {
        error = "YOLOv8 architecture/task metadata mismatch";
        return false;
    }
    if (config.instance_masks &&
        (!read_uint32(metadata, "yolo.mask_count", config.mask_count) ||
         config.mask_count == 0)) {
        error = "invalid YOLOv8 Segment mask metadata";
        return false;
    }
    if (config.keypoints &&
        (!read_uint32(metadata, "instance.keypoint_count", config.keypoint_count) ||
         !read_uint32(metadata, "yolo.keypoint_dimensions", config.keypoint_dimensions) ||
         config.keypoint_count == 0 ||
         (config.keypoint_dimensions != 2 && config.keypoint_dimensions != 3))) {
        error = "invalid YOLOv8 Pose keypoint metadata";
        return false;
    }
    if (config.oriented_boxes &&
        (!read_uint32(metadata, "yolo.angle_count", config.angle_count) ||
         config.angle_count != 1)) {
        error = "invalid YOLOv8 OBB angle metadata";
        return false;
    }
    if (config.input_width == 0 || config.input_height == 0 ||
        config.input_width % 32 != 0 || config.input_height % 32 != 0 ||
        config.class_count == 0 || config.reg_max == 0) {
        error = "invalid YOLOv8 input or output dimensions";
        return false;
    }
    return true;
}

class YoloModel;

class YoloSession final : public IInstanceSession {
public:
    explicit YoloSession(std::shared_ptr<YoloModel> model) : model_(std::move(model)) {}
    bool perceive(const Request& request, Result& result) override;

private:
    std::shared_ptr<YoloModel> model_;
};

class YoloModel final : public IInstanceModel,
                        public std::enable_shared_from_this<YoloModel> {
public:
    YoloModel(RuntimeConfig runtime, ggml_backend_t backend, ModelConfig config,
              std::unique_ptr<v8::Model> network)
        : runtime_(std::move(runtime)), backend_(backend), config_(std::move(config)),
          network_(std::move(network)) {}
    ~YoloModel() override {
        network_.reset();
        if (backend_) ggml_backend_free(backend_);
        ggml_ops_ext::release_ops_hook();
    }

    std::unique_ptr<IInstanceSession> create_session() override {
        return std::make_unique<YoloSession>(shared_from_this());
    }
    instance_capabilities capabilities() const override {
        return {true, config_.oriented_boxes, config_.instance_masks, config_.keypoints,
                config_.class_count, config_.keypoint_count};
    }
    const char* label(int32_t class_id) const override {
        return class_id >= 0 && static_cast<size_t>(class_id) < config_.labels.size()
            ? config_.labels[static_cast<size_t>(class_id)].c_str() : nullptr;
    }

    bool run(const Request& request, Result& result) {
        LetterboxImage letterbox;
        if (!request.image || !make_letterbox(
                *request.image, config_.input_width, config_.input_height, letterbox)) return false;
        try {
            nn::Context context(64 * 1024 * 1024, true);
            ggml_tensor* input = context.input<float>(
                "yolo.input", {config_.input_width, config_.input_height, 3, 1},
                nn::data::borrow(letterbox.pixels));
            v8::Outputs outputs = network_->forward_outputs(context, input, backend_);
            if (!outputs.predictions ||
                (request.task == INSTANCE_TASK_MASKS && !outputs.prototypes)) return false;
            ggml_cgraph* graph = request.task == INSTANCE_TASK_MASKS
                ? context.build({outputs.predictions, outputs.prototypes}, 65536)
                : context.build(outputs.predictions, 65536);
            nn::Executor executor(backend_);
            executor.prepare(context, graph);
            executor.compute(context, graph);
            const size_t count = static_cast<size_t>(ggml_nelements(outputs.predictions));
            std::vector<float> values(count);
            context.read(outputs.predictions, values.data(), values.size());
            if (request.task == INSTANCE_TASK_MASKS) {
                const uint32_t prototype_width = static_cast<uint32_t>(outputs.prototypes->ne[0]);
                const uint32_t prototype_height = static_cast<uint32_t>(outputs.prototypes->ne[1]);
                if (outputs.prototypes->ne[2] != config_.mask_count) return false;
                std::vector<float> prototypes(
                    static_cast<size_t>(ggml_nelements(outputs.prototypes)));
                context.read(outputs.prototypes, prototypes.data(), prototypes.size());
                return decode_instance_masks(
                    values.data(), values.size(), prototypes.data(), prototypes.size(),
                    prototype_width, prototype_height, config_, letterbox,
                    *request.image, request, result);
            }
            if (request.task == INSTANCE_TASK_KEYPOINTS) {
                return decode_keypoints(
                    values.data(), values.size(), config_, letterbox,
                    *request.image, request, result);
            }
            return decode_detections(
                values.data(), values.size(), config_, letterbox, *request.image, request, result);
        } catch (const std::exception& exception) {
            std::cerr << "[YOLO] inference failed: " << exception.what() << '\n';
            return false;
        }
    }

private:
    RuntimeConfig runtime_;
    ggml_backend_t backend_ = nullptr;
    ModelConfig config_;
    std::unique_ptr<v8::Model> network_;
};

bool YoloSession::perceive(const Request& request, Result& result) {
    return model_ && model_->run(request, result);
}

class YoloProvider final : public IInstanceProvider {
public:
    const char* name() const override { return "yolo"; }

    std::shared_ptr<IInstanceModel> load(
        const std::string& path, const RuntimeConfig& runtime) const override {
        initialize_backends();
        ggml_backend_dev_t device = select_device(runtime.device);
        if (!device) {
            std::cerr << "[YOLO] backend device is unavailable: " << runtime.device << '\n';
            return nullptr;
        }
        ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
        if (!backend) return nullptr;
        OpsHookGuard hook;
        if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            if (ggml_backend_reg_t registration = ggml_backend_dev_backend_reg(device)) {
                using SetThreads = void (*)(ggml_backend_t, int);
                auto set_threads = reinterpret_cast<SetThreads>(
                    ggml_backend_reg_get_proc_address(
                        registration, "ggml_backend_cpu_set_n_threads"));
                if (set_threads) set_threads(backend, static_cast<int>(runtime.n_threads));
            }
            ggml_ops_ext_cpu_set_n_threads(backend, static_cast<int>(runtime.n_threads));
        }

        try {
            nn::io::GGUFSource source(path);
            ModelConfig common_config;
            std::string error;
            if (!read_model_config(source.metadata_context(), common_config, error)) {
                std::cerr << "[YOLO] " << error << '\n';
                ggml_backend_free(backend);
                return nullptr;
            }
            v8::Config version_config = v8::Config::from_source(
                source, static_cast<int>(common_config.class_count),
                static_cast<int>(common_config.reg_max),
                common_config.instance_masks
                    ? v8::Task::instance_segmentation
                    : (common_config.keypoints
                           ? v8::Task::pose
                           : (common_config.oriented_boxes
                                  ? v8::Task::oriented_detection : v8::Task::detection)),
                static_cast<int>(common_config.mask_count),
                static_cast<int>(common_config.keypoint_count),
                static_cast<int>(common_config.keypoint_dimensions),
                static_cast<int>(common_config.angle_count), error);
            if (!error.empty()) {
                std::cerr << "[YOLO] " << error << '\n';
                ggml_backend_free(backend);
                return nullptr;
            }
            auto network = std::make_unique<v8::Model>(version_config);
            nn::io::LoadResult loaded = nn::io::load_into(*network, source, backend);
            if (!loaded) {
                std::cerr << "[YOLO] " << loaded.error << '\n';
                ggml_backend_free(backend);
                return nullptr;
            }
            network->to(backend);
            auto model = std::make_shared<YoloModel>(
                runtime, backend, std::move(common_config), std::move(network));
            hook.transfer_to_model();
            return model;
        } catch (const std::exception& exception) {
            std::cerr << "[YOLO] model load failed: " << exception.what() << '\n';
            ggml_backend_free(backend);
            return nullptr;
        }
    }
};

[[maybe_unused]] const bool registered = [] {
    ProviderRegistry::get().register_architecture("yolo_v8", [] {
        return std::make_unique<YoloProvider>();
    });
    ProviderRegistry::get().register_architecture("yolo_v8_seg", [] {
        return std::make_unique<YoloProvider>();
    });
    ProviderRegistry::get().register_architecture("yolo_v8_pose", [] {
        return std::make_unique<YoloProvider>();
    });
    ProviderRegistry::get().register_architecture("yolo_v8_obb", [] {
        return std::make_unique<YoloProvider>();
    });
    return true;
}();

} // namespace
} // namespace visual_perception::instance::yolo
