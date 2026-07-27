#include "providers/classification_provider.h"
#include "providers/yolo/v8/classification_model.h"

#include "ggml-backend.h"
#include "gguf.h"
#include "nn/core/executor.h"
#include "nn/io/gguf.h"
#include "nn/io/load.h"
#include "ops/cpu.h"
#include "ops/ops.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
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

namespace visual_perception::classification::yolo {
namespace {

namespace model_v8 = visual_perception::yolo::v8::classification;

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
    ~OpsHookGuard() { if (active_) ggml_ops_ext::release_ops_hook(); }
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

bool read_labels(const gguf_context* metadata, uint32_t class_count,
                 std::vector<std::string>& labels) {
    const int64_t key = gguf_find_key(metadata, "classification.labels");
    if (key < 0 || gguf_get_kv_type(metadata, key) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(metadata, key) != GGUF_TYPE_STRING ||
        gguf_get_arr_n(metadata, key) != class_count) return false;
    labels.clear();
    labels.reserve(class_count);
    for (uint32_t index = 0; index < class_count; ++index) {
        const char* label = gguf_get_arr_str(metadata, key, index);
        if (!label) return false;
        labels.emplace_back(label);
    }
    return true;
}

struct ModelConfig {
    uint32_t input_width = 0;
    uint32_t input_height = 0;
    uint32_t class_count = 0;
    std::vector<std::string> labels;
};

bool read_model_config(const gguf_context* metadata, ModelConfig& config,
                       std::string& error) {
    config = {};
    std::string architecture;
    std::string resize;
    std::string normalization;
    if (!read_string(metadata, "general.architecture", architecture) ||
        architecture != "yolo_v8_cls" ||
        !read_uint32(metadata, "classification.input.width", config.input_width) ||
        !read_uint32(metadata, "classification.input.height", config.input_height) ||
        !read_uint32(metadata, "classification.class_count", config.class_count) ||
        !read_string(metadata, "classification.input.resize", resize) ||
        resize != "shortest_center_crop" ||
        !read_string(metadata, "classification.input.normalization", normalization) ||
        normalization != "zero_to_one" ||
        !read_labels(metadata, config.class_count, config.labels) ||
        config.input_width == 0 || config.input_height == 0 ||
        config.input_width != config.input_height || config.class_count == 0) {
        error = "invalid YOLOv8 classification metadata";
        return false;
    }
    return true;
}

float source_channel(const classification_image& image, int x, int y, int channel) {
    x = std::clamp(x, 0, static_cast<int>(image.width) - 1);
    y = std::clamp(y, 0, static_cast<int>(image.height) - 1);
    const int source_channel_index = image.channels == 1 ? 0 : channel;
    return image.data[(static_cast<size_t>(y) * image.width + x) * image.channels +
                      source_channel_index] / 255.0f;
}

float bilinear_channel(const classification_image& image, float x, float y, int channel) {
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const float wx = x - x0;
    const float wy = y - y0;
    const float top = source_channel(image, x0, y0, channel) * (1.0f - wx) +
                      source_channel(image, x0 + 1, y0, channel) * wx;
    const float bottom = source_channel(image, x0, y0 + 1, channel) * (1.0f - wx) +
                         source_channel(image, x0 + 1, y0 + 1, channel) * wx;
    return top * (1.0f - wy) + bottom * wy;
}

bool preprocess(const classification_image& image, uint32_t target,
                std::vector<float>& pixels) {
    if (!image.data || image.width == 0 || image.height == 0 || target == 0 ||
        (image.channels != 1 && image.channels != 3 && image.channels != 4)) return false;
    uint32_t resized_width = target;
    uint32_t resized_height = target;
    if (image.width < image.height) {
        resized_height = static_cast<uint32_t>(
            static_cast<uint64_t>(target) * image.height / image.width);
    } else if (image.height < image.width) {
        resized_width = static_cast<uint32_t>(
            static_cast<uint64_t>(target) * image.width / image.height);
    }
    const uint32_t crop_x = (resized_width - target) / 2;
    const uint32_t crop_y = (resized_height - target) / 2;
    pixels.resize(static_cast<size_t>(target) * target * 3);
    for (uint32_t y = 0; y < target; ++y) {
        const float source_y = (crop_y + y + 0.5f) * image.height /
                               resized_height - 0.5f;
        for (uint32_t x = 0; x < target; ++x) {
            const float source_x = (crop_x + x + 0.5f) * image.width /
                                   resized_width - 0.5f;
            for (int channel = 0; channel < 3; ++channel) {
                pixels[x + static_cast<size_t>(target) *
                    (y + static_cast<size_t>(target) * channel)] =
                    bilinear_channel(image, source_x, source_y, channel);
            }
        }
    }
    return true;
}

class YoloModel;

class YoloSession final : public IClassificationSession {
public:
    explicit YoloSession(std::shared_ptr<YoloModel> model) : model_(std::move(model)) {}
    bool classify(const Request& request, Result& result) override;
private:
    std::shared_ptr<YoloModel> model_;
};

class YoloModel final : public IClassificationModel,
                        public std::enable_shared_from_this<YoloModel> {
public:
    YoloModel(ggml_backend_t backend, ModelConfig config,
              std::unique_ptr<model_v8::Model> network)
        : backend_(backend), config_(std::move(config)), network_(std::move(network)) {}
    ~YoloModel() override {
        network_.reset();
        if (backend_) ggml_backend_free(backend_);
        ggml_ops_ext::release_ops_hook();
    }

    std::unique_ptr<IClassificationSession> create_session() override {
        return std::make_unique<YoloSession>(shared_from_this());
    }
    classification_capabilities capabilities() const override {
        return {config_.class_count};
    }
    const char* label(int32_t class_id) const override {
        return class_id >= 0 && static_cast<size_t>(class_id) < config_.labels.size()
            ? config_.labels[static_cast<size_t>(class_id)].c_str() : nullptr;
    }

    bool run(const Request& request, Result& result) {
        std::vector<float> pixels;
        if (!request.image || !preprocess(*request.image, config_.input_width, pixels)) return false;
        try {
            nn::Context context(32 * 1024 * 1024, true);
            ggml_tensor* input = context.input<float>(
                "classification.input", {config_.input_width, config_.input_height, 3, 1},
                nn::data::borrow(pixels));
            ggml_tensor* logits = network_->forward(context, input, backend_);
            if (!logits || ggml_nelements(logits) != config_.class_count) return false;
            ggml_cgraph* graph = context.build(logits, 32768);
            nn::Executor executor(backend_);
            executor.prepare(context, graph);
            executor.compute(context, graph);
            std::vector<float> values(config_.class_count);
            context.read(logits, values.data(), values.size());
            const float maximum = *std::max_element(values.begin(), values.end());
            float denominator = 0.0f;
            for (float& value : values) {
                value = std::exp(value - maximum);
                denominator += value;
            }
            if (!(denominator > 0.0f)) return false;
            std::vector<uint32_t> order(config_.class_count);
            std::iota(order.begin(), order.end(), 0u);
            const size_t count = std::min<size_t>(request.top_k, order.size());
            std::partial_sort(order.begin(), order.begin() + count, order.end(),
                [&](uint32_t left, uint32_t right) { return values[left] > values[right]; });
            result.scores.clear();
            result.scores.reserve(count);
            for (size_t index = 0; index < count; ++index) {
                result.scores.push_back({static_cast<int32_t>(order[index]),
                                         values[order[index]] / denominator});
            }
            return true;
        } catch (const std::exception& exception) {
            std::cerr << "[YOLO Classify] inference failed: " << exception.what() << '\n';
            return false;
        }
    }

private:
    ggml_backend_t backend_ = nullptr;
    ModelConfig config_;
    std::unique_ptr<model_v8::Model> network_;
};

bool YoloSession::classify(const Request& request, Result& result) {
    return model_ && model_->run(request, result);
}

class YoloProvider final : public IClassificationProvider {
public:
    const char* name() const override { return "yolo"; }

    std::shared_ptr<IClassificationModel> load(
        const std::string& path, const RuntimeConfig& runtime) const override {
        initialize_backends();
        ggml_backend_dev_t device = select_device(runtime.device);
        if (!device) return nullptr;
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
                std::cerr << "[YOLO Classify] " << error << '\n';
                ggml_backend_free(backend);
                return nullptr;
            }
            model_v8::Config version_config = model_v8::Config::from_source(
                source, static_cast<int>(common_config.class_count), error);
            if (!error.empty()) {
                std::cerr << "[YOLO Classify] " << error << '\n';
                ggml_backend_free(backend);
                return nullptr;
            }
            auto network = std::make_unique<model_v8::Model>(version_config);
            nn::io::LoadResult loaded = nn::io::load_into(*network, source, backend);
            if (!loaded) {
                std::cerr << "[YOLO Classify] " << loaded.error << '\n';
                ggml_backend_free(backend);
                return nullptr;
            }
            network->to(backend);
            auto model = std::make_shared<YoloModel>(
                backend, std::move(common_config), std::move(network));
            hook.transfer_to_model();
            return model;
        } catch (const std::exception& exception) {
            std::cerr << "[YOLO Classify] model load failed: " << exception.what() << '\n';
            ggml_backend_free(backend);
            return nullptr;
        }
    }
};

[[maybe_unused]] const bool registered = [] {
    ProviderRegistry::get().register_architecture("yolo_v8_cls", [] {
        return std::make_unique<YoloProvider>();
    });
    return true;
}();

} // namespace
} // namespace visual_perception::classification::yolo
