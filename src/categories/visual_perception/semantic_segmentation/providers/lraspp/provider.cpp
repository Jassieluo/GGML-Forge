#include "providers/segmentation_provider.h"
#include "providers/lraspp/model.h"
#include "image/preprocess.h"

#include "ggml-backend.h"
#include "gguf.h"
#include "nn/core/executor.h"
#include "nn/io/gguf.h"
#include "nn/io/load.h"
#include "ops/cpu.h"
#include "ops/ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
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

namespace visual_perception::segmentation::lraspp_provider {
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
    ~OpsHookGuard() { if (active_) ggml_ops_ext::release_ops_hook(); }
    void transfer_to_model() { active_ = false; }
private:
    bool active_ = true;
};

ggml_backend_dev_t select_device(const std::string& requested) {
    if (requested.empty() || requested == "auto") {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t device = ggml_backend_dev_get(i);
            if (!device) continue;
            const auto type = ggml_backend_dev_type(device);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU)
                return device;
        }
    }
    if (requested.empty() || requested == "auto" || requested == "cpu") {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t device = ggml_backend_dev_get(i);
            if (device && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) return device;
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

struct ModelConfig {
    uint32_t class_count = 0;
    uint32_t shortest_edge = 0;
    std::vector<std::string> labels;
    std::vector<uint8_t> palette;
};

bool read_config(const gguf_context* metadata, ModelConfig& config) {
    std::string architecture, normalization, resize;
    if (!read_string(metadata, "general.architecture", architecture) ||
        architecture != "lraspp_mobilenet_v3_large" ||
        !read_uint32(metadata, "segmentation.class_count", config.class_count) ||
        !read_uint32(metadata, "segmentation.input.shortest_edge", config.shortest_edge) ||
        !read_string(metadata, "segmentation.input.normalization", normalization) ||
        normalization != "imagenet" ||
        !read_string(metadata, "segmentation.input.resize", resize) ||
        resize != "shortest_edge" || config.class_count != 21 || config.shortest_edge == 0) return false;
    const int64_t labels_key = gguf_find_key(metadata, "segmentation.labels");
    if (labels_key < 0 || gguf_get_kv_type(metadata, labels_key) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(metadata, labels_key) != GGUF_TYPE_STRING ||
        gguf_get_arr_n(metadata, labels_key) != config.class_count) return false;
    for (uint32_t i = 0; i < config.class_count; ++i) {
        const char* label = gguf_get_arr_str(metadata, labels_key, i);
        if (!label) return false;
        config.labels.emplace_back(label);
    }
    const int64_t palette_key = gguf_find_key(metadata, "segmentation.palette");
    if (palette_key >= 0 && gguf_get_kv_type(metadata, palette_key) == GGUF_TYPE_ARRAY &&
        gguf_get_arr_n(metadata, palette_key) == static_cast<int64_t>(config.class_count) * 3) {
        const void* data = gguf_get_arr_data(metadata, palette_key);
        const size_t count = static_cast<size_t>(config.class_count) * 3;
        if (data) {
            config.palette.resize(count);
            const gguf_type type = gguf_get_arr_type(metadata, palette_key);
            for (size_t i = 0; i < count; ++i) {
                uint32_t value = type == GGUF_TYPE_UINT8 ? static_cast<const uint8_t*>(data)[i] :
                    type == GGUF_TYPE_UINT16 ? static_cast<const uint16_t*>(data)[i] :
                    type == GGUF_TYPE_UINT32 ? static_cast<const uint32_t*>(data)[i] : 256u;
                if (value > 255) { config.palette.clear(); break; }
                config.palette[i] = static_cast<uint8_t>(value);
            }
        }
    }
    return true;
}

class Model;
class Session final : public ::segmentation::ISegmentationSession {
public:
    explicit Session(std::shared_ptr<Model> model) : model_(std::move(model)) {}
    bool segment(const ::segmentation::Request& request, ::segmentation::Result& result) override;
private:
    std::shared_ptr<Model> model_;
};

class Model final : public ::segmentation::ISegmentationModel,
                    public std::enable_shared_from_this<Model> {
public:
    Model(ggml_backend_t backend, ModelConfig config,
          std::unique_ptr<visual_perception::segmentation::lraspp::Model> network)
        : backend_(backend), config_(std::move(config)), network_(std::move(network)) {}
    ~Model() override {
        network_.reset();
        if (backend_) ggml_backend_free(backend_);
        ggml_ops_ext::release_ops_hook();
    }
    std::unique_ptr<::segmentation::ISegmentationSession> create_session() override {
        return std::make_unique<Session>(shared_from_this());
    }
    segmentation_capabilities capabilities() const override { return {config_.class_count, true}; }
    const char* label(int32_t id) const override {
        return id >= 0 && static_cast<size_t>(id) < config_.labels.size()
            ? config_.labels[static_cast<size_t>(id)].c_str() : nullptr;
    }
    bool color(int32_t id, uint8_t rgb[3]) const override {
        if (!rgb || id < 0 || static_cast<size_t>(id) * 3 + 2 >= config_.palette.size()) return false;
        std::memcpy(rgb, config_.palette.data() + static_cast<size_t>(id) * 3, 3);
        return true;
    }
    bool run(const ::segmentation::Request& request, ::segmentation::Result& result) {
        static constexpr float mean[3] = {0.485f, 0.456f, 0.406f};
        static constexpr float stddev[3] = {0.229f, 0.224f, 0.225f};
        uint32_t input_width = 0, input_height = 0;
        std::vector<float> pixels;
        if (!request.image || !forge::media::resize_shortest_normalized_rgb8(
                request.image->data, request.image->width, request.image->height,
                request.image->channels, config_.shortest_edge, mean, stddev,
                input_width, input_height, pixels)) return false;
        try {
            nn::Context context(64 * 1024 * 1024, true);
            ggml_tensor* input = context.input<float>(
                "segmentation.input", {input_width, input_height, 3, 1}, nn::data::borrow(pixels));
            ggml_tensor* logits = network_->forward(context, input, backend_);
            if (!logits || logits->ne[0] != input_width || logits->ne[1] != input_height ||
                logits->ne[2] != config_.class_count) return false;
            ggml_cgraph* graph = context.build(logits, 65536);
            nn::Executor executor(backend_);
            executor.prepare(context, graph);
            executor.compute(context, graph);
            std::vector<float> values(static_cast<size_t>(input_width) * input_height * config_.class_count);
            context.read(logits, values.data(), values.size());
            const uint32_t output_width = request.image->width;
            const uint32_t output_height = request.image->height;
            result.width = output_width;
            result.height = output_height;
            result.class_map.resize(static_cast<size_t>(output_width) * output_height);
            if (request.want_confidence) result.confidence.resize(result.class_map.size());
            std::vector<float> sampled(config_.class_count);
            for (uint32_t y = 0; y < output_height; ++y) {
                const double fy = std::clamp((y + 0.5) * static_cast<double>(input_height) / output_height - 0.5,
                                             0.0, static_cast<double>(input_height - 1));
                const uint32_t y0 = static_cast<uint32_t>(fy), y1 = std::min(y0 + 1, input_height - 1);
                const float wy = static_cast<float>(fy - y0);
                for (uint32_t x = 0; x < output_width; ++x) {
                    const double fx = std::clamp((x + 0.5) * static_cast<double>(input_width) / output_width - 0.5,
                                                 0.0, static_cast<double>(input_width - 1));
                    const uint32_t x0 = static_cast<uint32_t>(fx), x1 = std::min(x0 + 1, input_width - 1);
                    const float wx = static_cast<float>(fx - x0);
                    float best = -INFINITY;
                    float denominator = 0.0f;
                    int32_t best_class = 0;
                    for (uint32_t c = 0; c < config_.class_count; ++c) {
                        auto at = [&](uint32_t sx, uint32_t sy) {
                            return values[sx + static_cast<size_t>(input_width) *
                                (sy + static_cast<size_t>(input_height) * c)];
                        };
                        const float top = at(x0, y0) + wx * (at(x1, y0) - at(x0, y0));
                        const float bottom = at(x0, y1) + wx * (at(x1, y1) - at(x0, y1));
                        sampled[c] = top + wy * (bottom - top);
                        if (sampled[c] > best) { best = sampled[c]; best_class = static_cast<int32_t>(c); }
                    }
                    const size_t out = static_cast<size_t>(y) * output_width + x;
                    result.class_map[out] = best_class;
                    if (request.want_confidence) {
                        for (float value : sampled) denominator += std::exp(value - best);
                        result.confidence[out] = 1.0f / denominator;
                    }
                }
            }
            return true;
        } catch (const std::exception& exception) {
            std::cerr << "[LR-ASPP] inference failed: " << exception.what() << '\n';
            return false;
        }
    }
private:
    ggml_backend_t backend_ = nullptr;
    ModelConfig config_;
    std::unique_ptr<visual_perception::segmentation::lraspp::Model> network_;
};

bool Session::segment(const ::segmentation::Request& request, ::segmentation::Result& result) {
    return model_ && model_->run(request, result);
}

class Provider final : public ::segmentation::ISegmentationProvider {
public:
    const char* name() const override { return "torchvision_lraspp"; }
    std::shared_ptr<::segmentation::ISegmentationModel> load(
        const std::string& path, const ::segmentation::RuntimeConfig& runtime) const override {
        initialize_backends();
        ggml_backend_dev_t device = select_device(runtime.device);
        if (!device) return nullptr;
        ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
        if (!backend) return nullptr;
        OpsHookGuard hook;
        if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            if (ggml_backend_reg_t registration = ggml_backend_dev_backend_reg(device)) {
                using SetThreads = void (*)(ggml_backend_t, int);
                auto set_threads = reinterpret_cast<SetThreads>(ggml_backend_reg_get_proc_address(
                    registration, "ggml_backend_cpu_set_n_threads"));
                if (set_threads) set_threads(backend, static_cast<int>(runtime.n_threads));
            }
            ggml_ops_ext_cpu_set_n_threads(backend, static_cast<int>(runtime.n_threads));
        }
        try {
            nn::io::GGUFSource source(path);
            ModelConfig config;
            if (!read_config(source.metadata_context(), config)) {
                std::cerr << "[LR-ASPP] invalid metadata\n";
                ggml_backend_free(backend);
                return nullptr;
            }
            auto network = std::make_unique<visual_perception::segmentation::lraspp::Model>(
                static_cast<int>(config.class_count));
            nn::io::LoadResult loaded = nn::io::load_into(*network, source, backend);
            if (!loaded) {
                std::cerr << "[LR-ASPP] " << loaded.error << '\n';
                ggml_backend_free(backend);
                return nullptr;
            }
            network->to(backend);
            auto model = std::make_shared<Model>(backend, std::move(config), std::move(network));
            hook.transfer_to_model();
            return model;
        } catch (const std::exception& exception) {
            std::cerr << "[LR-ASPP] model load failed: " << exception.what() << '\n';
            ggml_backend_free(backend);
            return nullptr;
        }
    }
};

[[maybe_unused]] const bool registered = [] {
    ::segmentation::ProviderRegistry::get().register_architecture(
        "lraspp_mobilenet_v3_large", [] { return std::make_unique<Provider>(); });
    return true;
}();

} // namespace
} // namespace visual_perception::segmentation::lraspp_provider
