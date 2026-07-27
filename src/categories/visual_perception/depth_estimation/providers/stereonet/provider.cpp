#include "providers/depth_provider.h"
#include "providers/stereonet/model.h"
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

namespace visual_perception::depth::stereonet_provider {
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

struct Config { uint32_t max_side = 625; };

bool read_config(const gguf_context* metadata, Config& config) {
    std::string architecture, normalization, kind;
    return read_string(metadata, "general.architecture", architecture) && architecture == "stereonet" &&
        read_uint32(metadata, "depth.input.max_side", config.max_side) && config.max_side >= 256 &&
        read_string(metadata, "depth.input.normalization", normalization) &&
        normalization == "keystone_gray" && read_string(metadata, "depth.output.kind", kind) &&
        kind == "disparity";
}

bool preprocess(const depth_image& image, uint32_t width, uint32_t height,
                float mean, float standard_deviation, std::vector<float>& gray) {
    std::vector<float> rgb;
    if (!forge::media::resize_rgb8_planar(image.data, image.width, image.height, image.channels,
                                          width, height, rgb)) return false;
    const size_t plane = static_cast<size_t>(width) * height;
    gray.resize(plane);
    for (size_t i = 0; i < plane; ++i) {
        const float luminance = 255.0f * (0.299f * rgb[i] + 0.587f * rgb[plane + i] +
                                          0.114f * rgb[2 * plane + i]);
        gray[i] = (luminance - mean) / standard_deviation;
    }
    return true;
}

std::vector<float> resize_disparity(const std::vector<float>& source, uint32_t source_width,
                                    uint32_t source_height, uint32_t width, uint32_t height) {
    std::vector<float> output(static_cast<size_t>(width) * height);
    const float disparity_scale = static_cast<float>(width) / source_width;
    for (uint32_t y = 0; y < height; ++y) {
        const double fy = height == 1 ? 0.0 : y * static_cast<double>(source_height - 1) / (height - 1);
        const uint32_t y0 = static_cast<uint32_t>(fy), y1 = std::min(y0 + 1, source_height - 1);
        const float wy = static_cast<float>(fy - y0);
        for (uint32_t x = 0; x < width; ++x) {
            const double fx = width == 1 ? 0.0 : x * static_cast<double>(source_width - 1) / (width - 1);
            const uint32_t x0 = static_cast<uint32_t>(fx), x1 = std::min(x0 + 1, source_width - 1);
            const float wx = static_cast<float>(fx - x0);
            auto at = [&](uint32_t sx, uint32_t sy) { return source[static_cast<size_t>(sy) * source_width + sx]; };
            const float top = at(x0, y0) + wx * (at(x1, y0) - at(x0, y0));
            const float bottom = at(x0, y1) + wx * (at(x1, y1) - at(x0, y1));
            output[static_cast<size_t>(y) * width + x] =
                std::max(0.0f, (top + wy * (bottom - top)) * disparity_scale);
        }
    }
    return output;
}

class Model;
class Session final : public ::depth::IDepthSession {
public:
    explicit Session(std::shared_ptr<Model> model) : model_(std::move(model)) {}
    bool estimate(const ::depth::Request& request, ::depth::Result& result) override;
private:
    std::shared_ptr<Model> model_;
};

class Model final : public ::depth::IDepthModel, public std::enable_shared_from_this<Model> {
public:
    Model(ggml_backend_t backend, Config config,
          std::unique_ptr<visual_perception::depth::stereonet::Model> network)
        : backend_(backend), config_(config), network_(std::move(network)) {}
    ~Model() override {
        network_.reset();
        if (backend_) ggml_backend_free(backend_);
        ggml_ops_ext::release_ops_hook();
    }
    std::unique_ptr<::depth::IDepthSession> create_session() override {
        return std::make_unique<Session>(shared_from_this());
    }
    depth_capabilities capabilities() const override { return {false, true, false}; }
    bool run(const ::depth::Request& request, ::depth::Result& result) {
        if (!request.image || !request.right || request.task != DEPTH_TASK_STEREO ||
            request.image->width != request.right->width ||
            request.image->height != request.right->height) return false;
        const uint32_t original_width = request.image->width;
        const uint32_t original_height = request.image->height;
        const double scale = std::min(1.0, static_cast<double>(config_.max_side) /
                                           std::max(original_width, original_height));
        const uint32_t width = std::max(1u, static_cast<uint32_t>(original_width * scale));
        const uint32_t height = std::max(1u, static_cast<uint32_t>(original_height * scale));
        if ((width + 7) / 8 < 32 || height < 8) {
            std::cerr << "[StereoNet] input is too narrow for 256 candidate disparities\n";
            return false;
        }
        std::vector<float> left, right;
        if (!preprocess(*request.image, width, height, 111.5684f, 61.9625f, left) ||
            !preprocess(*request.right, width, height, 113.6528f, 62.0313f, right)) return false;
        try {
            nn::Context context(64 * 1024 * 1024, true);
            ggml_tensor* left_tensor = context.input<float>(
                "depth.left", {width, height, 1, 1}, nn::data::borrow(left));
            ggml_tensor* right_tensor = context.input<float>(
                "depth.right", {width, height, 1, 1}, nn::data::borrow(right));
            ggml_tensor* output = network_->forward(context, left_tensor, right_tensor, backend_);
            if (!output || output->ne[0] != width || output->ne[1] != height || output->ne[2] != 1)
                return false;
            ggml_cgraph* graph = context.build(output, 65536);
            nn::Executor executor(backend_);
            executor.prepare(context, graph);
            executor.compute(context, graph);
            std::vector<float> values(static_cast<size_t>(width) * height);
            context.read(output, values.data(), values.size());
            result.width = original_width;
            result.height = original_height;
            result.kind = DEPTH_MAP_DISPARITY;
            result.values = resize_disparity(values, width, height, original_width, original_height);
            return true;
        } catch (const std::exception& exception) {
            std::cerr << "[StereoNet] inference failed: " << exception.what() << '\n';
            return false;
        }
    }
private:
    ggml_backend_t backend_ = nullptr;
    Config config_;
    std::unique_ptr<visual_perception::depth::stereonet::Model> network_;
};

bool Session::estimate(const ::depth::Request& request, ::depth::Result& result) {
    return model_ && model_->run(request, result);
}

class Provider final : public ::depth::IDepthProvider {
public:
    const char* name() const override { return "stereonet"; }
    std::shared_ptr<::depth::IDepthModel> load(
        const std::string& path, const ::depth::RuntimeConfig& runtime) const override {
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
            Config config;
            if (!read_config(source.metadata_context(), config)) {
                std::cerr << "[StereoNet] invalid metadata\n";
                ggml_backend_free(backend); return nullptr;
            }
            auto network = std::make_unique<visual_perception::depth::stereonet::Model>();
            nn::io::LoadResult loaded = nn::io::load_into(*network, source, backend);
            if (!loaded) {
                std::cerr << "[StereoNet] " << loaded.error << '\n';
                ggml_backend_free(backend); return nullptr;
            }
            network->to(backend);
            auto model = std::make_shared<Model>(backend, config, std::move(network));
            hook.transfer_to_model();
            return model;
        } catch (const std::exception& exception) {
            std::cerr << "[StereoNet] model load failed: " << exception.what() << '\n';
            ggml_backend_free(backend); return nullptr;
        }
    }
};

[[maybe_unused]] const bool registered = [] {
    ::depth::ProviderRegistry::get().register_architecture(
        "stereonet", [] { return std::make_unique<Provider>(); });
    return true;
}();

} // namespace
} // namespace visual_perception::depth::stereonet_provider
