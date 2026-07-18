#pragma once

#include "categories/visual_generation/visual_generation.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace visual {

struct RuntimeConfig {
    std::string backend;
    std::string params_backend;
    std::string max_vram;
    uint32_t n_threads = 1;
    bool stream_layers = false;
    bool eager_load = false;
    bool flash_attention = true;
};

struct ModelConfig {
    std::string model;
    std::string diffusion_model;
    std::string high_noise_diffusion_model;
    std::string unconditioned_diffusion_model;
    std::string clip_l;
    std::string clip_g;
    std::string clip_vision;
    std::string t5xxl;
    std::string llm;
    std::string llm_vision;
    std::string vae;
    std::string audio_vae;
    std::string taesd;
    std::string control_net;
    std::string motion_module;
    std::string photo_maker;
    std::string pulid;
    std::string upscaler;
    std::string adetailer;
    std::string weight_type;
};

struct ImageBuffer {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t channels = 0;
    std::vector<uint8_t> data;
};

struct AudioBuffer {
    uint32_t sample_rate = 0;
    uint32_t channels = 0;
    std::vector<float> data;
};

struct VideoBuffer {
    uint32_t fps = 0;
    std::vector<ImageBuffer> frames;
    AudioBuffer audio;
};

struct Callbacks {
    visual_progress_callback progress = nullptr;
    visual_preview_callback preview = nullptr;
    void* user_data = nullptr;
};

class IVisualSession {
public:
    virtual ~IVisualSession() = default;
    virtual void set_callbacks(Callbacks callbacks) = 0;
    virtual bool cancel(visual_cancel_mode mode) = 0;
    virtual bool generate_images(const visual_image_request& request, std::vector<ImageBuffer>& output) = 0;
    virtual bool generate_video(const visual_video_request& request, VideoBuffer& output) = 0;
    virtual bool upscale(const visual_image& input, uint32_t factor, std::vector<ImageBuffer>& output) = 0;
    virtual bool adetail(
        const visual_image& input,
        const char* prompt,
        const char* negative_prompt,
        const visual_image_request& request,
        std::vector<ImageBuffer>& output) = 0;
};

class IVisualModel {
public:
    virtual ~IVisualModel() = default;
    virtual std::unique_ptr<IVisualSession> create_session() = 0;
    virtual visual_capabilities capabilities() const = 0;
};

class IVisualProvider {
public:
    virtual ~IVisualProvider() = default;
    virtual const char* name() const = 0;
    virtual std::shared_ptr<IVisualModel> load(const ModelConfig& model, const RuntimeConfig& runtime) const = 0;
};

class ProviderRegistry {
public:
    using Creator = std::function<std::unique_ptr<IVisualProvider>()>;

    static ProviderRegistry& get() {
        static ProviderRegistry registry;
        return registry;
    }

    void register_provider(std::string name, Creator creator) {
        creators_[std::move(name)] = std::move(creator);
    }

    std::unique_ptr<IVisualProvider> create(const std::string& name) const {
        const auto it = creators_.find(name);
        return it == creators_.end() ? nullptr : it->second();
    }

private:
    std::unordered_map<std::string, Creator> creators_;
};

bool preprocess_canny(visual_image& image, float high, float low, bool inverse);

} // namespace visual
