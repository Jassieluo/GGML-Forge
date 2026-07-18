#pragma once

#include "server_config.h"

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#if FORGE_SERVER_HAS_LLM
#include "categories/llm/llm.h"
#endif
#if FORGE_SERVER_HAS_ASR
#include "categories/asr/asr.h"
#endif
#if FORGE_SERVER_HAS_TTS
#include "categories/tts/tts.h"
#endif
#if FORGE_SERVER_HAS_VISUAL
#include "categories/visual_generation/visual_generation.h"
#endif

namespace forge::server {

struct ChatMessage;

struct ServedModel {
    std::string id;
    std::string category;
    std::string provider;
};

struct SpeechReference {
    std::vector<float> audio;
    int32_t sample_rate = 0;
    std::string text;
    std::string language = "auto";
};

class ModelRegistry {
public:
#if FORGE_SERVER_HAS_VISUAL
    struct VideoFrame {
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t channels = 0;
        std::vector<uint8_t> pixels;
    };
#endif

    ModelRegistry() = default;
    ~ModelRegistry();
    ModelRegistry(const ModelRegistry&) = delete;
    ModelRegistry& operator=(const ModelRegistry&) = delete;

    bool initialize(const ServerConfig& config, std::string& error);
    std::vector<ServedModel> models() const;
    bool has_category(const std::string& category) const;

#if FORGE_SERVER_HAS_LLM
    bool generate(
        const std::string& prompt,
        llm_generation_params params,
        const std::function<bool(const char*, size_t)>& callback,
        std::string& error);
    bool generate_chat(
        const std::vector<ChatMessage>& messages,
        llm_generation_params params,
        const std::function<bool(const char*, size_t)>& callback,
        std::string& error);
    llm_generation_params generation_defaults() const;
#endif

#if FORGE_SERVER_HAS_ASR
    bool transcribe(
        const std::vector<float>& audio,
        int32_t sample_rate,
        asr_request_params params,
        const std::function<bool(const asr_event&)>& callback,
        std::string& error);
#endif

#if FORGE_SERVER_HAS_TTS
    bool synthesize(
        const std::string& text,
        const std::string& language,
        float speed,
        const SpeechReference* reference,
        std::vector<float>& audio,
        int32_t& sample_rate,
        std::string& error);
#endif

#if FORGE_SERVER_HAS_VISUAL
    visual_capabilities get_visual_capabilities() const;
    bool generate_image(
        const visual_image_request& request,
        std::vector<uint8_t>& pixels,
        uint32_t& width,
        uint32_t& height,
        uint32_t& channels,
        std::string& error);
    bool generate_video(
        const visual_video_request& request,
        std::vector<VideoFrame>& frames,
        uint32_t& fps,
        std::string& error);
#endif

private:
    std::vector<ServedModel> served_models_;

#if FORGE_SERVER_HAS_LLM
    llm_runtime_ptr llm_runtime_ = nullptr;
    llm_model_ptr llm_model_ = nullptr;
    mutable std::mutex llm_mutex_;
#endif
#if FORGE_SERVER_HAS_ASR
    asr_runtime_ptr asr_runtime_ = nullptr;
    asr_model_ptr asr_model_ = nullptr;
    mutable std::mutex asr_mutex_;
#endif
#if FORGE_SERVER_HAS_TTS
    tts_runtime_ptr tts_runtime_ = nullptr;
    tts_model_ptr tts_model_ = nullptr;
    mutable std::mutex tts_mutex_;
#endif
#if FORGE_SERVER_HAS_VISUAL
    visual_runtime_ptr visual_runtime_ = nullptr;
    visual_model_ptr visual_model_ = nullptr;
    mutable std::mutex visual_mutex_;
#endif
};

} // namespace forge::server
