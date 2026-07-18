#include "model_registry.h"
#include "protocols/chat.h"

#include <algorithm>
#include <cstring>

namespace forge::server {

ModelRegistry::~ModelRegistry() {
#if FORGE_SERVER_HAS_VISUAL
    visual_free_model(visual_model_);
    visual_runtime_free(visual_runtime_);
#endif
#if FORGE_SERVER_HAS_TTS
    tts_free_model(tts_model_);
    tts_runtime_free(tts_runtime_);
#endif
#if FORGE_SERVER_HAS_ASR
    asr_free_model(asr_model_);
    asr_runtime_free(asr_runtime_);
#endif
#if FORGE_SERVER_HAS_LLM
    llm_free_model(llm_model_);
    llm_runtime_free(llm_runtime_);
#endif
}

bool ModelRegistry::initialize(const ServerConfig& config, std::string& error) {
#if FORGE_SERVER_HAS_LLM
    if (!config.llm_model.empty()) {
        llm_runtime_params runtime = llm_runtime_default_params();
        runtime.n_threads = config.threads;
        runtime.n_gpu_layers = config.llm_gpu_layers;
        llm_runtime_ = llm_runtime_create(runtime);
        llm_model_params model = llm_model_default_params();
        model.model = config.llm_model.c_str();
        model.mmproj = config.llm_mmproj.empty() ? nullptr : config.llm_mmproj.c_str();
        llm_model_ = llm_runtime_ ? llm_load_model_with_params(llm_runtime_, &model) : nullptr;
        if (!llm_model_) { error = "failed to load LLM model"; return false; }
        served_models_.push_back({"llm", "llm", llm_model_get_provider(llm_model_)});
    }
#else
    if (!config.llm_model.empty()) { error = "server was built without LLM support"; return false; }
#endif

#if FORGE_SERVER_HAS_ASR
    if (!config.asr_model.empty()) {
        asr_runtime_params runtime = asr_runtime_default_params();
        runtime.device = config.device.c_str();
        runtime.n_threads = config.threads;
        asr_runtime_ = asr_runtime_create(runtime);
        asr_model_ = asr_runtime_ ? asr_load_model(asr_runtime_, config.asr_model.c_str()) : nullptr;
        if (!asr_model_) { error = "failed to load ASR model"; return false; }
        served_models_.push_back({"asr", "asr", asr_model_get_provider(asr_model_)});
    }
#else
    if (!config.asr_model.empty()) { error = "server was built without ASR support"; return false; }
#endif

#if FORGE_SERVER_HAS_TTS
    if (!config.tts_model.empty()) {
        tts_runtime_params runtime = tts_runtime_default_params();
        runtime.device = config.device.c_str();
        runtime.n_threads = config.threads;
        runtime.max_concurrency = config.max_concurrency;
        tts_runtime_ = tts_runtime_create(runtime);
        tts_model_ = tts_runtime_ ? tts_load_model(tts_runtime_, config.tts_model.c_str()) : nullptr;
        if (!tts_model_) { error = "failed to load TTS model"; return false; }
        served_models_.push_back({tts_model_get_name(tts_model_), "tts", tts_model_get_provider(tts_model_)});
    }
#else
    if (!config.tts_model.empty()) { error = "server was built without TTS support"; return false; }
#endif

#if FORGE_SERVER_HAS_VISUAL
    if (!config.visual_model.empty()) {
        visual_runtime_params runtime = visual_runtime_default_params();
        runtime.backend = config.device.c_str();
        runtime.n_threads = config.threads;
        visual_runtime_ = visual_runtime_create(runtime);
        visual_model_params model = visual_model_default_params();
        model.model = config.visual_model.c_str();
        visual_model_ = visual_runtime_ ? visual_load_model(visual_runtime_, &model) : nullptr;
        if (!visual_model_) { error = "failed to load visual model"; return false; }
        served_models_.push_back({"visual", "visual_generation", visual_model_get_provider(visual_model_)});
    }
#else
    if (!config.visual_model.empty()) { error = "server was built without visual generation support"; return false; }
#endif
    return !served_models_.empty();
}

std::vector<ServedModel> ModelRegistry::models() const { return served_models_; }

bool ModelRegistry::has_category(const std::string& category) const {
    return std::any_of(served_models_.begin(), served_models_.end(),
        [&](const ServedModel& model) { return model.category == category; });
}

#if FORGE_SERVER_HAS_LLM
llm_generation_params ModelRegistry::generation_defaults() const {
    return llm_generation_default_params();
}

bool ModelRegistry::generate(
    const std::string& prompt,
    llm_generation_params params,
    const std::function<bool(const char*, size_t)>& callback,
    std::string& error
) {
    if (!llm_model_) { error = "no LLM model is loaded"; return false; }
    std::lock_guard<std::mutex> lock(llm_mutex_);
    llm_session_ptr session = llm_create_session(llm_model_);
    if (!session) { error = "failed to create LLM session"; return false; }
    struct CallbackState { const std::function<bool(const char*, size_t)>* callback; } state{&callback};
    const bool ok = llm_generate(
        session, prompt.c_str(), params,
        [](const char* text, size_t length, void* opaque) {
            auto* current = static_cast<CallbackState*>(opaque);
            return (*current->callback)(text, length);
        }, &state);
    llm_free_session(session);
    if (!ok) error = "LLM generation failed";
    return ok;
}

bool ModelRegistry::generate_chat(
    const std::vector<ChatMessage>& messages,
    llm_generation_params params,
    const std::function<bool(const char*, size_t)>& callback,
    std::string& error
) {
    if (!llm_model_) { error = "no LLM model is loaded"; return false; }
    std::lock_guard<std::mutex> lock(llm_mutex_);
    llm_session_ptr session = llm_create_session(llm_model_);
    if (!session) { error = "failed to create LLM session"; return false; }
    struct CallbackState { const std::function<bool(const char*, size_t)>* callback; } state{&callback};
    const auto sink = [](const char* text, size_t length, void* opaque) {
        auto* current = static_cast<CallbackState*>(opaque);
        return (*current->callback)(text, length);
    };
    const bool has_media = std::any_of(messages.begin(), messages.end(), [](const ChatMessage& message) {
        return std::any_of(message.parts.begin(), message.parts.end(), [](const ChatMessage::Part& part) {
            return part.type != ChatMessage::Part::Type::Text;
        });
    });
    bool ok = false;
    if (has_media) {
        std::vector<std::vector<llm_content_part>> parts(messages.size());
        std::vector<llm_chat_content_message> native;
        native.reserve(messages.size());
        for (size_t i = 0; i < messages.size(); ++i) {
            const ChatMessage& message = messages[i];
            if (message.parts.empty()) {
                parts[i].push_back({LLM_CONTENT_TEXT, message.content.data(), message.content.size(), "text/plain"});
            } else {
                parts[i].reserve(message.parts.size());
                for (const ChatMessage::Part& part : message.parts) {
                    const llm_content_type type = part.type == ChatMessage::Part::Type::Text ? LLM_CONTENT_TEXT :
                        (part.type == ChatMessage::Part::Type::Image ? LLM_CONTENT_IMAGE : LLM_CONTENT_AUDIO);
                    const void* data = part.type == ChatMessage::Part::Type::Text ?
                        static_cast<const void*>(part.text.data()) : static_cast<const void*>(part.data.data());
                    const size_t size = part.type == ChatMessage::Part::Type::Text ? part.text.size() : part.data.size();
                    parts[i].push_back({type, data, size, part.mime_type.c_str()});
                }
            }
            native.push_back({message.role.c_str(), parts[i].data(), parts[i].size()});
        }
        ok = llm_generate_chat_content(session, native.data(), native.size(), params, sink, &state);
    } else {
        std::vector<llm_chat_message> native;
        native.reserve(messages.size());
        for (const ChatMessage& message : messages) native.push_back({message.role.c_str(), message.content.c_str()});
        ok = llm_generate_chat(session, native.data(), native.size(), params, sink, &state);
    }
    llm_free_session(session);
    if (!ok) error = "LLM chat generation failed";
    return ok;
}
#endif

#if FORGE_SERVER_HAS_ASR
bool ModelRegistry::transcribe(
    const std::vector<float>& audio,
    int32_t sample_rate,
    asr_request_params params,
    const std::function<bool(const asr_event&)>& callback,
    std::string& error
) {
    if (!asr_model_) { error = "no ASR model is loaded"; return false; }
    std::lock_guard<std::mutex> lock(asr_mutex_);
    asr_session_ptr session = asr_create_session(asr_model_);
    if (!session) { error = "failed to create ASR session"; return false; }
    struct CallbackState { const std::function<bool(const asr_event&)>* callback; } state{&callback};
    const bool ok = asr_transcribe(
        session, audio.data(), audio.size(), sample_rate, params,
        [](const asr_event* event, void* opaque) {
            if (!event) return true;
            auto* current = static_cast<CallbackState*>(opaque);
            return (*current->callback)(*event);
        }, &state);
    asr_free_session(session);
    if (!ok) error = "ASR transcription failed";
    return ok;
}
#endif

#if FORGE_SERVER_HAS_TTS
bool ModelRegistry::synthesize(
    const std::string& text,
    const std::string& language,
    float speed,
    const SpeechReference* reference,
    std::vector<float>& audio,
    int32_t& sample_rate,
    std::string& error
) {
    if (!tts_model_) { error = "no TTS model is loaded"; return false; }
    std::lock_guard<std::mutex> lock(tts_mutex_);
    tts_session_ptr session = tts_create_session(tts_model_);
    if (!session) { error = "failed to create TTS session"; return false; }
    if (reference && !tts_session_set_reference(
            session, reference->audio.data(), reference->audio.size(), reference->sample_rate,
            reference->text.c_str(), reference->language.c_str())) {
        tts_free_session(session);
        error = "failed to attach TTS voice reference";
        return false;
    }
    int32_t count = 0;
    const float* samples = tts_synthesize(
        session, text.c_str(), language.c_str(), speed, &count);
    sample_rate = tts_session_get_output_sample_rate(session);
    if (samples && count > 0 && sample_rate > 0) audio.assign(samples, samples + count);
    tts_free_session(session);
    if (audio.empty()) error = "TTS synthesis failed";
    return !audio.empty();
}
#endif

#if FORGE_SERVER_HAS_VISUAL
visual_capabilities ModelRegistry::get_visual_capabilities() const {
    return visual_model_get_capabilities(visual_model_);
}

bool ModelRegistry::generate_image(
    const visual_image_request& request,
    std::vector<uint8_t>& pixels,
    uint32_t& width,
    uint32_t& height,
    uint32_t& channels,
    std::string& error
) {
    if (!visual_model_) { error = "no visual model is loaded"; return false; }
    std::lock_guard<std::mutex> lock(visual_mutex_);
    visual_session_ptr session = visual_create_session(visual_model_);
    if (!session) { error = "failed to create visual session"; return false; }
    visual_image* images = nullptr;
    size_t count = 0;
    const bool ok = visual_generate_images(session, &request, &images, &count);
    if (ok && count > 0 && images[0].data) {
        width = images[0].width;
        height = images[0].height;
        channels = images[0].channels;
        pixels.assign(images[0].data, images[0].data +
            static_cast<size_t>(width) * height * channels);
    }
    visual_free_images(images, count);
    visual_free_session(session);
    if (pixels.empty()) error = "image generation failed";
    return !pixels.empty();
}

bool ModelRegistry::generate_video(
    const visual_video_request& request,
    std::vector<VideoFrame>& frames,
    uint32_t& fps,
    std::string& error
) {
    if (!visual_model_) { error = "no visual model is loaded"; return false; }
    std::lock_guard<std::mutex> lock(visual_mutex_);
    visual_session_ptr session = visual_create_session(visual_model_);
    if (!session) { error = "failed to create visual session"; return false; }
    visual_video video{};
    const bool ok = visual_generate_video(session, &request, &video);
    if (ok && video.frames && video.frame_count > 0) {
        fps = video.fps;
        frames.reserve(video.frame_count);
        for (size_t i = 0; i < video.frame_count; ++i) {
            const visual_image& source = video.frames[i];
            VideoFrame frame;
            frame.width = source.width;
            frame.height = source.height;
            frame.channels = source.channels;
            if (source.data) frame.pixels.assign(
                source.data, source.data + static_cast<size_t>(source.width) * source.height * source.channels);
            frames.push_back(std::move(frame));
        }
    }
    visual_free_video(&video);
    visual_free_session(session);
    if (frames.empty()) error = "video generation failed";
    return !frames.empty();
}
#endif

} // namespace forge::server
