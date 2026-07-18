#include "providers/asr_provider.h"

#include "ggml-backend.h"
#include "whisper.h"

#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>

namespace asr {
namespace {

bool configure_device(const std::string& requested, whisper_context_params& params) {
    ggml_backend_load_all();
    if (requested.empty() || requested == "auto") return true;
    if (requested == "cpu") {
        params.use_gpu = false;
        return true;
    }

    int gpu_ordinal = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (!device) continue;
        const char* name = ggml_backend_dev_name(device);
        const auto type = ggml_backend_dev_type(device);
        if (name && requested == name) {
            if (type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                params.use_gpu = false;
                return true;
            }
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                params.use_gpu = true;
                params.gpu_device = gpu_ordinal;
                return true;
            }
            return false;
        }
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            ++gpu_ordinal;
        }
    }
    return false;
}

class WhisperModel;

class WhisperSession final : public IASRSession {
public:
    explicit WhisperSession(std::shared_ptr<WhisperModel> model);
    ~WhisperSession() override;

    bool ready() const { return state_ != nullptr; }
    bool reset() override;
    bool transcribe(const Request& request, EventSink sink) override;

private:
    std::shared_ptr<WhisperModel> model_;
    whisper_state* state_ = nullptr;
};

class WhisperModel final : public IASRModel, public std::enable_shared_from_this<WhisperModel> {
public:
    WhisperModel(whisper_context* context, RuntimeConfig config)
        : context_(context), config_(std::move(config)) {}
    ~WhisperModel() override { if (context_) whisper_free(context_); }

    std::unique_ptr<IASRSession> create_session() override {
        auto session = std::make_unique<WhisperSession>(shared_from_this());
        return session->ready() ? std::move(session) : nullptr;
    }

    asr_capabilities capabilities() const override {
        const bool multilingual = whisper_is_multilingual(context_) != 0;
        return {false, multilingual, multilingual, true, true};
    }

    whisper_context* context() const { return context_; }
    const RuntimeConfig& config() const { return config_; }
    std::mutex& mutex() { return mutex_; }

private:
    whisper_context* context_ = nullptr;
    RuntimeConfig config_;
    std::mutex mutex_;
};

WhisperSession::WhisperSession(std::shared_ptr<WhisperModel> model) : model_(std::move(model)) {
    std::lock_guard<std::mutex> lock(model_->mutex());
    state_ = whisper_init_state(model_->context());
}

WhisperSession::~WhisperSession() {
    if (!state_) return;
    std::lock_guard<std::mutex> lock(model_->mutex());
    whisper_free_state(state_);
}

bool WhisperSession::reset() {
    std::lock_guard<std::mutex> lock(model_->mutex());
    if (state_) whisper_free_state(state_);
    state_ = whisper_init_state(model_->context());
    return state_ != nullptr;
}

bool WhisperSession::transcribe(const Request& request, EventSink sink) {
    if (!state_ || !request.audio || request.sample_count == 0 || !sink) return false;
    std::lock_guard<std::mutex> lock(model_->mutex());
    const bool multilingual = whisper_is_multilingual(model_->context()) != 0;
    if (!multilingual && request.task == ASR_TASK_TRANSLATE) return false;
    whisper_free_state(state_);
    state_ = whisper_init_state(model_->context());
    if (!state_) return false;

    auto params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.n_threads = static_cast<int>(model_->config().n_threads);
    params.translate = request.task == ASR_TASK_TRANSLATE;
    params.language = !multilingual
        ? "en"
        : (request.language.empty() || request.language == "auto" ? "auto" : request.language.c_str());
    params.detect_language = false;
    params.no_context = true;
    params.token_timestamps = request.token_timestamps;
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.print_special = false;

    if (request.sample_count > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
    if (whisper_full_with_state(
            model_->context(), state_, params, request.audio,
            static_cast<int>(request.sample_count)) != 0) return false;

    const int language_id = whisper_full_lang_id_from_state(state_);
    const char* language = language_id >= 0 ? whisper_lang_str(language_id) : nullptr;
    if (language) {
        const asr_event event = {
            ASR_EVENT_LANGUAGE, nullptr, 0, language, -1, -1, -1.0f
        };
        if (!sink(event)) return true;
    }

    const int segment_count = whisper_full_n_segments_from_state(state_);
    const whisper_token eot = whisper_token_eot(model_->context());
    for (int segment_index = 0; segment_index < segment_count; ++segment_index) {
        const char* text = whisper_full_get_segment_text_from_state(state_, segment_index);
        const asr_event segment = {
            ASR_EVENT_SEGMENT,
            text,
            text ? std::strlen(text) : 0,
            language,
            whisper_full_get_segment_t0_from_state(state_, segment_index) * 10,
            whisper_full_get_segment_t1_from_state(state_, segment_index) * 10,
            -1.0f,
        };
        if (!sink(segment)) return true;
        if (!request.token_timestamps) continue;

        const int token_count = whisper_full_n_tokens_from_state(state_, segment_index);
        for (int token_index = 0; token_index < token_count; ++token_index) {
            const whisper_token token_id = whisper_full_get_token_id_from_state(
                state_, segment_index, token_index);
            if (token_id >= eot) continue;
            const char* token_text = whisper_full_get_token_text_from_state(
                model_->context(), state_, segment_index, token_index);
            const asr_event token = {
                ASR_EVENT_TOKEN,
                token_text,
                token_text ? std::strlen(token_text) : 0,
                language,
                whisper_full_get_token_t0_from_state(state_, segment_index, token_index) * 10,
                whisper_full_get_token_t1_from_state(state_, segment_index, token_index) * 10,
                whisper_full_get_token_p_from_state(state_, segment_index, token_index),
            };
            if (!sink(token)) return true;
        }
    }
    return true;
}

class WhisperCppProvider final : public IASRProvider {
public:
    const char* name() const override { return "whisper.cpp"; }

    std::shared_ptr<IASRModel> load(const std::string& path, const RuntimeConfig& runtime) const override {
        auto params = whisper_context_default_params();
        if (!configure_device(runtime.device, params)) return nullptr;
        whisper_context* context = whisper_init_from_file_with_params_no_state(path.c_str(), params);
        return context ? std::make_shared<WhisperModel>(context, runtime) : nullptr;
    }
};

[[maybe_unused]] const bool registered = [] {
    ProviderRegistry::get().register_provider("whisper.cpp", [] {
        return std::make_unique<WhisperCppProvider>();
    });
    return true;
}();

} // namespace
} // namespace asr
