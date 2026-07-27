#include "categories/asr/asr.h"
#include "asr_internal.h"
#include "audio/resample.h"

#include <memory>
#include <thread>

struct asr_runtime_params asr_runtime_default_params(void) {
    const auto threads = std::thread::hardware_concurrency();
    return {"auto", threads > 0 ? threads : 4u};
}

asr_runtime_ptr asr_runtime_create(struct asr_runtime_params params) {
    auto runtime = std::make_unique<asr_runtime>();
    runtime->config.device = params.device && params.device[0] ? params.device : "auto";
    runtime->config.n_threads = params.n_threads > 0 ? params.n_threads : 1u;
    return runtime.release();
}

void asr_runtime_free(asr_runtime_ptr runtime) { delete runtime; }

const char* asr_runtime_get_device(asr_runtime_ptr runtime) {
    return runtime ? runtime->config.device.c_str() : nullptr;
}

uint32_t asr_runtime_get_thread_count(asr_runtime_ptr runtime) {
    return runtime ? runtime->config.n_threads : 0;
}

asr_model_ptr asr_load_model(asr_runtime_ptr runtime, const char* path) {
    if (!runtime || !path || !path[0]) return nullptr;
    auto provider = asr::ProviderRegistry::get().create("whisper.cpp");
    if (!provider) return nullptr;
    auto implementation = provider->load(path, runtime->config);
    if (!implementation) return nullptr;
    auto model = std::make_unique<asr_model>();
    model->provider_name = provider->name();
    model->implementation = std::move(implementation);
    return model.release();
}

void asr_free_model(asr_model_ptr model) { delete model; }

const char* asr_model_get_provider(asr_model_ptr model) {
    return model ? model->provider_name.c_str() : nullptr;
}

struct asr_capabilities asr_model_get_capabilities(asr_model_ptr model) {
    return model && model->implementation
        ? model->implementation->capabilities()
        : asr_capabilities{};
}

asr_session_ptr asr_create_session(asr_model_ptr model) {
    if (!model || !model->implementation) return nullptr;
    auto session = std::make_unique<asr_session>();
    session->model = model->implementation;
    session->implementation = session->model->create_session();
    return session->implementation ? session.release() : nullptr;
}

void asr_free_session(asr_session_ptr session) { delete session; }

bool asr_session_reset(asr_session_ptr session) {
    return session && session->implementation && session->implementation->reset();
}

struct asr_request_params asr_request_default_params(void) {
    return {ASR_TASK_TRANSCRIBE, "auto", false};
}

bool asr_transcribe(
    asr_session_ptr session,
    const float* mono_audio,
    size_t sample_count,
    int32_t sample_rate,
    struct asr_request_params params,
    asr_event_callback callback,
    void* user_data
) {
    if (!session || !session->implementation || !callback ||
        (params.task != ASR_TASK_TRANSCRIBE && params.task != ASR_TASK_TRANSLATE) ||
        !forge::media::valid_mono_audio(mono_audio, sample_count, sample_rate)) return false;
    auto normalized = forge::media::resample_mono(mono_audio, sample_count, sample_rate, 16000);
    if (normalized.empty()) return false;

    asr::Request request;
    request.audio = normalized.data();
    request.sample_count = normalized.size();
    request.language = params.language ? params.language : "";
    request.task = params.task;
    request.token_timestamps = params.token_timestamps;
    return session->implementation->transcribe(request, [callback, user_data](const asr_event& event) {
        return callback(&event, user_data);
    });
}
