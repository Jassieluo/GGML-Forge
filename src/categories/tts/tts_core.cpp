#include "categories/tts/tts.h"
#include "tts_internal.h"
#include "pipelines/tts_pipeline.h"
#include "model_config.h"
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cstring>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <thread>

static int g_log_level = 0; // Default to info

void tts_set_log_level(int level) {
    g_log_level = level;
}

struct tts_runtime_params tts_runtime_default_params(void) {
    const unsigned hardware_threads = std::thread::hardware_concurrency();
    return {
        /* .device    = */ "auto",
        /* .n_threads = */ hardware_threads > 0 ? hardware_threads : 4u,
        /* .max_concurrency = */ 2u,
    };
}

tts_runtime_ptr tts_runtime_create(struct tts_runtime_params params) {
    ggml_backend_load_all();

    auto runtime = std::make_unique<tts_runtime>();
    auto& context = runtime->context;
    context.n_threads = params.n_threads > 0 ? params.n_threads : 1;
    context.max_concurrency = params.max_concurrency > 0 ? params.max_concurrency : 1;

    std::string requested_device = params.device ? params.device : "auto";
    std::transform(requested_device.begin(), requested_device.end(), requested_device.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    ggml_backend_dev_t selected_device = nullptr;
    const size_t device_count = ggml_backend_dev_count();
    if (requested_device.empty() || requested_device == "auto") {
        for (size_t i = 0; i < device_count; ++i) {
            ggml_backend_dev_t device = ggml_backend_dev_get(i);
            if (device && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                selected_device = device;
                break;
            }
        }
        if (!selected_device) requested_device = "cpu";
    }

    if (!selected_device && requested_device == "cpu") {
        for (size_t i = 0; i < device_count; ++i) {
            ggml_backend_dev_t device = ggml_backend_dev_get(i);
            if (device && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                selected_device = device;
                break;
            }
        }
    } else if (!selected_device) {
        selected_device = ggml_backend_dev_by_name(params.device);
    }

    if (!selected_device) {
        std::cerr << "[TTS Runtime] Error: Device is unavailable: "
                  << (params.device ? params.device : "auto") << std::endl;
        return nullptr;
    }

    context.device_name = ggml_backend_dev_name(selected_device);
    return runtime.release();
}

void tts_runtime_free(tts_runtime_ptr runtime) {
    delete runtime;
}

const char* tts_runtime_get_device(tts_runtime_ptr runtime) {
    return runtime ? runtime->context.device_name.c_str() : nullptr;
}

uint32_t tts_runtime_get_thread_count(tts_runtime_ptr runtime) {
    return runtime ? runtime->context.n_threads : 0;
}

uint32_t tts_runtime_get_max_concurrency(tts_runtime_ptr runtime) {
    return runtime ? runtime->context.max_concurrency : 0;
}

bool tts_runtime_set_component_policy(
    tts_runtime_ptr runtime,
    const char* component,
    const char* device,
    enum tts_component_residency residency
) {
    if (!runtime || !component || component[0] == '\0') return false;
    if (residency != TTS_COMPONENT_RESIDENT && residency != TTS_COMPONENT_ON_DEMAND) return false;

    std::string device_name = device ? device : "";
    if (!device_name.empty() && !ggml_backend_dev_by_name(device_name.c_str())) {
        std::cerr << "[TTS Runtime] Error: Component device is unavailable: " << device_name << std::endl;
        return false;
    }

    tts::ComponentRuntimePolicy policy;
    policy.device = std::move(device_name);
    policy.residency = residency == TTS_COMPONENT_ON_DEMAND
        ? tts::ComponentResidency::on_demand
        : tts::ComponentResidency::resident;

    std::lock_guard<std::mutex> lock(runtime->policy_mutex);
    runtime->context.components[component] = std::move(policy);
    return true;
}

tts_model_ptr tts_load_model(tts_runtime_ptr runtime, const char* path) {
    if (!runtime || !path) return nullptr;

    std::cout << "[TTS Core] Loading model from: " << path << std::endl;

    auto model = std::make_unique<tts_model>();

    // 1. Parse the portable model composition.
    std::string path_str(path);
    std::string arch_name;
    tts::ModelConfig config;
    const std::filesystem::path input_path = std::filesystem::u8path(path_str);

    if (!std::filesystem::is_regular_file(input_path) || input_path.extension() != ".json") {
        std::cerr << "[TTS Core] Error: Expected a tts-model JSON composition: " << path_str << std::endl;
        return nullptr;
    }
    std::string error;
    if (!tts::load_model_config(input_path, config, error)) {
        std::cerr << "[TTS Core] Error: " << error << std::endl;
        return nullptr;
    }
    arch_name = config.provider;

    model->arch_name = arch_name;
    model->config = config;
    {
        std::lock_guard<std::mutex> lock(runtime->policy_mutex);
        model->runtime = std::make_shared<const tts::RuntimeContext>(runtime->context);
    }

    // 2. Instantiate pipeline using the registry
    model->pipeline = tts::TTSPipelineRegistry::get().create(arch_name);
    if (!model->pipeline) {
        std::cerr << "[TTS Core] Error: Unsupported model architecture: " << model->arch_name << std::endl;
        return nullptr;
    }

    // 3. Load provider components using the machine-specific runtime selection.
    if (!model->pipeline->load(model->config, *model->runtime)) {
        std::cerr << "[TTS Core] Error: Failed to load pipeline for architecture: " << model->arch_name << std::endl;
        return nullptr;
    }

    return model.release();
}

void tts_free_model(tts_model_ptr model) {
    if (model) {
        delete model;
    }
}

tts_session_ptr tts_create_session(tts_model_ptr model) {
    if (!model) return nullptr;

    auto session = std::make_unique<tts_session>();
    session->pipeline = model->pipeline;
    session->session = session->pipeline->create_session();
    if (!session->session) {
        std::cerr << "[TTS Core] Error: Provider failed to create a session." << std::endl;
        return nullptr;
    }

    return session.release();
}

void tts_free_session(tts_session_ptr session) {
    if (session) {
        delete session;
    }
}

bool tts_session_set_reference(
    tts_session_ptr session,
    const float* audio,
    size_t sample_count,
    int32_t sample_rate,
    const char* text,
    const char* language
) {
    if (!session || !session->session || !audio || sample_count == 0 || sample_rate <= 0) return false;

    tts::VoiceReference reference;
    reference.audio.assign(audio, audio + sample_count);
    reference.sample_rate = sample_rate;
    reference.text = text ? text : "";
    reference.language = language ? language : "zh";
    return session->session->set_reference(reference);
}

const float* tts_synthesize(
    tts_session_ptr session,
    const char*     text,
    const char*     lang,
    float           speed,
    int32_t*        out_samples_count
) {
    if (!session || !text || !out_samples_count) {
        if (out_samples_count) *out_samples_count = 0;
        return nullptr;
    }

    if (!session->session) {
        std::cerr << "[TTS Core] Error: Session is not initialized" << std::endl;
        *out_samples_count = 0;
        return nullptr;
    }

    tts::SynthesisRequest req;
    req.text = text;
    req.language = lang ? lang : "zh";
    req.float_params["speed"] = speed;

    // Execute synthesis
    session->audio_output = session->session->synthesize(req);

    *out_samples_count = static_cast<int32_t>(session->audio_output.size());
    return session->audio_output.data();
}
