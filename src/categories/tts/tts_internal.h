#pragma once

#include "categories/tts/tts.h"
#include "providers/tts_provider.h"
#include "runtime.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>

// Forward declarations
namespace tts {
    class ITTSProvider;
}

struct tts_runtime {
    tts::RuntimeContext context;
    std::mutex policy_mutex;
};

// Internal implementation of tts_model
struct tts_model {
    std::string provider_name;
    tts::ModelConfig config;
    std::shared_ptr<const tts::RuntimeContext> runtime;
    std::shared_ptr<tts::ITTSModel> implementation;
};

// Internal implementation of tts_session
struct tts_session {
    std::shared_ptr<tts::ITTSModel> model;
    std::unique_ptr<tts::ITTSSession> session;
    std::unordered_map<std::string, float> float_params;
    std::unordered_map<std::string, std::string> string_params;
    std::vector<float> audio_output;
    tts_progress_callback progress_callback = nullptr;
    void* progress_user_data = nullptr;
};
