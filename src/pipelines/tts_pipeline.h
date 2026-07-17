#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include "ggml.h"
#include "../model_config.h"
#include "../runtime.h"

namespace tts {

struct SynthesisRequest {
    std::string text;
    std::string language;
    std::vector<float> ref_audio;
    std::vector<float> speaker_embedding;
    std::unordered_map<std::string, float> float_params;
    std::unordered_map<std::string, std::string> string_params;
};

struct VoiceReference {
    std::vector<float> audio;
    int32_t sample_rate = 0;
    std::string text;
    std::string language;
};

using AudioChunkCallback = std::function<void(const float* audio_data, size_t n_samples)>;

class ITTSSession {
public:
    virtual ~ITTSSession() = default;

    virtual bool set_reference(const VoiceReference&) { return false; }
    virtual std::vector<float> synthesize(const SynthesisRequest& request) = 0;
    virtual bool synthesize_streaming(
        const SynthesisRequest& request,
        AudioChunkCallback callback
    ) = 0;
};

class ITTSPipeline {
public:
    virtual ~ITTSPipeline() = default;

    // Load provider-defined shared model artifacts.
    virtual bool load(const ModelConfig& config, const RuntimeContext& runtime) = 0;

    // Create isolated request/voice state while sharing loaded model artifacts.
    virtual std::unique_ptr<ITTSSession> create_session() = 0;
};

class TTSPipelineRegistry {
public:
    using Creator = std::function<std::unique_ptr<ITTSPipeline>()>;

    static TTSPipelineRegistry& get() {
        static TTSPipelineRegistry instance;
        return instance;
    }

    void register_pipeline(const std::string& name, Creator creator) {
        creators_[name] = creator;
    }

    std::unique_ptr<ITTSPipeline> create(const std::string& name) {
        auto it = creators_.find(name);
        if (it != creators_.end()) {
            return it->second();
        }
        return nullptr;
    }

private:
    std::unordered_map<std::string, Creator> creators_;
};

} // namespace tts
