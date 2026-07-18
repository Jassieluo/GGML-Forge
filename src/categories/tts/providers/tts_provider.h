#pragma once

// Internal provider contract for the TTS category.

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
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

struct TTSCapabilities {
    bool streaming = false;
    bool voice_cloning = false;
    bool speaker_id = false;
    bool speaker_embedding = false;
    bool emotion = false;
    bool deterministic_seed = false;
    bool speed_control = false;
};

class ITTSSession {
public:
    virtual ~ITTSSession() = default;

    virtual bool set_reference(const VoiceReference&) { return false; }
    virtual std::vector<float> synthesize(const SynthesisRequest& request) = 0;
    virtual bool synthesize_streaming(
        const SynthesisRequest& request,
        AudioChunkCallback callback
    ) = 0;
    virtual int32_t output_sample_rate() const = 0;
};

// A loaded model owns immutable artifacts and provider-defined execution lanes.
class ITTSModel {
public:
    virtual ~ITTSModel() = default;
    virtual const TTSCapabilities& capabilities() const = 0;
    virtual std::unique_ptr<ITTSSession> create_session() = 0;
};

// A registry entry is a lightweight, stateless factory for one composition format.
class ITTSProvider {
public:
    virtual ~ITTSProvider() = default;
    virtual const char* name() const = 0;
    virtual bool validate(const ModelConfig& config, std::string& error) const = 0;

    virtual std::shared_ptr<ITTSModel> load(
        const ModelConfig& config,
        const RuntimeContext& runtime
    ) const = 0;
};

class TTSProviderRegistry {
public:
    using Creator = std::function<std::unique_ptr<ITTSProvider>()>;

    static TTSProviderRegistry& get() {
        static TTSProviderRegistry instance;
        return instance;
    }

    void register_provider(const std::string& name, Creator creator) {
        creators_[name] = creator;
    }

    std::unique_ptr<ITTSProvider> create(const std::string& name) {
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
