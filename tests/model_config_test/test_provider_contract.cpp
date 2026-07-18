#include "providers/tts_provider.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

class FakeSession final : public tts::ITTSSession {
public:
    std::vector<float> synthesize(const tts::SynthesisRequest&) override { return {0.25f}; }
    bool synthesize_streaming(const tts::SynthesisRequest&, tts::AudioChunkCallback callback) override {
        const float sample = 0.25f;
        callback(&sample, 1);
        return true;
    }
    int32_t output_sample_rate() const override { return 24000; }
};

class FakeModel final : public tts::ITTSModel {
public:
    const tts::TTSCapabilities& capabilities() const override {
        static const tts::TTSCapabilities value = {
            true, false, false, false, false, false, true,
        };
        return value;
    }
    std::unique_ptr<tts::ITTSSession> create_session() override {
        return std::make_unique<FakeSession>();
    }
};

class FakeProvider final : public tts::ITTSProvider {
public:
    const char* name() const override { return "contract-test"; }
    bool validate(const tts::ModelConfig& config, std::string& error) const override {
        if (config.provider == name()) return true;
        error = "provider mismatch";
        return false;
    }
    std::shared_ptr<tts::ITTSModel> load(
        const tts::ModelConfig&,
        const tts::RuntimeContext&
    ) const override {
        return std::make_shared<FakeModel>();
    }
};

} // namespace

int main() {
    try {
        tts::TTSProviderRegistry::get().register_provider("contract-test", [] {
            return std::make_unique<FakeProvider>();
        });
        auto provider = tts::TTSProviderRegistry::get().create("contract-test");
        require(provider && std::string(provider->name()) == "contract-test", "provider factory failed");

        tts::ModelConfig config;
        config.provider = "contract-test";
        std::string error;
        require(provider->validate(config, error), "provider validation failed");

        tts::RuntimeContext runtime;
        auto model = provider->load(config, runtime);
        require(model && model->capabilities().streaming, "loaded model contract failed");
        auto session = model->create_session();
        require(session && session->output_sample_rate() == 24000, "session contract failed");

        tts::SynthesisRequest request;
        require(session->synthesize(request).size() == 1, "session synthesis failed");
        std::cout << "TTS provider contract tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
