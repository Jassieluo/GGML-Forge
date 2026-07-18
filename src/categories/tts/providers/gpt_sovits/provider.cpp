#include "provider.h"

#include <filesystem>
#include <memory>
#include <string>

namespace tts {
namespace {

class GPTSoVITSProvider final : public ITTSProvider {
public:
    const char* name() const override { return "gpt-sovits"; }

    bool validate(const ModelConfig& config, std::string& error) const override {
        if (!config.adapters.empty()) {
            error = "adapters are declared but are not supported";
            return false;
        }
        for (const char* component : {"dict", "hubert", "bert", "t2s", "vits"}) {
            const auto it = config.models.find(component);
            if (it == config.models.end() || it->second.empty()) {
                error = std::string("missing required component '") + component + "'";
                return false;
            }
            const auto path = config.resolve_path(it->second);
            if (!std::filesystem::exists(path)) {
                error = std::string("component '") + component + "' does not exist: " + path.string();
                return false;
            }
        }
        const auto speaker_encoder = config.models.find("speaker_encoder");
        if (speaker_encoder != config.models.end() &&
            !std::filesystem::exists(config.resolve_path(speaker_encoder->second))) {
            error = "component 'speaker_encoder' does not exist: " +
                config.resolve_path(speaker_encoder->second).string();
            return false;
        }
        return true;
    }

    std::shared_ptr<ITTSModel> load(
        const ModelConfig& config,
        const RuntimeContext& runtime
    ) const override {
        return gpt_sovits_provider::load_model(config, runtime);
    }
};

[[maybe_unused]] const bool registered = [] {
    TTSProviderRegistry::get().register_provider("gpt-sovits", [] {
        return std::make_unique<GPTSoVITSProvider>();
    });
    return true;
}();

} // namespace
} // namespace tts
