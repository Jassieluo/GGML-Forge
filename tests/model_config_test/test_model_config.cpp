#include "model_config.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

std::filesystem::path write_config(
    const std::filesystem::path& directory,
    const std::string& name,
    const std::string& json
) {
    const std::filesystem::path path = directory / name;
    std::ofstream output(path, std::ios::binary);
    output << json;
    return path;
}

bool load(const std::filesystem::path& path, tts::ModelConfig& config, std::string& error) {
    error.clear();
    return tts::load_model_config(path, config, error);
}

} // namespace

int main() {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "gpt_sovits_model_config_test";
    std::filesystem::create_directories(directory);

    tts::ModelConfig config;
    std::string error;

    const auto valid = write_config(directory, "valid.json", R"({
        "format": "tts-model",
        "name": "test model",
        "provider": "test-provider",
        "models": {"encoder": "weights/encoder.gguf"},
        "adapters": []
    })");
    check(load(valid, config, error), "valid config should load: " + error);
    check(config.provider == "test-provider", "provider should be preserved");
    check(config.resolve_path("weights/encoder.gguf") ==
          (directory / "weights/encoder.gguf").lexically_normal(),
          "relative paths should resolve from the config directory");

    const auto unicode_name = write_config(directory, "unicode-name.json", R"({
        "format": "tts-model",
        "name": "voice \uD83C\uDFA4", "provider": "p", "models": {"m": "m.gguf"}
    })");
    check(load(unicode_name, config, error), "valid unicode surrogate pair should load: " + error);
    check(config.name == "voice \xF0\x9F\x8E\xA4", "unicode surrogate pair should decode as UTF-8");

    const auto wrong_format = write_config(directory, "wrong-format.json", R"({
        "format": "other", "name": "x",
        "provider": "p", "models": {"m": "m.gguf"}
    })");
    check(!load(wrong_format, config, error), "wrong format should be rejected");

    const auto missing_provider = write_config(directory, "missing-provider.json", R"({
        "format": "tts-model", "name": "x",
        "models": {"m": "m.gguf"}
    })");
    check(!load(missing_provider, config, error), "missing provider should be rejected");

    const auto missing_name = write_config(directory, "missing-name.json", R"({
        "format": "tts-model",
        "provider": "p", "models": {"m": "m.gguf"}
    })");
    check(!load(missing_name, config, error), "missing name should be rejected");

    const auto missing_models = write_config(directory, "missing-models.json", R"({
        "format": "tts-model", "name": "x", "provider": "p"
    })");
    check(!load(missing_models, config, error), "missing models should be rejected");

    const auto duplicate_model = write_config(directory, "duplicate-model.json", R"({
        "format": "tts-model", "name": "x",
        "provider": "p", "models": {"m": "a.gguf", "m": "b.gguf"}
    })");
    check(!load(duplicate_model, config, error), "duplicate model keys should be rejected");

    const auto duplicate_root = write_config(directory, "duplicate-root.json", R"({
        "format": "tts-model", "name": "x", "name": "y",
        "provider": "p", "models": {"m": "m.gguf"}
    })");
    check(!load(duplicate_root, config, error), "duplicate root fields should be rejected");

    std::filesystem::remove_all(directory);
    if (failures != 0) return 1;

    std::cout << "All model config tests passed.\n";
    return 0;
}
