#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace tts {

struct AdapterConfig {
    std::string name;
    std::string path;
    std::string target;
    float scale = 1.0f;
};

struct ModelConfig {
    std::string name;
    std::string provider;
    std::unordered_map<std::string, std::string> models;
    std::vector<AdapterConfig> adapters;

    std::filesystem::path source_path;
    std::filesystem::path base_directory;

    std::filesystem::path resolve_path(const std::string& value) const;
};

bool load_model_config(
    const std::filesystem::path& path,
    ModelConfig& config,
    std::string& error
);

} // namespace tts
