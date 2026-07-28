#pragma once

#include "pages/state.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace app {

enum class ModelKind { Llm, Tts, ImageGeneration, Vision };

struct ModelEntry {
    ModelKind kind = ModelKind::Llm;
    VisionTask vision_task = VisionTask::Classification;
    std::string name;
    std::string path;
    std::string detail;
    std::uintmax_t bytes = 0;
};

class ModelCatalog {
public:
    const std::vector<ModelEntry>& entries() const { return entries_; }

    std::vector<const ModelEntry*> models(ModelKind kind) const {
        std::vector<const ModelEntry*> result;
        for (const auto& entry : entries_) if (entry.kind == kind) result.push_back(&entry);
        return result;
    }

    std::vector<const ModelEntry*> visionModels(VisionTask task) const {
        std::vector<const ModelEntry*> result;
        for (const auto& entry : entries_) {
            if (entry.kind == ModelKind::Vision && entry.vision_task == task) result.push_back(&entry);
        }
        return result;
    }

    const ModelEntry* find(const std::string& path) const {
        for (const auto& entry : entries_) if (entry.path == path) return &entry;
        return nullptr;
    }

    static ModelCatalog scan() {
        ModelCatalog catalog;
        catalog.scanTree(projectRoot() / "models");
        std::sort(catalog.entries_.begin(), catalog.entries_.end(),
                  [](const ModelEntry& a, const ModelEntry& b) {
                      if (a.kind != b.kind) return a.kind < b.kind;
                      if (a.vision_task != b.vision_task) return a.vision_task < b.vision_task;
                      return a.name < b.name;
                  });
        return catalog;
    }

private:
    static std::string lower(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return value;
    }

    static std::string sizeLabel(std::uintmax_t bytes) {
        const double mb = static_cast<double>(bytes) / (1024.0 * 1024.0);
        char buffer[32]{};
        if (mb >= 1024.0) std::snprintf(buffer, sizeof(buffer), "%.1f GB", mb / 1024.0);
        else std::snprintf(buffer, sizeof(buffer), "%.0f MB", mb);
        return buffer;
    }

    void add(ModelKind kind, const std::filesystem::directory_entry& file,
             VisionTask task = VisionTask::Classification) {
        ModelEntry entry;
        entry.kind = kind;
        entry.vision_task = task;
        entry.name = file.path().stem().u8string();
        entry.path = file.path().u8string();
        std::error_code ec;
        entry.bytes = file.file_size(ec);
        entry.detail = sizeLabel(entry.bytes);
        entries_.push_back(std::move(entry));
    }

    void scanTree(const std::filesystem::path& root) {
        std::error_code ec;
        if (!std::filesystem::exists(root, ec)) return;
        for (std::filesystem::recursive_directory_iterator it(root, ec), end;
             it != end && !ec; it.increment(ec)) {
            if (!it->is_regular_file(ec)) continue;
            const std::string path = lower(it->path().generic_u8string());
            const std::string name = lower(it->path().filename().u8string());
            const std::string ext = lower(it->path().extension().u8string());
            if (path.find("/.cache/") != std::string::npos || name.find("f16") != std::string::npos) continue;

            if (path.find("/llm/") != std::string::npos && ext == ".gguf" &&
                name.find("q4") != std::string::npos && name.find("mmproj") == std::string::npos) {
                add(ModelKind::Llm, *it);
            } else if (path.find("/tts/") != std::string::npos && ext == ".json" &&
                       name.find("q4") != std::string::npos) {
                add(ModelKind::Tts, *it);
            } else if (path.find("/visual_generation/") != std::string::npos &&
                       ext == ".gguf" && name.find("q4") != std::string::npos) {
                add(ModelKind::ImageGeneration, *it);
            } else if (path.find("/visual_perception/") != std::string::npos && ext == ".gguf" &&
                       name.find("q4") != std::string::npos) {
                if (path.find("image_classification") != std::string::npos) {
                    add(ModelKind::Vision, *it, VisionTask::Classification);
                } else if (path.find("semantic_segmentation") != std::string::npos) {
                    add(ModelKind::Vision, *it, VisionTask::SemanticSegmentation);
                } else if (path.find("depth_estimation") != std::string::npos) {
                    add(ModelKind::Vision, *it,
                        name.find("stereo") != std::string::npos ? VisionTask::StereoDepth
                                                                : VisionTask::MonocularDepth);
                } else if (path.find("instance_perception") != std::string::npos) {
                    const VisionTask task = name.find("-seg") != std::string::npos ? VisionTask::InstanceSegmentation
                                          : name.find("-pose") != std::string::npos ? VisionTask::Pose
                                          : name.find("-obb") != std::string::npos ? VisionTask::Obb
                                                                                  : VisionTask::Detection;
                    add(ModelKind::Vision, *it, task);
                }
            }
        }
    }

    std::vector<ModelEntry> entries_;
};

inline const ModelCatalog& modelCatalog() {
    static const ModelCatalog value = ModelCatalog::scan();
    return value;
}

inline std::string firstModel(ModelKind kind) {
    const auto items = modelCatalog().models(kind);
    if (kind == ModelKind::Tts) {
        for (const ModelEntry* item : items) {
            if (item->name == "v2-q4") return item->path;
        }
    }
    return items.empty() ? std::string{} : items.front()->path;
}

inline std::string firstVisionModel(VisionTask task) {
    const auto items = modelCatalog().visionModels(task);
    return items.empty() ? std::string{} : items.front()->path;
}

} // namespace app
