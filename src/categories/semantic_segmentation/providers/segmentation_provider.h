#pragma once

#include "categories/semantic_segmentation/semantic_segmentation.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace segmentation {

struct RuntimeConfig {
    std::string device = "auto";
    uint32_t n_threads = 1;
};

struct Request {
    const segmentation_image* image = nullptr;
    bool want_confidence = false;
};

// Sessions fill Result with storage of their own; the core deep-copies it
// into the caller-owned C segmentation_result, so the vectors only need to
// stay valid until segment() returns.
struct Result {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<int32_t> class_map;  // width * height, row-major
    std::vector<float> confidence;   // empty, or width * height when requested
};

class ISegmentationSession {
public:
    virtual ~ISegmentationSession() = default;
    virtual bool segment(const Request& request, Result& result) = 0;
};

class ISegmentationModel {
public:
    virtual ~ISegmentationModel() = default;
    virtual std::unique_ptr<ISegmentationSession> create_session() = 0;
    virtual segmentation_capabilities capabilities() const = 0;
    // Borrowed; valid for the model's lifetime. Null when out of range.
    virtual const char* label(int32_t class_id) const = 0;
    // False when out of range or the model ships no palette.
    virtual bool color(int32_t class_id, uint8_t rgb[3]) const = 0;
};

class ISegmentationProvider {
public:
    virtual ~ISegmentationProvider() = default;
    virtual const char* name() const = 0;
    virtual std::shared_ptr<ISegmentationModel> load(const std::string& path,
                                                     const RuntimeConfig& runtime) const = 0;
};

// Providers register under the GGUF `general.architecture` values they load
// (one provider may claim several architectures).
class ProviderRegistry {
public:
    using Creator = std::function<std::unique_ptr<ISegmentationProvider>()>;

    static ProviderRegistry& get() {
        static ProviderRegistry registry;
        return registry;
    }

    void register_architecture(std::string architecture, Creator creator) {
        creators_[std::move(architecture)] = std::move(creator);
    }

    std::unique_ptr<ISegmentationProvider> create_for(const std::string& architecture) const {
        const auto it = creators_.find(architecture);
        return it == creators_.end() ? nullptr : it->second();
    }

private:
    std::unordered_map<std::string, Creator> creators_;
};

} // namespace segmentation
