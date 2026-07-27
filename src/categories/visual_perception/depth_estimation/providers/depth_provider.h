#pragma once

#include "categories/visual_perception/depth_estimation.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace depth {

struct RuntimeConfig {
    std::string device = "auto";
    uint32_t n_threads = 1;
};

struct Request {
    const depth_image* image = nullptr;
    const depth_image* right = nullptr; // stereo only
    depth_task task = DEPTH_TASK_MONOCULAR;
};

// Sessions fill Result with storage of their own; the core deep-copies it
// into the caller-owned C depth_map, so values only needs to stay valid until
// estimate() returns.
struct Result {
    uint32_t width = 0;
    uint32_t height = 0;
    depth_map_kind kind = DEPTH_MAP_RELATIVE;
    std::vector<float> values; // width * height, row-major
};

class IDepthSession {
public:
    virtual ~IDepthSession() = default;
    virtual bool estimate(const Request& request, Result& result) = 0;
};

class IDepthModel {
public:
    virtual ~IDepthModel() = default;
    virtual std::unique_ptr<IDepthSession> create_session() = 0;
    virtual depth_capabilities capabilities() const = 0;
};

class IDepthProvider {
public:
    virtual ~IDepthProvider() = default;
    virtual const char* name() const = 0;
    virtual std::shared_ptr<IDepthModel> load(const std::string& path,
                                              const RuntimeConfig& runtime) const = 0;
};

// Providers register under the GGUF `general.architecture` values they load
// (one provider may claim several architectures).
class ProviderRegistry {
public:
    using Creator = std::function<std::unique_ptr<IDepthProvider>()>;

    static ProviderRegistry& get() {
        static ProviderRegistry registry;
        return registry;
    }

    void register_architecture(std::string architecture, Creator creator) {
        creators_[std::move(architecture)] = std::move(creator);
    }

    std::unique_ptr<IDepthProvider> create_for(const std::string& architecture) const {
        const auto it = creators_.find(architecture);
        return it == creators_.end() ? nullptr : it->second();
    }

private:
    std::unordered_map<std::string, Creator> creators_;
};

} // namespace depth
