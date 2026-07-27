#pragma once

#include "categories/visual_perception/instance_perception.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace visual_perception::instance {

struct RuntimeConfig {
    std::string device = "auto";
    uint32_t n_threads = 1;
};

struct Request {
    const instance_image* image = nullptr;
    instance_task task = INSTANCE_TASK_BOXES;
    float score_threshold = 0.25f;
    float iou_threshold = 0.45f;
    int32_t max_instances = 300;
};

// Sessions fill Result with storage of their own; the core deep-copies it
// into the caller-owned C instance_result, so mask/keypoint pointers only
// need to stay valid until perceive() returns.
struct Result {
    std::vector<perceived_instance> instances;
    std::vector<std::vector<uint8_t>> masks;
    std::vector<std::vector<instance_keypoint>> keypoints;
};

class IInstanceSession {
public:
    virtual ~IInstanceSession() = default;
    virtual bool perceive(const Request& request, Result& result) = 0;
};

class IInstanceModel {
public:
    virtual ~IInstanceModel() = default;
    virtual std::unique_ptr<IInstanceSession> create_session() = 0;
    virtual instance_capabilities capabilities() const = 0;
    // Borrowed; valid for the model's lifetime. Null when out of range.
    virtual const char* label(int32_t class_id) const = 0;
};

class IInstanceProvider {
public:
    virtual ~IInstanceProvider() = default;
    virtual const char* name() const = 0;
    virtual std::shared_ptr<IInstanceModel> load(const std::string& path,
                                                 const RuntimeConfig& runtime) const = 0;
};

// Providers register under the GGUF `general.architecture` values they load
// (one provider may claim several architectures).
class ProviderRegistry {
public:
    using Creator = std::function<std::unique_ptr<IInstanceProvider>()>;

    static ProviderRegistry& get() {
        static ProviderRegistry registry;
        return registry;
    }

    void register_architecture(std::string architecture, Creator creator) {
        creators_[std::move(architecture)] = std::move(creator);
    }

    std::unique_ptr<IInstanceProvider> create_for(const std::string& architecture) const {
        const auto it = creators_.find(architecture);
        return it == creators_.end() ? nullptr : it->second();
    }

private:
    std::unordered_map<std::string, Creator> creators_;
};

} // namespace visual_perception::instance
