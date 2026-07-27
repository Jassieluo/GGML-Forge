#pragma once

#include "categories/visual_perception/image_classification.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace visual_perception::classification {

struct RuntimeConfig {
    std::string device = "auto";
    uint32_t n_threads = 1;
};

struct Request {
    const classification_image* image = nullptr;
    uint32_t top_k = 5;
};

struct Result {
    std::vector<classification_score> scores;
};

class IClassificationSession {
public:
    virtual ~IClassificationSession() = default;
    virtual bool classify(const Request& request, Result& result) = 0;
};

class IClassificationModel {
public:
    virtual ~IClassificationModel() = default;
    virtual std::unique_ptr<IClassificationSession> create_session() = 0;
    virtual classification_capabilities capabilities() const = 0;
    virtual const char* label(int32_t class_id) const = 0;
};

class IClassificationProvider {
public:
    virtual ~IClassificationProvider() = default;
    virtual const char* name() const = 0;
    virtual std::shared_ptr<IClassificationModel> load(
        const std::string& path, const RuntimeConfig& runtime) const = 0;
};

class ProviderRegistry {
public:
    using Creator = std::function<std::unique_ptr<IClassificationProvider>()>;

    static ProviderRegistry& get() {
        static ProviderRegistry registry;
        return registry;
    }

    void register_architecture(std::string architecture, Creator creator) {
        creators_[std::move(architecture)] = std::move(creator);
    }

    std::unique_ptr<IClassificationProvider> create_for(
        const std::string& architecture) const {
        const auto it = creators_.find(architecture);
        return it == creators_.end() ? nullptr : it->second();
    }

private:
    std::unordered_map<std::string, Creator> creators_;
};

} // namespace visual_perception::classification
