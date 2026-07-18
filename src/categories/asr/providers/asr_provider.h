#pragma once

#include "categories/asr/asr.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace asr {

struct RuntimeConfig {
    std::string device = "auto";
    uint32_t n_threads = 1;
};

struct Request {
    const float* audio = nullptr;
    size_t sample_count = 0;
    std::string language;
    asr_task task = ASR_TASK_TRANSCRIBE;
    bool token_timestamps = false;
};

using EventSink = std::function<bool(const asr_event&)>;

class IASRSession {
public:
    virtual ~IASRSession() = default;
    virtual bool reset() = 0;
    virtual bool transcribe(const Request& request, EventSink sink) = 0;
};

class IASRModel {
public:
    virtual ~IASRModel() = default;
    virtual std::unique_ptr<IASRSession> create_session() = 0;
    virtual asr_capabilities capabilities() const = 0;
};

class IASRProvider {
public:
    virtual ~IASRProvider() = default;
    virtual const char* name() const = 0;
    virtual std::shared_ptr<IASRModel> load(const std::string& path, const RuntimeConfig& runtime) const = 0;
};

class ProviderRegistry {
public:
    using Creator = std::function<std::unique_ptr<IASRProvider>()>;

    static ProviderRegistry& get() {
        static ProviderRegistry registry;
        return registry;
    }

    void register_provider(std::string name, Creator creator) {
        creators_[std::move(name)] = std::move(creator);
    }

    std::unique_ptr<IASRProvider> create(const std::string& name) const {
        const auto it = creators_.find(name);
        return it == creators_.end() ? nullptr : it->second();
    }

private:
    std::unordered_map<std::string, Creator> creators_;
};

} // namespace asr
