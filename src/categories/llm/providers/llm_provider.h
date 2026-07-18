#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace llm {

struct RuntimeConfig {
    uint32_t n_ctx = 4096;
    uint32_t n_batch = 512;
    uint32_t n_threads = 1;
    int32_t n_gpu_layers = 0;
};

struct GenerationRequest {
    std::string prompt;
    int32_t max_tokens = 128;
    float temperature = 0.8f;
    int32_t top_k = 40;
    float top_p = 0.95f;
    uint32_t seed = 0xFFFFFFFFu;
};

using TextSink = std::function<bool(const char*, size_t)>;

class ILLMSession {
public:
    virtual ~ILLMSession() = default;
    virtual bool reset() = 0;
    virtual bool generate(const GenerationRequest& request, TextSink sink) = 0;
};

class ILLMModel {
public:
    virtual ~ILLMModel() = default;
    virtual std::unique_ptr<ILLMSession> create_session() = 0;
};

class ILLMProvider {
public:
    virtual ~ILLMProvider() = default;
    virtual const char* name() const = 0;
    virtual std::shared_ptr<ILLMModel> load(const std::string& path, const RuntimeConfig& runtime) const = 0;
};

class ProviderRegistry {
public:
    using Creator = std::function<std::unique_ptr<ILLMProvider>()>;

    static ProviderRegistry& get() {
        static ProviderRegistry registry;
        return registry;
    }

    void register_provider(std::string name, Creator creator) {
        creators_[std::move(name)] = std::move(creator);
    }

    std::unique_ptr<ILLMProvider> create(const std::string& name) const {
        const auto it = creators_.find(name);
        return it == creators_.end() ? nullptr : it->second();
    }

private:
    std::unordered_map<std::string, Creator> creators_;
};

} // namespace llm
