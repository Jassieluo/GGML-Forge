#pragma once

#include <cstdint>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace llm {

struct RuntimeConfig {
    uint32_t n_ctx = 4096;
    uint32_t n_batch = 512;
    uint32_t n_threads = 1;
    int32_t n_gpu_layers = 0;
    std::string device = "auto";
};

struct ModelConfig {
    std::string model;
    std::string mmproj;
};

struct Capabilities {
    bool vision = false;
    bool audio = false;
};

struct ContentPart {
    enum class Type { Text, Image, Audio } type = Type::Text;
    const void* data = nullptr;
    size_t size = 0;
};

struct GenerationRequest {
    std::string prompt;
    std::vector<ContentPart> content;
    int32_t max_tokens = 128;
    float temperature = 0.8f;
    int32_t top_k = 40;
    float top_p = 0.95f;
    uint32_t seed = 0xFFFFFFFFu;
};

struct ChatMessage {
    std::string role;
    std::string content;
    std::vector<ContentPart> parts;
};

using TextSink = std::function<bool(const char*, size_t)>;

class ILLMSession {
public:
    virtual ~ILLMSession() = default;
    virtual bool reset() = 0;
    virtual bool generate(const GenerationRequest& request, TextSink sink) = 0;
    virtual bool generate_chat(
        const std::vector<ChatMessage>& messages,
        const GenerationRequest& parameters,
        TextSink sink) = 0;
};

class ILLMModel {
public:
    virtual ~ILLMModel() = default;
    virtual std::unique_ptr<ILLMSession> create_session() = 0;
    virtual Capabilities capabilities() const = 0;
    virtual bool format_chat(const std::vector<ChatMessage>& messages, std::string& prompt) const = 0;
};

class ILLMProvider {
public:
    virtual ~ILLMProvider() = default;
    virtual const char* name() const = 0;
    virtual std::shared_ptr<ILLMModel> load(const ModelConfig& model, const RuntimeConfig& runtime) const = 0;
};

class ProviderRegistry {
public:
    using Creator = std::function<std::unique_ptr<ILLMProvider>()>;

    static ProviderRegistry& get();

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
