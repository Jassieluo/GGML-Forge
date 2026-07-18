#include "protocols/chat.h"
#include "codecs.h"

#include <algorithm>
#include <atomic>
#include <chrono>

namespace forge::server {
namespace {

bool decode_data_url(const std::string& url, std::string& mime, std::vector<uint8_t>& data) {
    if (url.rfind("data:", 0) != 0) return false;
    const size_t separator = url.find(',');
    if (separator == std::string::npos) return false;
    const std::string metadata = url.substr(5, separator - 5);
    const size_t encoding = metadata.find(';');
    mime = metadata.substr(0, encoding);
    if (encoding == std::string::npos || metadata.substr(encoding + 1) != "base64") return false;
    return base64_decode(url.substr(separator + 1), data);
}

bool content_parts(const Json& value, bool anthropic, ChatMessage& message, std::string& error) {
    if (value.is_string()) {
        message.content = value.get<std::string>();
        message.parts.push_back({ChatMessage::Part::Type::Text, message.content, {}, "text/plain"});
        return true;
    }
    if (!value.is_array()) return false;
    for (const Json& part : value) {
        if (!part.is_object()) return false;
        const std::string type = part.value("type", "");
        if (type == "text" || type == "input_text") {
            if (!part.contains("text") || !part["text"].is_string()) return false;
            const std::string text = part["text"].get<std::string>();
            message.content += text;
            message.parts.push_back({ChatMessage::Part::Type::Text, text, {}, "text/plain"});
        } else if (!anthropic && type == "image_url") {
            if (!part.contains("image_url")) return false;
            const Json& image = part["image_url"];
            const std::string url = image.is_string() ? image.get<std::string>() : image.value("url", "");
            ChatMessage::Part parsed;
            parsed.type = ChatMessage::Part::Type::Image;
            if (!decode_data_url(url, parsed.mime_type, parsed.data)) {
                error = "image_url currently requires a base64 data URL";
                return false;
            }
            message.parts.push_back(std::move(parsed));
        } else if (!anthropic && type == "input_audio") {
            if (!part.contains("input_audio") || !part["input_audio"].is_object()) return false;
            ChatMessage::Part parsed;
            parsed.type = ChatMessage::Part::Type::Audio;
            parsed.mime_type = "audio/" + part["input_audio"].value("format", "wav");
            if (!base64_decode(part["input_audio"].value("data", ""), parsed.data)) return false;
            message.parts.push_back(std::move(parsed));
        } else if (anthropic && type == "image") {
            if (!part.contains("source") || !part["source"].is_object() ||
                part["source"].value("type", "") != "base64") return false;
            ChatMessage::Part parsed;
            parsed.type = ChatMessage::Part::Type::Image;
            parsed.mime_type = part["source"].value("media_type", "image/png");
            if (!base64_decode(part["source"].value("data", ""), parsed.data)) return false;
            message.parts.push_back(std::move(parsed));
        } else {
            error = "unsupported message content block: " + type;
            return false;
        }
    }
    return true;
}

void parse_sampling(const Json& body, ChatRequest& request) {
    if (body.contains("max_tokens") && body["max_tokens"].is_number_integer())
        request.max_tokens = body["max_tokens"].get<int32_t>();
    if (body.contains("max_completion_tokens") && body["max_completion_tokens"].is_number_integer())
        request.max_tokens = body["max_completion_tokens"].get<int32_t>();
    if (body.contains("temperature") && body["temperature"].is_number())
        request.temperature = body["temperature"].get<float>();
    if (body.contains("top_p") && body["top_p"].is_number()) request.top_p = body["top_p"].get<float>();
    if (body.contains("top_k") && body["top_k"].is_number_integer()) request.top_k = body["top_k"].get<int32_t>();
    if (body.contains("seed") && body["seed"].is_number_unsigned()) request.seed = body["seed"].get<uint32_t>();
    request.stream = body.value("stream", false);
}

} // namespace

bool parse_openai_chat(const Json& body, ChatRequest& request, std::string& error) {
    if (!body.is_object() || !body.contains("messages") || !body["messages"].is_array()) {
        error = "messages must be an array";
        return false;
    }
    request.model = body.value("model", "llm");
    parse_sampling(body, request);
    for (const Json& item : body["messages"]) {
        if (!item.is_object() || !item.contains("role") || !item["role"].is_string() ||
            !item.contains("content")) {
            error = "each message requires role and content";
            return false;
        }
        ChatMessage message;
        message.role = item["role"].get<std::string>();
        if (!content_parts(item["content"], false, message, error)) {
            if (error.empty()) error = "invalid OpenAI message content";
            return false;
        }
        if (message.role == "system") {
            if (!request.system.empty()) request.system += "\n";
            request.system += message.content;
        } else {
            request.messages.push_back(std::move(message));
        }
    }
    if (request.messages.empty() || request.max_tokens <= 0) {
        error = "at least one non-system message and positive max_tokens are required";
        return false;
    }
    return true;
}

bool parse_anthropic_messages(const Json& body, ChatRequest& request, std::string& error) {
    if (!body.is_object() || !body.contains("messages") || !body["messages"].is_array()) {
        error = "messages must be an array";
        return false;
    }
    request.model = body.value("model", "llm");
    if (body.contains("system")) {
        ChatMessage system;
        if (!content_parts(body["system"], true, system, error) ||
            std::any_of(system.parts.begin(), system.parts.end(), [](const ChatMessage::Part& part) {
                return part.type != ChatMessage::Part::Type::Text;
            })) {
            error = "system must be text or an array of text blocks";
            return false;
        }
        request.system = std::move(system.content);
    }
    parse_sampling(body, request);
    for (const Json& item : body["messages"]) {
        if (!item.is_object() || !item.contains("role") || !item["role"].is_string() ||
            !item.contains("content")) {
            error = "each message requires role and content";
            return false;
        }
        ChatMessage message;
        message.role = item["role"].get<std::string>();
        if (!content_parts(item["content"], true, message, error)) {
            if (error.empty()) error = "invalid Anthropic message content";
            return false;
        }
        request.messages.push_back(std::move(message));
    }
    if (request.messages.empty() || request.max_tokens <= 0) {
        error = "at least one message and positive max_tokens are required";
        return false;
    }
    return true;
}

int64_t unix_seconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string request_id(const char* prefix) {
    static std::atomic<uint64_t> counter{1};
    return std::string(prefix) + std::to_string(unix_seconds()) + "-" +
        std::to_string(counter.fetch_add(1, std::memory_order_relaxed));
}

Json openai_chat_response(
    const ChatRequest& request, const std::string& id, const std::string& content
) {
    return {
        {"id", id}, {"object", "chat.completion"}, {"created", unix_seconds()},
        {"model", request.model},
        {"choices", Json::array({{{"index", 0},
            {"message", {{"role", "assistant"}, {"content", content}}},
            {"finish_reason", "stop"}}})},
        {"usage", {{"prompt_tokens", 0}, {"completion_tokens", 0}, {"total_tokens", 0}}}
    };
}

Json anthropic_message_response(
    const ChatRequest& request, const std::string& id, const std::string& content
) {
    return {
        {"id", id}, {"type", "message"}, {"role", "assistant"},
        {"model", request.model},
        {"content", Json::array({{{"type", "text"}, {"text", content}}})},
        {"stop_reason", "end_turn"}, {"stop_sequence", nullptr},
        {"usage", {{"input_tokens", 0}, {"output_tokens", 0}}}
    };
}

} // namespace forge::server
