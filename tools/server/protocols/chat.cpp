#include "protocols/chat.h"
#include "codecs.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>

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

bool valid_tool_calls(const Json& value) {
    if (!value.is_array()) return false;
    for (const Json& call : value) {
        if (!call.is_object() || !call.contains("type") || !call["type"].is_string() ||
            call["type"] != "function" || !call.contains("function") ||
            !call["function"].is_object() || !call["function"].contains("name") ||
            !call["function"]["name"].is_string() ||
            !call["function"].contains("arguments") ||
            !call["function"]["arguments"].is_string()) {
            return false;
        }
    }
    return true;
}

bool parse_sampling_options_impl(const Json& body, ChatRequest& request, std::string& error) {
    try {
        if (body.contains("max_tokens")) {
            if (!body["max_tokens"].is_number_integer()) {
                error = "max_tokens must be an integer";
                return false;
            }
            request.max_tokens = body["max_tokens"].get<int32_t>();
            if (request.max_tokens <= 0) {
                error = "max_tokens must be positive";
                return false;
            }
        }
        if (body.contains("max_completion_tokens")) {
            if (!body["max_completion_tokens"].is_number_integer()) {
                error = "max_completion_tokens must be an integer";
                return false;
            }
            request.max_tokens = body["max_completion_tokens"].get<int32_t>();
            if (request.max_tokens <= 0) {
                error = "max_completion_tokens must be positive";
                return false;
            }
        }
        if (body.contains("temperature")) {
            if (!body["temperature"].is_number()) {
                error = "temperature must be a number";
                return false;
            }
            request.temperature = body["temperature"].get<float>();
        }
        if (body.contains("top_p")) {
            if (!body["top_p"].is_number()) {
                error = "top_p must be a number";
                return false;
            }
            request.top_p = body["top_p"].get<float>();
            if (request.top_p <= 0.0f || request.top_p > 1.0f) {
                error = "top_p must be in the range (0, 1]";
                return false;
            }
        }
        if (body.contains("top_k")) {
            if (!body["top_k"].is_number_integer()) {
                error = "top_k must be an integer";
                return false;
            }
            request.top_k = body["top_k"].get<int32_t>();
            if (request.top_k < 0) {
                error = "top_k must be non-negative";
                return false;
            }
        }
        if (body.contains("seed")) {
            if (!body["seed"].is_number_unsigned()) {
                error = "seed must be an unsigned integer";
                return false;
            }
            request.seed = body["seed"].get<uint32_t>();
        }
        if (body.contains("stream")) {
            if (!body["stream"].is_boolean()) {
                error = "stream must be a boolean";
                return false;
            }
            request.stream = body["stream"].get<bool>();
        }
    } catch (const std::exception&) {
        error = "sampling parameter is out of range";
        return false;
    }
    return true;
}

} // namespace

bool parse_sampling_options(const Json& body, ChatRequest& request, std::string& error) {
    return parse_sampling_options_impl(body, request, error);
}

bool parse_openai_chat(const Json& body, ChatRequest& request, std::string& error) {
    request = ChatRequest{};
    error.clear();
    try {
        if (!body.is_object() || !body.contains("messages") || !body["messages"].is_array()) {
            error = "messages must be an array";
            return false;
        }
        request.model = body.value("model", "llm");
        if (!parse_sampling_options_impl(body, request, error)) return false;
        request.openai_messages = body["messages"];
        if (body.contains("tools")) {
            if (!body["tools"].is_array()) { error = "tools must be an array"; return false; }
            request.tools = body["tools"];
            request.structured_chat = !request.tools.empty();
            request.enable_thinking = request.tools.empty();
            for (const Json& tool : request.tools) {
                if (!tool.is_object() || tool.value("type", "") != "function" ||
                    !tool.contains("function") || !tool["function"].is_object() ||
                    !tool["function"].contains("name") || !tool["function"]["name"].is_string() ||
                    (tool["function"].contains("parameters") && !tool["function"]["parameters"].is_object())) {
                    error = "each tool must be an OpenAI function with a name and object parameters";
                    return false;
                }
            }
        }
        if (body.contains("parallel_tool_calls")) {
            if (!body["parallel_tool_calls"].is_boolean()) {
                error = "parallel_tool_calls must be a boolean"; return false;
            }
            request.parallel_tool_calls = body["parallel_tool_calls"].get<bool>();
        }
        if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object() &&
            body["chat_template_kwargs"].contains("enable_thinking")) {
            if (!body["chat_template_kwargs"]["enable_thinking"].is_boolean()) {
                error = "chat_template_kwargs.enable_thinking must be a boolean";
                return false;
            }
            request.enable_thinking = body["chat_template_kwargs"]["enable_thinking"].get<bool>();
        }
        if (body.contains("tool_choice")) {
            const Json& choice = body["tool_choice"];
            if (choice.is_string()) {
                request.tool_choice = choice.get<std::string>();
                if (request.tool_choice != "auto" && request.tool_choice != "required" &&
                    request.tool_choice != "none") {
                    error = "tool_choice must be auto, required, none, or a named function"; return false;
                }
            } else if (choice.is_object() && choice.value("type", "") == "function" &&
                       choice.contains("function") && choice["function"].is_object() &&
                       choice["function"].contains("name") && choice["function"]["name"].is_string()) {
                const std::string name = choice["function"]["name"].get<std::string>();
                Json selected = Json::array();
                for (const Json& tool : request.tools) {
                    if (tool.is_object() && tool.contains("function") && tool["function"].is_object() &&
                        tool["function"].value("name", "") == name) selected.push_back(tool);
                }
                if (selected.empty()) { error = "named tool_choice does not match any tool"; return false; }
                request.tools = std::move(selected);
                request.tool_choice = "required";
            } else {
                error = "invalid tool_choice"; return false;
            }
        }
        if (request.tool_choice == "required" && request.tools.empty()) {
            error = "tool_choice required needs at least one tool";
            return false;
        }
        for (const Json& item : body["messages"]) {
            if (!item.is_object() || !item.contains("role") || !item["role"].is_string() ||
                (!item.contains("content") && !item.contains("tool_calls"))) {
                error = "each message requires role and content or tool_calls";
                return false;
            }
            if (item.contains("tool_calls") && !valid_tool_calls(item["tool_calls"])) {
                error = "tool_calls must be an array of function calls";
                return false;
            }
            ChatMessage message;
            message.role = item["role"].get<std::string>();
            if (message.role == "tool" || item.contains("tool_calls")) request.structured_chat = true;
            if (item.contains("content") && !item["content"].is_null() &&
                !content_parts(item["content"], false, message, error)) {
                if (error.empty()) error = "invalid OpenAI message content";
                return false;
            }
            if (message.role == "system") {
                if (std::any_of(message.parts.begin(), message.parts.end(), [](const ChatMessage::Part& part) {
                        return part.type != ChatMessage::Part::Type::Text;
                    })) {
                    error = "system messages may contain text only";
                    return false;
                }
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
    } catch (const std::exception&) {
        error = "invalid OpenAI chat request";
        return false;
    }
}

bool parse_anthropic_messages(const Json& body, ChatRequest& request, std::string& error) {
    request = ChatRequest{};
    error.clear();
    try {
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
        if (!parse_sampling_options_impl(body, request, error)) return false;
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
    } catch (const std::exception&) {
        error = "invalid Anthropic messages request";
        return false;
    }
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
    const ChatRequest& request, const std::string& id, const Json& message
) {
    const bool called_tool = message.contains("tool_calls") && message["tool_calls"].is_array() &&
        !message["tool_calls"].empty();
    return {
        {"id", id}, {"object", "chat.completion"}, {"created", unix_seconds()},
        {"model", request.model},
        {"choices", Json::array({{{"index", 0},
            {"message", message},
            {"finish_reason", called_tool ? "tool_calls" : "stop"}}})},
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
