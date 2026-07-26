#pragma once

// Text-protocol tool calls for the chat agent. The LLM is instructed (via
// system prompt in pages/actions.h) to emit
//   <tool_call>{"name":"...","arguments":{...}}</tool_call>
// This header extracts and strips such calls from a finished reply. Parsing is
// deliberately minimal: flat string/number fields with \\ and \" escapes,
// which covers everything a 4B local model emits for these two tools.

#include <algorithm>
#include <cctype>
#include <string>

namespace app {

struct ToolCall {
    std::string name;       // "generate_image" | "text_to_speech"
    std::string prompt;     // generate_image
    int steps = 0;          // generate_image, 0 = keep current UI value
    std::string text;       // text_to_speech
    float speed = 0.0f;     // text_to_speech, 0 = keep current UI value
};

namespace tool_calls {

// Splits a Qwen-style <think>...</think> prefix off `reply`. Returns the
// reasoning text (empty when absent); `reply` is left holding the visible
// answer. An unterminated block (cancel mid-think) becomes all reasoning.
inline std::string stripThinking(std::string& reply) {
    size_t start = 0;
    while (start < reply.size() &&
           std::isspace(static_cast<unsigned char>(reply[start]))) {
        ++start;
    }
    if (reply.compare(start, 7, "<think>") != 0) return {};

    const size_t body = start + 7;
    const size_t close = reply.find("</think>", body);
    std::string thinking = reply.substr(body, close == std::string::npos
                                                  ? std::string::npos
                                                  : close - body);
    std::string answer = close == std::string::npos ? std::string{}
                                                    : reply.substr(close + 8);
    const auto trim = [](std::string& value) {
        size_t begin = 0;
        while (begin < value.size() &&
               std::isspace(static_cast<unsigned char>(value[begin]))) ++begin;
        size_t end = value.size();
        while (end > begin &&
               std::isspace(static_cast<unsigned char>(value[end - 1]))) --end;
        value = value.substr(begin, end - begin);
    };
    trim(thinking);
    trim(answer);
    reply = std::move(answer);
    return thinking;
}

inline std::string extractString(const std::string& json, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    size_t at = json.find(needle);
    if (at == std::string::npos) return {};
    at = json.find(':', at + needle.size());
    if (at == std::string::npos) return {};
    at = json.find('"', at + 1);
    if (at == std::string::npos) return {};
    std::string value;
    for (size_t i = at + 1; i < json.size(); ++i) {
        const char c = json[i];
        if (c == '\\' && i + 1 < json.size()) {
            const char next = json[++i];
            if (next == 'n') value += '\n';
            else if (next == 't') value += '\t';
            else value += next;  // \" \\ / and anything exotic verbatim
        } else if (c == '"') {
            return value;
        } else {
            value += c;
        }
    }
    return value;
}

inline double extractNumber(const std::string& json, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    size_t at = json.find(needle);
    if (at == std::string::npos) return 0.0;
    at = json.find(':', at + needle.size());
    if (at == std::string::npos) return 0.0;
    ++at;
    while (at < json.size() && std::isspace(static_cast<unsigned char>(json[at]))) ++at;
    // Numbers only; a quoted value here would belong to extractString.
    size_t end = at;
    while (end < json.size() &&
           (std::isdigit(static_cast<unsigned char>(json[end])) ||
            json[end] == '.' || json[end] == '-' || json[end] == '+')) {
        ++end;
    }
    if (end == at) return 0.0;
    try {
        return std::stod(json.substr(at, end - at));
    } catch (...) {
        return 0.0;
    }
}

// Returns true when `reply` contains a tool call. `cleaned` receives the reply
// with the call block removed (the model's visible commentary, if any).
inline bool parse(const std::string& reply, ToolCall& call, std::string& cleaned) {
    const size_t open = reply.find("<tool_call>");
    if (open == std::string::npos) {
        cleaned = reply;
        return false;
    }
    size_t close = reply.find("</tool_call>", open);
    size_t body_end = close == std::string::npos ? reply.size() : close;
    const std::string body = reply.substr(open + 11, body_end - open - 11);

    call = ToolCall{};
    call.name = extractString(body, "name");
    call.prompt = extractString(body, "prompt");
    call.text = extractString(body, "text");
    call.steps = static_cast<int>(extractNumber(body, "steps"));
    call.speed = static_cast<float>(extractNumber(body, "speed"));

    const bool valid = call.name == "generate_image" ? !call.prompt.empty()
                     : call.name == "text_to_speech" ? !call.text.empty()
                                                     : false;
    if (!valid) {
        cleaned = reply;  // Keep the malformed block visible for debugging.
        return false;
    }

    cleaned = reply.substr(0, open);
    if (close != std::string::npos) cleaned += reply.substr(close + 12);
    // Trim whitespace left behind by the removed block.
    const auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!cleaned.empty() && is_space(cleaned.back())) cleaned.pop_back();
    size_t start = 0;
    while (start < cleaned.size() && is_space(static_cast<unsigned char>(cleaned[start]))) ++start;
    cleaned.erase(0, start);
    return true;
}

} // namespace tool_calls
} // namespace app
