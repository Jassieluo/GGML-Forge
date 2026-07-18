#pragma once

#include "json.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace forge::server {

using Json = nlohmann::ordered_json;

struct ChatMessage {
    struct Part {
        enum class Type { Text, Image, Audio } type = Type::Text;
        std::string text;
        std::vector<uint8_t> data;
        std::string mime_type;
    };
    std::string role;
    std::string content;
    std::vector<Part> parts;
};

struct ChatRequest {
    std::string model = "llm";
    std::string system;
    std::vector<ChatMessage> messages;
    int32_t max_tokens = 256;
    float temperature = 0.8f;
    int32_t top_k = 40;
    float top_p = 0.95f;
    uint32_t seed = 0;
    bool stream = false;
};

bool parse_openai_chat(const Json& body, ChatRequest& request, std::string& error);
bool parse_anthropic_messages(const Json& body, ChatRequest& request, std::string& error);
std::string request_id(const char* prefix);
int64_t unix_seconds();

Json openai_chat_response(
    const ChatRequest& request,
    const std::string& id,
    const std::string& content);
Json anthropic_message_response(
    const ChatRequest& request,
    const std::string& id,
    const std::string& content);

} // namespace forge::server
