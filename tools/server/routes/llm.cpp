#include "routes/llm.h"

#include "protocols/chat.h"
#include "routes/common.h"

#include <memory>

namespace forge::server {
namespace {

#if FORGE_SERVER_HAS_LLM
llm_generation_params generation_params(ModelRegistry& registry, const ChatRequest& request) {
    llm_generation_params params = registry.generation_defaults();
    params.max_tokens = request.max_tokens;
    params.temperature = request.temperature;
    params.top_k = request.top_k;
    params.top_p = request.top_p;
    params.seed = request.seed;
    return params;
}

std::vector<ChatMessage> provider_messages(const ChatRequest& request) {
    std::vector<ChatMessage> messages;
    messages.reserve(request.messages.size() + (request.system.empty() ? 0 : 1));
    if (!request.system.empty()) messages.push_back({"system", request.system});
    messages.insert(messages.end(), request.messages.begin(), request.messages.end());
    return messages;
}

void openai_stream(
    httplib::Response& response,
    ModelRegistry& registry,
    ChatRequest request,
    std::string id
) {
    response.set_header("Cache-Control", "no-cache");
    response.set_header("X-Accel-Buffering", "no");
    auto started = std::make_shared<bool>(false);
    response.set_chunked_content_provider(
        "text/event-stream",
        [&, request = std::move(request), id = std::move(id), started](size_t, httplib::DataSink& sink) mutable {
            if (*started) { sink.done(); return false; }
            *started = true;
            std::string error;
            const Json role = {
                {"id", id}, {"object", "chat.completion.chunk"}, {"created", unix_seconds()},
                {"model", request.model},
                {"choices", Json::array({{{"index", 0}, {"delta", {{"role", "assistant"}}},
                    {"finish_reason", nullptr}}})}
            };
            std::string event = "data: " + role.dump() + "\n\n";
            if (!sink.write(event.data(), event.size())) return false;
            const bool ok = registry.generate_chat(
                provider_messages(request), generation_params(registry, request),
                [&](const char* text, size_t length) {
                    Json chunk = {
                        {"id", id}, {"object", "chat.completion.chunk"}, {"created", unix_seconds()},
                        {"model", request.model},
                        {"choices", Json::array({{{"index", 0},
                            {"delta", {{"content", std::string(text, length)}}},
                            {"finish_reason", nullptr}}})}
                    };
                    const std::string line = "data: " + chunk.dump() + "\n\n";
                    return sink.write(line.data(), line.size());
                }, error);
            if (ok) {
                Json final_chunk = {
                    {"id", id}, {"object", "chat.completion.chunk"}, {"created", unix_seconds()},
                    {"model", request.model},
                    {"choices", Json::array({{{"index", 0}, {"delta", Json::object()},
                        {"finish_reason", "stop"}}})}
                };
                const std::string ending = "data: " + final_chunk.dump() + "\n\ndata: [DONE]\n\n";
                sink.write(ending.data(), ending.size());
            }
            sink.done();
            return false;
        });
}

void anthropic_stream(
    httplib::Response& response,
    ModelRegistry& registry,
    ChatRequest request,
    std::string id
) {
    response.set_header("Cache-Control", "no-cache");
    response.set_header("X-Accel-Buffering", "no");
    auto started = std::make_shared<bool>(false);
    response.set_chunked_content_provider(
        "text/event-stream",
        [&, request = std::move(request), id = std::move(id), started](size_t, httplib::DataSink& sink) mutable {
            if (*started) { sink.done(); return false; }
            *started = true;
            auto send = [&](const char* name, const Json& data) {
                const std::string event = std::string("event: ") + name + "\ndata: " + data.dump() + "\n\n";
                return sink.write(event.data(), event.size());
            };
            send("message_start", {{"type", "message_start"}, {"message", {
                {"id", id}, {"type", "message"}, {"role", "assistant"},
                {"model", request.model}, {"content", Json::array()},
                {"stop_reason", nullptr}, {"usage", {{"input_tokens", 0}, {"output_tokens", 0}}}
            }}});
            send("content_block_start", {{"type", "content_block_start"}, {"index", 0},
                {"content_block", {{"type", "text"}, {"text", ""}}}});
            std::string error;
            const bool ok = registry.generate_chat(
                provider_messages(request), generation_params(registry, request),
                [&](const char* text, size_t length) {
                    return send("content_block_delta", {{"type", "content_block_delta"}, {"index", 0},
                        {"delta", {{"type", "text_delta"}, {"text", std::string(text, length)}}}});
                }, error);
            if (ok) {
                send("content_block_stop", {{"type", "content_block_stop"}, {"index", 0}});
                send("message_delta", {{"type", "message_delta"},
                    {"delta", {{"stop_reason", "end_turn"}, {"stop_sequence", nullptr}}},
                    {"usage", {{"output_tokens", 0}}}});
                send("message_stop", {{"type", "message_stop"}});
            }
            sink.done();
            return false;
        });
}

void completion_stream(
    httplib::Response& response,
    ModelRegistry& registry,
    std::string prompt,
    ChatRequest request,
    std::string id
) {
    response.set_header("Cache-Control", "no-cache");
    response.set_header("X-Accel-Buffering", "no");
    auto started = std::make_shared<bool>(false);
    response.set_chunked_content_provider(
        "text/event-stream",
        [&, prompt = std::move(prompt), request = std::move(request), id = std::move(id), started]
        (size_t, httplib::DataSink& sink) mutable {
            if (*started) { sink.done(); return false; }
            *started = true;
            std::string error;
            const bool ok = registry.generate(prompt, generation_params(registry, request),
                [&](const char* text, size_t length) {
                    Json chunk = {
                        {"id", id}, {"object", "text_completion"}, {"created", unix_seconds()},
                        {"model", request.model},
                        {"choices", Json::array({{{"index", 0}, {"text", std::string(text, length)},
                            {"finish_reason", nullptr}}})}
                    };
                    const std::string line = "data: " + chunk.dump() + "\n\n";
                    return sink.write(line.data(), line.size());
                }, error);
            if (ok) {
                Json final_chunk = {
                    {"id", id}, {"object", "text_completion"}, {"created", unix_seconds()},
                    {"model", request.model},
                    {"choices", Json::array({{{"index", 0}, {"text", ""}, {"finish_reason", "stop"}}})}
                };
                const std::string ending = "data: " + final_chunk.dump() + "\n\ndata: [DONE]\n\n";
                sink.write(ending.data(), ending.size());
            }
            sink.done();
            return false;
        });
}
#endif

} // namespace

void register_llm_routes(httplib::Server& server, ModelRegistry& registry) {
#if FORGE_SERVER_HAS_LLM
    if (!registry.has_category("llm")) return;
    server.Post("/v1/chat/completions", [&](const httplib::Request& http, httplib::Response& response) {
        Json body;
        if (!parse_json_body(http, body, response)) return;
        ChatRequest request;
        std::string error;
        if (!parse_openai_chat(body, request, error)) { error_response(response, 400, error); return; }
        const std::string id = request_id("chatcmpl-");
        if (request.stream) { openai_stream(response, registry, std::move(request), id); return; }
        std::string content;
        if (!registry.generate_chat(provider_messages(request), generation_params(registry, request),
                [&](const char* text, size_t length) { content.append(text, length); return true; }, error)) {
            error_response(response, 500, error, "server_error"); return;
        }
        json_response(response, openai_chat_response(request, id, content));
    });
    server.Post("/v1/messages", [&](const httplib::Request& http, httplib::Response& response) {
        Json body;
        if (!parse_json_body(http, body, response)) return;
        ChatRequest request;
        std::string error;
        if (!parse_anthropic_messages(body, request, error)) { error_response(response, 400, error); return; }
        const std::string id = request_id("msg_");
        if (request.stream) { anthropic_stream(response, registry, std::move(request), id); return; }
        std::string content;
        if (!registry.generate_chat(provider_messages(request), generation_params(registry, request),
                [&](const char* text, size_t length) { content.append(text, length); return true; }, error)) {
            error_response(response, 500, error, "api_error"); return;
        }
        json_response(response, anthropic_message_response(request, id, content));
    });
    server.Post("/v1/completions", [&](const httplib::Request& http, httplib::Response& response) {
        Json body;
        if (!parse_json_body(http, body, response)) return;
        if (!body.contains("prompt") || !body["prompt"].is_string()) {
            error_response(response, 400, "prompt must be a string"); return;
        }
        ChatRequest request;
        request.model = body.value("model", "llm");
        request.max_tokens = body.value("max_tokens", 256);
        request.temperature = body.value("temperature", 0.8f);
        request.top_p = body.value("top_p", 0.95f);
        request.top_k = body.value("top_k", 40);
        request.seed = body.value("seed", 0u);
        request.stream = body.value("stream", false);
        const std::string prompt = body["prompt"].get<std::string>();
        const std::string id = request_id("cmpl-");
        if (request.stream) { completion_stream(response, registry, prompt, std::move(request), id); return; }
        std::string content;
        std::string error;
        if (!registry.generate(prompt, generation_params(registry, request),
                [&](const char* text, size_t length) { content.append(text, length); return true; }, error)) {
            error_response(response, 500, error, "server_error"); return;
        }
        json_response(response, {
            {"id", id}, {"object", "text_completion"}, {"created", unix_seconds()},
            {"model", request.model},
            {"choices", Json::array({{{"index", 0}, {"text", content}, {"finish_reason", "stop"}}})},
            {"usage", {{"prompt_tokens", 0}, {"completion_tokens", 0}, {"total_tokens", 0}}}
        });
    });
    server.Post("/v1/responses", [&](const httplib::Request& http, httplib::Response& response) {
        Json body;
        if (!parse_json_body(http, body, response)) return;
        if (body.value("stream", false)) {
            error_response(response, 400, "streaming Responses API is not implemented yet"); return;
        }
        ChatRequest request;
        request.model = body.value("model", "llm");
        request.max_tokens = body.value("max_output_tokens", 256);
        request.temperature = body.value("temperature", 0.8f);
        if (!body.contains("input")) { error_response(response, 400, "input is required"); return; }
        if (body["input"].is_string()) {
            request.messages.push_back({"user", body["input"].get<std::string>(), {}});
        } else if (body["input"].is_array()) {
            Json wrapper = body;
            wrapper["messages"] = body["input"];
            std::string parse_error;
            if (!parse_openai_chat(wrapper, request, parse_error)) {
                error_response(response, 400, parse_error); return;
            }
        } else {
            error_response(response, 400, "input must be text or an array of messages"); return;
        }
        const std::string id = request_id("resp_");
        std::string content;
        std::string error;
        if (!registry.generate_chat(provider_messages(request), generation_params(registry, request),
                [&](const char* text, size_t length) { content.append(text, length); return true; }, error)) {
            error_response(response, 500, error, "server_error"); return;
        }
        json_response(response, {
            {"id", id}, {"object", "response"}, {"created_at", unix_seconds()},
            {"status", "completed"}, {"model", request.model},
            {"output", Json::array({{{"id", request_id("msg_")}, {"type", "message"},
                {"status", "completed"}, {"role", "assistant"},
                {"content", Json::array({{{"type", "output_text"}, {"text", content},
                    {"annotations", Json::array()}}})}}})},
            {"usage", {{"input_tokens", 0}, {"output_tokens", 0}, {"total_tokens", 0}}}
        });
    });
#else
    (void)server; (void)registry;
#endif
}

} // namespace forge::server
