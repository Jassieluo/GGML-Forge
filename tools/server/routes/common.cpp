#include "routes/common.h"

#include <exception>

namespace forge::server {

bool parse_json_body(
    const httplib::Request& request, Json& body, httplib::Response& response
) {
    try {
        body = Json::parse(request.body);
        return true;
    } catch (const std::exception& exception) {
        error_response(response, 400, std::string("invalid JSON: ") + exception.what());
        return false;
    }
}

void json_response(httplib::Response& response, const Json& body, int status) {
    response.status = status;
    response.set_content(body.dump(), "application/json; charset=utf-8");
}

void error_response(
    httplib::Response& response,
    int status,
    const std::string& message,
    const std::string& type
) {
    json_response(response, {{"error", {{"message", message}, {"type", type}, {"code", nullptr}}}}, status);
}

void register_common_routes(httplib::Server& server, ModelRegistry& registry) {
    server.Get("/health", [&](const httplib::Request&, httplib::Response& response) {
        json_response(response, {{"status", "ok"}, {"models", registry.models().size()}});
    });
    server.Get("/v1/models", [&](const httplib::Request&, httplib::Response& response) {
        Json data = Json::array();
        for (const ServedModel& model : registry.models()) {
            data.push_back({
                {"id", model.id}, {"object", "model"}, {"created", 0},
                {"owned_by", "ggml-forge"}, {"category", model.category},
                {"provider", model.provider}
            });
        }
        json_response(response, {{"object", "list"}, {"data", std::move(data)}});
    });
    server.Get("/forge/v1/capabilities", [&](const httplib::Request&, httplib::Response& response) {
        Json features = Json::object();
#if FORGE_SERVER_HAS_VISUAL
        const visual_capabilities visual = registry.get_visual_capabilities();
        features["visual_generation"] = {
            {"text_to_image", visual.text_to_image}, {"image_to_image", visual.image_to_image},
            {"video", visual.video}, {"video_audio", visual.video_audio},
            {"upscale", visual.upscale}, {"cancellation", visual.cancellation}
        };
#endif
        json_response(response, {
            {"protocols", Json::array({"openai", "anthropic", "forge"})},
            {"categories", {
                {"llm", registry.has_category("llm")},
                {"asr", registry.has_category("asr")},
                {"tts", registry.has_category("tts")},
                {"visual_generation", registry.has_category("visual_generation")}
            }},
            {"features", std::move(features)},
            {"endpoints", Json::array({
                "/v1/chat/completions", "/v1/completions", "/v1/responses",
                "/v1/messages", "/v1/audio/transcriptions", "/v1/audio/speech",
                "/v1/images/generations", "/forge/v1/videos/generations"
            })}
        });
    });
}

} // namespace forge::server
