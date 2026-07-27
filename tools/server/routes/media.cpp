#include "routes/media.h"

#include "audio/audio_io.h"
#include "codecs.h"
#include "protocols/chat.h"
#include "routes/common.h"

#include <algorithm>
#include <cstdlib>

namespace forge::server {
namespace {

bool decode_base64_wav(const std::string& encoded, SpeechReference& reference) {
    const size_t comma = encoded.find(',');
    const std::string payload = encoded.rfind("data:", 0) == 0 && comma != std::string::npos
        ? encoded.substr(comma + 1) : encoded;
    std::vector<uint8_t> bytes;
    if (!base64_decode(payload, bytes)) return false;
    uint32_t sample_rate = 0;
    std::string error;
    if (!forge::media::decode_wav_mono(
            bytes.data(), bytes.size(), reference.audio, sample_rate, error)) return false;
    reference.sample_rate = static_cast<int32_t>(sample_rate);
    return true;
}

} // namespace

void register_media_routes(httplib::Server& server, ModelRegistry& registry) {
#if FORGE_SERVER_HAS_ASR
    if (registry.has_category("asr")) {
        server.Post("/v1/audio/transcriptions", [&](const httplib::Request& http, httplib::Response& response) {
            if (!http.form.has_file("file")) { error_response(response, 400, "multipart field 'file' is required"); return; }
            const auto file = http.form.get_file("file");
            std::vector<float> audio;
            uint32_t decoded_rate = 0;
            std::string media_error;
            if (!forge::media::decode_wav_mono(
                    reinterpret_cast<const uint8_t*>(file.content.data()), file.content.size(),
                    audio, decoded_rate, media_error)) {
                error_response(response, 400, "only PCM16 or float32 WAV uploads are currently supported");
                return;
            }
            const int32_t sample_rate = static_cast<int32_t>(decoded_rate);
            const std::string language = http.form.has_field("language") ? http.form.get_field("language") : "auto";
            const std::string format = http.form.has_field("response_format") ?
                http.form.get_field("response_format") : "json";
            asr_request_params params = asr_request_default_params();
            params.language = language.c_str();
            std::string text;
            std::string detected_language;
            std::string error;
            const bool ok = registry.transcribe(audio, sample_rate, params, [&](const asr_event& event) {
                if (event.type == ASR_EVENT_LANGUAGE && event.language) detected_language = event.language;
                if (event.type == ASR_EVENT_SEGMENT && event.text) text.append(event.text, event.text_length);
                return true;
            }, error);
            if (!ok) { error_response(response, 500, error, "server_error"); return; }
            if (format == "text") response.set_content(text, "text/plain; charset=utf-8");
            else json_response(response, {{"text", text}, {"language", detected_language}});
        });
    }
#endif

#if FORGE_SERVER_HAS_TTS
    if (registry.has_category("tts")) {
        server.Post("/v1/audio/speech", [&](const httplib::Request& http, httplib::Response& response) {
            Json body;
            if (!parse_json_body(http, body, response)) return;
            if (!body.contains("input") || !body["input"].is_string()) {
                error_response(response, 400, "input must be a string"); return;
            }
            const std::string text = body["input"].get<std::string>();
            const std::string language = body.value("language", "auto");
            const float speed = body.value("speed", 1.0f);
            SpeechReference reference;
            const SpeechReference* reference_ptr = nullptr;
            if (body.contains("reference_audio")) {
                if (!body["reference_audio"].is_string() ||
                    !decode_base64_wav(body["reference_audio"].get<std::string>(), reference)) {
                    error_response(response, 400, "reference_audio must contain a base64 PCM WAV"); return;
                }
                reference.text = body.value("reference_text", "");
                reference.language = body.value("reference_language", language);
                reference_ptr = &reference;
            }
            std::vector<float> audio;
            int32_t sample_rate = 0;
            std::string error;
            if (!registry.synthesize(text, language, speed, reference_ptr, audio, sample_rate, error)) {
                error_response(response, 500, error, "server_error"); return;
            }
            std::vector<uint8_t> wav;
            if (sample_rate <= 0 || !forge::media::encode_wav_pcm16_mono(
                    audio.data(), audio.size(), static_cast<uint32_t>(sample_rate),
                    wav, error)) {
                error_response(response, 500, "failed to encode WAV", "server_error"); return;
            }
            response.set_content(
                std::string(reinterpret_cast<const char*>(wav.data()), wav.size()), "audio/wav");
        });
    }
#endif

#if FORGE_SERVER_HAS_VISUAL
    if (registry.has_category("visual_generation")) {
        server.Post("/v1/images/generations", [&](const httplib::Request& http, httplib::Response& response) {
            if (!registry.get_visual_capabilities().text_to_image) {
                error_response(response, 400, "loaded visual model does not support image generation"); return;
            }
            Json body;
            if (!parse_json_body(http, body, response)) return;
            if (!body.contains("prompt") || !body["prompt"].is_string()) {
                error_response(response, 400, "prompt must be a string"); return;
            }
            visual_image_request request = visual_image_request_default_params();
            const std::string prompt = body["prompt"].get<std::string>();
            const std::string negative = body.value("negative_prompt", "");
            request.prompt = prompt.c_str();
            request.negative_prompt = negative.c_str();
            request.seed = body.value("seed", static_cast<int64_t>(-1));
            request.sample.steps = body.value("steps", request.sample.steps);
            request.sample.text_guidance = body.value("guidance_scale", request.sample.text_guidance);
            const std::string size = body.value("size", "512x512");
            const size_t separator = size.find('x');
            if (separator != std::string::npos) {
                request.width = std::max(64, std::atoi(size.substr(0, separator).c_str()));
                request.height = std::max(64, std::atoi(size.substr(separator + 1).c_str()));
            }
            std::vector<uint8_t> pixels;
            uint32_t width = 0, height = 0, channels = 0;
            std::string error;
            if (!registry.generate_image(request, pixels, width, height, channels, error)) {
                error_response(response, 500, error, "server_error"); return;
            }
            const std::vector<uint8_t> png = encode_png(pixels, width, height, channels);
            if (png.empty()) { error_response(response, 500, "failed to encode PNG", "server_error"); return; }
            const std::string response_format = body.value("response_format", "b64_json");
            Json image;
            if (response_format == "url") {
                image["url"] = "data:image/png;base64," + base64_encode(png.data(), png.size());
            } else {
                image["b64_json"] = base64_encode(png.data(), png.size());
            }
            image["revised_prompt"] = prompt;
            json_response(response, {{"created", unix_seconds()}, {"data", Json::array({image})}});
        });
        server.Post("/forge/v1/videos/generations", [&](const httplib::Request& http, httplib::Response& response) {
            if (!registry.get_visual_capabilities().video) {
                error_response(response, 400, "loaded visual model does not support video generation"); return;
            }
            Json body;
            if (!parse_json_body(http, body, response)) return;
            if (!body.contains("prompt") || !body["prompt"].is_string()) {
                error_response(response, 400, "prompt must be a string"); return;
            }
            visual_video_request request = visual_video_request_default_params();
            const std::string prompt = body["prompt"].get<std::string>();
            const std::string negative = body.value("negative_prompt", "");
            request.prompt = prompt.c_str();
            request.negative_prompt = negative.c_str();
            request.width = body.value("width", request.width);
            request.height = body.value("height", request.height);
            request.frame_count = body.value("frame_count", request.frame_count);
            request.fps = body.value("fps", request.fps);
            request.seed = body.value("seed", static_cast<int64_t>(-1));
            request.sample.steps = body.value("steps", request.sample.steps);
            request.sample.text_guidance = body.value("guidance_scale", request.sample.text_guidance);
            std::vector<ModelRegistry::VideoFrame> frames;
            uint32_t fps = 0;
            std::string error;
            if (!registry.generate_video(request, frames, fps, error)) {
                error_response(response, 500, error, "server_error"); return;
            }
            Json encoded = Json::array();
            for (const auto& frame : frames) {
                const std::vector<uint8_t> png = encode_png(
                    frame.pixels, frame.width, frame.height, frame.channels);
                if (png.empty()) { error_response(response, 500, "failed to encode video frame", "server_error"); return; }
                encoded.push_back(base64_encode(png.data(), png.size()));
            }
            json_response(response, {
                {"id", request_id("video_")}, {"status", "completed"},
                {"fps", fps}, {"frame_count", encoded.size()},
                {"mime_type", "image/png"}, {"frames_b64", std::move(encoded)}
            });
        });
    }
#endif
}

} // namespace forge::server
