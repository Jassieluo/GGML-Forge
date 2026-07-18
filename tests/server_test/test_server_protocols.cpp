#include "codecs.h"
#include "protocols/chat.h"
#include "server_config.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

} // namespace

int main() {
    using namespace forge::server;
    const std::vector<uint8_t> bytes = {0, 1, 2, 127, 128, 255};
    std::vector<uint8_t> decoded;
    require(base64_decode(base64_encode(bytes.data(), bytes.size()), decoded) && decoded == bytes,
        "base64 round trip failed");

    const std::vector<float> samples = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    const std::string wav = encode_wav_pcm16(samples, 16000);
    std::vector<float> wav_samples;
    int32_t sample_rate = 0;
    require(decode_wav(wav, wav_samples, sample_rate) && sample_rate == 16000 &&
        wav_samples.size() == samples.size(), "WAV round trip failed");
    require(std::abs(wav_samples[3] - samples[3]) < 0.001f, "WAV sample mismatch");

    const std::vector<uint8_t> pixels = {255, 0, 0, 0, 255, 0};
    const std::vector<uint8_t> png = encode_png(pixels, 2, 1, 3);
    require(png.size() > 32 && png[0] == 137 && png[1] == 80 && png[2] == 78 && png[3] == 71,
        "PNG encoding failed");

    ChatRequest openai;
    std::string error;
    require(parse_openai_chat({
        {"model", "demo"}, {"messages", Json::array({
            {{"role", "system"}, {"content", "Be concise"}},
            {{"role", "user"}, {"content", "Hello"}}
        })}, {"stream", true}
    }, openai, error), "OpenAI chat parsing failed");
    require(openai.system == "Be concise" && openai.messages.size() == 1 && openai.stream,
        "OpenAI chat fields mismatch");

    const std::string data_url = "data:image/png;base64," + base64_encode(png.data(), png.size());
    ChatRequest multimodal;
    require(parse_openai_chat({
        {"messages", Json::array({{{"role", "user"}, {"content", Json::array({
            {{"type", "text"}, {"text", "describe"}},
            {{"type", "image_url"}, {"image_url", {{"url", data_url}}}}
        })}}})}
    }, multimodal, error), "OpenAI multimodal parsing failed");
    require(multimodal.messages[0].parts.size() == 2 &&
        multimodal.messages[0].parts[1].data == png, "OpenAI image payload mismatch");

    ChatRequest anthropic;
    require(parse_anthropic_messages({
        {"model", "demo"}, {"max_tokens", 32},
        {"system", Json::array({{{"type", "text"}, {"text", "Helpful"}}})},
        {"messages", Json::array({{{"role", "user"}, {"content", Json::array({
            {{"type", "text"}, {"text", "Hello"}}
        })}}})}
    }, anthropic, error), "Anthropic parsing failed");
    require(anthropic.system == "Helpful" && anthropic.max_tokens == 32,
        "Anthropic fields mismatch");

    char executable[] = "forge-server";
    char model_option[] = "--llm-model";
    char model[] = "model.gguf";
    char gpu_option[] = "--llm-gpu-layers";
    char gpu[] = "-1";
    char* arguments[] = {executable, model_option, model, gpu_option, gpu};
    ServerConfig config;
    require(parse_server_config(5, arguments, config, error) && config.llm_gpu_layers == -1,
        "server config parsing failed");
    return 0;
}
