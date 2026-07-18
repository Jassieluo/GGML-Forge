#include "server_config.h"

#include <cstdlib>
#include <iostream>
#include <limits>

namespace forge::server {

void print_usage(std::ostream& output, const char* executable) {
    output
        << "Usage: " << executable << " [options]\n"
        << "  --host <address>          Listen address (default: 127.0.0.1)\n"
        << "  --port <number>           Listen port (default: 8080)\n"
        << "  --api-key <key>           Optional Bearer/x-api-key authentication\n"
        << "  --device <name>           auto, cpu, CUDA0, or another backend device\n"
        << "  --threads <count>         CPU worker count\n"
        << "  --max-concurrency <count> TTS runtime execution lanes\n"
        << "  --llm-gpu-layers <count>  LLM layers offloaded to GPU (-1 means all)\n"
        << "  --llm-model <gguf>        LLM model\n"
        << "  --llm-mmproj <gguf>       Optional multimodal projector\n"
        << "  --asr-model <bin>         ASR model\n"
        << "  --tts-model <json>        TTS model composition\n"
        << "  --visual-model <path>     Visual generation model\n";
}

bool parse_server_config(int argc, char** argv, ServerConfig& config, std::string& error) {
    auto parse_uint = [&](const char* text, uint32_t& value) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(text, &end, 10);
        if (!text[0] || !end || *end || parsed == 0 ||
            parsed > std::numeric_limits<uint32_t>::max()) return false;
        value = static_cast<uint32_t>(parsed);
        return true;
    };
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        auto value = [&]() -> const char* {
            if (i + 1 >= argc) return nullptr;
            return argv[++i];
        };
        if (option == "--help" || option == "-h") {
            print_usage(std::cout, argv[0]);
            return false;
        } else if (option == "--host") {
            const char* next = value(); if (!next) { error = "--host requires a value"; return false; }
            config.host = next;
        } else if (option == "--port") {
            const char* next = value(); uint32_t parsed = 0;
            if (!next || !parse_uint(next, parsed) || parsed > 65535) { error = "invalid --port"; return false; }
            config.port = static_cast<int>(parsed);
        } else if (option == "--api-key") {
            const char* next = value(); if (!next) { error = "--api-key requires a value"; return false; }
            config.api_key = next;
        } else if (option == "--device") {
            const char* next = value(); if (!next) { error = "--device requires a value"; return false; }
            config.device = next;
        } else if (option == "--threads") {
            const char* next = value();
            if (!next || !parse_uint(next, config.threads)) { error = "invalid --threads"; return false; }
        } else if (option == "--max-concurrency") {
            const char* next = value();
            if (!next || !parse_uint(next, config.max_concurrency)) { error = "invalid --max-concurrency"; return false; }
        } else if (option == "--llm-gpu-layers") {
            const char* next = value();
            if (!next) { error = "--llm-gpu-layers requires a value"; return false; }
            char* end = nullptr;
            const long parsed = std::strtol(next, &end, 10);
            if (!next[0] || !end || *end || parsed < -1 || parsed > std::numeric_limits<int32_t>::max()) {
                error = "invalid --llm-gpu-layers"; return false;
            }
            config.llm_gpu_layers = static_cast<int32_t>(parsed);
        } else if (option == "--llm-model" || option == "--llm-mmproj" ||
                   option == "--asr-model" || option == "--tts-model" ||
                   option == "--visual-model") {
            const char* next = value(); if (!next) { error = option + " requires a value"; return false; }
            if (option == "--llm-model") config.llm_model = next;
            else if (option == "--llm-mmproj") config.llm_mmproj = next;
            else if (option == "--asr-model") config.asr_model = next;
            else if (option == "--tts-model") config.tts_model = next;
            else config.visual_model = next;
        } else {
            error = "unknown option: " + option;
            return false;
        }
    }
    if (config.llm_model.empty() && config.asr_model.empty() &&
        config.tts_model.empty() && config.visual_model.empty()) {
        error = "at least one model option is required";
        return false;
    }
    return true;
}

} // namespace forge::server
