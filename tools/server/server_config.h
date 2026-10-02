#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>

namespace forge::server {

struct ServerConfig {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string api_key;
    std::string device = "auto";
    uint32_t threads = 4;
    uint32_t max_concurrency = 1;
    uint32_t llm_context = 4096;
    int32_t llm_gpu_layers = -1;

    std::string llm_model;
    std::string llm_mmproj;
    std::string asr_model;
    std::string tts_model;
    std::string visual_model;
};

void print_usage(std::ostream& output, const char* executable);
bool parse_server_config(int argc, char** argv, ServerConfig& config, std::string& error);

} // namespace forge::server
