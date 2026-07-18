#include "client_common.h"

#include <iostream>

int main(int argc, char** argv) try {
    using namespace forge::server_example;
    const Arguments args(argc, argv);
    Json request = {
        {"model", "tts"},
        {"input", args.get("text", "Hello from the GGML Forge speech service.")},
        {"language", args.get("language", "en")}, {"speed", 1.0}
    };
    const std::string reference = args.get("reference-audio");
    if (!reference.empty()) {
        request["reference_audio"] = "data:audio/wav;base64," + base64_encode(read_binary(reference));
        request["reference_text"] = args.require("reference-text");
        request["reference_language"] = args.get("reference-language", args.get("language", "en"));
    }
    Client client(args.get("base-url", "http://127.0.0.1:8080"), args.get("api-key"));
    const std::string output = args.get("output", "forge-speech.wav");
    write_binary(output, client.post_json("/v1/audio/speech", request));
    std::cout << "wrote " << output << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "server-speech: " << error.what() << '\n';
    return 1;
}
