#include "client_common.h"

#include <iostream>

int main(int argc, char** argv) try {
    using namespace forge::server_example;
    const Arguments args(argc, argv);
    const std::string audio_path = args.require("audio");
    const std::vector<uint8_t> audio = read_binary(audio_path);
    Client client(args.get("base-url", "http://127.0.0.1:8080"), args.get("api-key"));
    const httplib::UploadFormDataItems form = {
        {"file", std::string(reinterpret_cast<const char*>(audio.data()), audio.size()),
            "audio.wav", "audio/wav"},
        {"language", args.get("language", "auto"), "", ""},
        {"response_format", "json", "", ""},
    };
    const Json response = Json::parse(client.post_multipart("/v1/audio/transcriptions", form));
    std::cout << response["text"].get<std::string>() << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "server-transcribe: " << error.what() << '\n';
    return 1;
}
