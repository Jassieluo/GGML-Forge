#include "client_common.h"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

int main(int argc, char** argv) try {
    using namespace forge::server_example;
    const Arguments args(argc, argv);
    Client client(args.get("base-url", "http://127.0.0.1:8080"), args.get("api-key"));
    const Json response = Json::parse(client.post_json("/forge/v1/videos/generations", {
        {"prompt", args.get("prompt", "ocean waves at sunset")},
        {"width", std::stoi(args.get("width", "256"))},
        {"height", std::stoi(args.get("height", "256"))},
        {"frame_count", std::stoi(args.get("frames", "8"))},
        {"fps", std::stoi(args.get("fps", "8"))},
        {"steps", std::stoi(args.get("steps", "8"))}
    }));
    const std::filesystem::path directory = args.get("output", "forge-video-frames");
    std::filesystem::create_directories(directory);
    const Json& frames = response["frames_b64"];
    for (size_t i = 0; i < frames.size(); ++i) {
        std::ostringstream name;
        name << "frame-" << std::setfill('0') << std::setw(4) << i << ".png";
        write_binary((directory / name.str()).string(),
            base64_decode(frames[i].get<std::string>()));
    }
    std::cout << "wrote " << frames.size() << " frames to " << directory.string() << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "server-video: " << error.what() << '\n';
    return 1;
}
