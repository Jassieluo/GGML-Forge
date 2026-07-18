#include "client_common.h"

#include <iostream>

int main(int argc, char** argv) try {
    using namespace forge::server_example;
    const Arguments args(argc, argv);
    Client client(args.get("base-url", "http://127.0.0.1:8080"), args.get("api-key"));
    const Json response = Json::parse(client.post_json("/v1/images/generations", {
        {"model", "visual"},
        {"prompt", args.get("prompt", "a small red fox in a snowy forest, detailed illustration")},
        {"size", args.get("size", "256x256")}, {"steps", std::stoi(args.get("steps", "8"))},
        {"response_format", "b64_json"}
    }));
    const std::string output = args.get("output", "forge-image.png");
    write_binary(output, base64_decode(response["data"][0]["b64_json"].get<std::string>()));
    std::cout << "wrote " << output << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "server-image: " << error.what() << '\n';
    return 1;
}
