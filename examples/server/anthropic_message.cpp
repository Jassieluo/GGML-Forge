#include "client_common.h"

#include <iostream>

int main(int argc, char** argv) try {
    using namespace forge::server_example;
    const Arguments args(argc, argv);
    Client client(args.get("base-url", "http://127.0.0.1:8080"), args.get("api-key"));
    const Json response = Json::parse(client.post_json("/v1/messages", {
        {"model", "llm"}, {"max_tokens", 128},
        {"messages", Json::array({{{"role", "user"},
            {"content", args.get("prompt", "Introduce yourself in one sentence.")}}})}
    }, {{"anthropic-version", "2023-06-01"}}));
    std::cout << response["content"][0]["text"].get<std::string>() << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "server-anthropic-message: " << error.what() << '\n';
    return 1;
}
