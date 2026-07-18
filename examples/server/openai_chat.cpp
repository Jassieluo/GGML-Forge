#include "client_common.h"

#include <iostream>

int main(int argc, char** argv) try {
    using namespace forge::server_example;
    const Arguments args(argc, argv);
    Json content = Json::array({{{"type", "text"},
        {"text", args.get("prompt", "Introduce yourself in one sentence.")}}});
    const std::string image = args.get("image");
    if (!image.empty()) {
        content.push_back({
            {"type", "image_url"},
            {"image_url", {{"url", "data:" + image_mime_type(image) +
                ";base64," + base64_encode(read_binary(image))}}}
        });
    }
    Client client(args.get("base-url", "http://127.0.0.1:8080"), args.get("api-key"));
    const Json response = Json::parse(client.post_json("/v1/chat/completions", {
        {"model", "llm"}, {"max_tokens", 128},
        {"messages", Json::array({{{"role", "user"}, {"content", std::move(content)}}})}
    }));
    std::cout << response["choices"][0]["message"]["content"].get<std::string>() << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "server-openai-chat: " << error.what() << '\n';
    return 1;
}
