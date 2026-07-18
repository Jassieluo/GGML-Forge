#pragma once

#include "httplib.h"
#include "json.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace forge::server_example {

using Json = nlohmann::ordered_json;

class Arguments {
public:
    Arguments(int argc, char** argv);
    std::string get(const std::string& name, const std::string& fallback = {}) const;
    std::string require(const std::string& name) const;

private:
    std::unordered_map<std::string, std::string> values_;
};

class Client {
public:
    Client(std::string base_url, const std::string& api_key);
    std::string post_json(
        const std::string& path,
        const Json& body,
        const httplib::Headers& extra_headers = {});
    std::string post_multipart(
        const std::string& path,
        const httplib::UploadFormDataItems& items);

private:
    std::string checked_body(httplib::Result result) const;

    httplib::Client client_;
    httplib::Headers headers_;
};

std::vector<uint8_t> read_binary(const std::string& path);
void write_binary(const std::string& path, const std::string& bytes);
void write_binary(const std::string& path, const std::vector<uint8_t>& bytes);
std::string base64_encode(const std::vector<uint8_t>& data);
std::vector<uint8_t> base64_decode(const std::string& text);
std::string image_mime_type(const std::string& path);

} // namespace forge::server_example
