#include "client_common.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>

#ifdef _WIN32
#include <shellapi.h>
#include <windows.h>
#endif

namespace forge::server_example {

namespace {

#ifdef _WIN32
std::string utf8(const wchar_t* text) {
    if (!text || !*text) return {};
    const int size = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
    if (size <= 1) return {};
    std::string result(static_cast<size_t>(size), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text, -1, result.data(), size, nullptr, nullptr);
    result.resize(static_cast<size_t>(size - 1));
    return result;
}
#endif

} // namespace

Arguments::Arguments(int argc, char** argv) {
#ifdef _WIN32
    (void)argc;
    (void)argv;
    int wide_count = 0;
    wchar_t** wide = CommandLineToArgvW(GetCommandLineW(), &wide_count);
    if (!wide) throw std::runtime_error("failed to read the Windows command line");
    for (int i = 1; i < wide_count; ++i) {
        const std::string key = utf8(wide[i]);
        if (key.rfind("--", 0) != 0 || i + 1 >= wide_count) {
            LocalFree(wide);
            throw std::runtime_error("expected --name value arguments");
        }
        values_[key.substr(2)] = utf8(wide[++i]);
    }
    LocalFree(wide);
#else
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key.rfind("--", 0) != 0 || i + 1 >= argc) {
            throw std::runtime_error("expected --name value arguments");
        }
        values_[key.substr(2)] = argv[++i];
    }
#endif
}

std::string Arguments::get(const std::string& name, const std::string& fallback) const {
    const auto found = values_.find(name);
    return found == values_.end() ? fallback : found->second;
}

std::string Arguments::require(const std::string& name) const {
    const std::string value = get(name);
    if (value.empty()) throw std::runtime_error("missing required --" + name);
    return value;
}

Client::Client(std::string base_url, const std::string& api_key)
    : client_(std::move(base_url)) {
    client_.set_connection_timeout(10, 0);
    client_.set_read_timeout(600, 0);
    client_.set_write_timeout(60, 0);
    if (!api_key.empty()) headers_.emplace("Authorization", "Bearer " + api_key);
}

std::string Client::checked_body(httplib::Result result) const {
    if (!result) throw std::runtime_error("HTTP request failed: " + httplib::to_string(result.error()));
    if (result->status < 200 || result->status >= 300) {
        throw std::runtime_error("HTTP " + std::to_string(result->status) + ": " + result->body);
    }
    return std::move(result->body);
}

std::string Client::post_json(
    const std::string& path, const Json& body, const httplib::Headers& extra_headers
) {
    httplib::Headers headers = headers_;
    headers.insert(extra_headers.begin(), extra_headers.end());
    return checked_body(client_.Post(path, headers, body.dump(), "application/json; charset=utf-8"));
}

std::string Client::post_multipart(
    const std::string& path, const httplib::UploadFormDataItems& items
) {
    return checked_body(client_.Post(path, headers_, items));
}

std::vector<uint8_t> read_binary(const std::string& path) {
    std::ifstream input(std::filesystem::u8path(path), std::ios::binary);
    if (!input) throw std::runtime_error("failed to open " + path);
    return {std::istreambuf_iterator<char>(input), {}};
}

void write_binary(const std::string& path, const std::string& bytes) {
    std::ofstream output(std::filesystem::u8path(path), std::ios::binary);
    if (!output || !output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
        throw std::runtime_error("failed to write " + path);
    }
}

void write_binary(const std::string& path, const std::vector<uint8_t>& bytes) {
    write_binary(path, std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
}

std::string base64_encode(const std::vector<uint8_t>& data) {
    static constexpr char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string output;
    output.reserve((data.size() + 2) / 3 * 4);
    for (size_t i = 0; i < data.size(); i += 3) {
        const uint32_t value = static_cast<uint32_t>(data[i]) << 16 |
            (i + 1 < data.size() ? static_cast<uint32_t>(data[i + 1]) << 8 : 0) |
            (i + 2 < data.size() ? data[i + 2] : 0);
        output.push_back(alphabet[(value >> 18) & 63]);
        output.push_back(alphabet[(value >> 12) & 63]);
        output.push_back(i + 1 < data.size() ? alphabet[(value >> 6) & 63] : '=');
        output.push_back(i + 2 < data.size() ? alphabet[value & 63] : '=');
    }
    return output;
}

std::vector<uint8_t> base64_decode(const std::string& text) {
    static const std::string alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<uint8_t> output;
    uint32_t value = 0;
    int bits = -8;
    for (unsigned char byte : text) {
        if (byte == '=') break;
        const size_t index = alphabet.find(static_cast<char>(byte));
        if (index == std::string::npos) throw std::runtime_error("invalid base64 response");
        value = (value << 6) | static_cast<uint32_t>(index);
        bits += 6;
        if (bits >= 0) {
            output.push_back(static_cast<uint8_t>((value >> bits) & 0xff));
            bits -= 8;
        }
    }
    return output;
}

std::string image_mime_type(const std::string& path) {
    std::string extension = std::filesystem::u8path(path).extension().u8string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
        [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    if (extension == ".jpg" || extension == ".jpeg") return "image/jpeg";
    if (extension == ".webp") return "image/webp";
    if (extension == ".gif") return "image/gif";
    return "image/png";
}

} // namespace forge::server_example
