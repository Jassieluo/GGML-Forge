#include "codecs.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace forge::server {
namespace {

void append_be32(std::vector<uint8_t>& output, uint32_t value) {
    output.push_back(static_cast<uint8_t>(value >> 24));
    output.push_back(static_cast<uint8_t>(value >> 16));
    output.push_back(static_cast<uint8_t>(value >> 8));
    output.push_back(static_cast<uint8_t>(value));
}

uint32_t crc32(const uint8_t* data, size_t size) {
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}

void png_chunk(std::vector<uint8_t>& output, const char type[4], const std::vector<uint8_t>& data) {
    append_be32(output, static_cast<uint32_t>(data.size()));
    const size_t crc_begin = output.size();
    output.insert(output.end(), type, type + 4);
    output.insert(output.end(), data.begin(), data.end());
    append_be32(output, crc32(output.data() + crc_begin, output.size() - crc_begin));
}

} // namespace

std::string base64_encode(const uint8_t* data, size_t size) {
    static constexpr char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string output;
    output.reserve((size + 2) / 3 * 4);
    for (size_t i = 0; i < size; i += 3) {
        const uint32_t value = static_cast<uint32_t>(data[i]) << 16 |
            (i + 1 < size ? static_cast<uint32_t>(data[i + 1]) << 8 : 0) |
            (i + 2 < size ? data[i + 2] : 0);
        output.push_back(alphabet[(value >> 18) & 63]);
        output.push_back(alphabet[(value >> 12) & 63]);
        output.push_back(i + 1 < size ? alphabet[(value >> 6) & 63] : '=');
        output.push_back(i + 2 < size ? alphabet[value & 63] : '=');
    }
    return output;
}

bool base64_decode(const std::string& text, std::vector<uint8_t>& output) {
    static const std::string alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    output.clear();
    uint32_t value = 0;
    int bits = -8;
    for (unsigned char c : text) {
        if (c == '=') break;
        const size_t index = alphabet.find(static_cast<char>(c));
        if (index == std::string::npos) {
            if (c == ' ' || c == '\r' || c == '\n' || c == '\t') continue;
            output.clear();
            return false;
        }
        value = (value << 6) | static_cast<uint32_t>(index);
        bits += 6;
        if (bits >= 0) {
            output.push_back(static_cast<uint8_t>((value >> bits) & 0xff));
            bits -= 8;
        }
    }
    return !output.empty();
}

std::vector<uint8_t> encode_png(
    const std::vector<uint8_t>& pixels, uint32_t width, uint32_t height, uint32_t channels
) {
    if (!width || !height || (channels != 3 && channels != 4) ||
        pixels.size() < static_cast<size_t>(width) * height * channels) return {};
    static constexpr uint8_t signature[] = {137,80,78,71,13,10,26,10};
    std::vector<uint8_t> output(signature, signature + sizeof(signature));
    std::vector<uint8_t> ihdr;
    append_be32(ihdr, width); append_be32(ihdr, height);
    ihdr.push_back(8); ihdr.push_back(channels == 4 ? 6 : 2);
    ihdr.insert(ihdr.end(), {0, 0, 0});
    png_chunk(output, "IHDR", ihdr);

    std::vector<uint8_t> raw;
    raw.reserve((static_cast<size_t>(width) * channels + 1) * height);
    for (uint32_t y = 0; y < height; ++y) {
        raw.push_back(0);
        const size_t begin = static_cast<size_t>(y) * width * channels;
        raw.insert(raw.end(), pixels.begin() + begin, pixels.begin() + begin + width * channels);
    }
    std::vector<uint8_t> zlib = {0x78, 0x01};
    for (size_t offset = 0; offset < raw.size();) {
        const uint16_t block = static_cast<uint16_t>(std::min<size_t>(65535, raw.size() - offset));
        const bool final = offset + block == raw.size();
        zlib.push_back(final ? 1 : 0);
        zlib.push_back(static_cast<uint8_t>(block)); zlib.push_back(static_cast<uint8_t>(block >> 8));
        const uint16_t inverse = static_cast<uint16_t>(~block);
        zlib.push_back(static_cast<uint8_t>(inverse)); zlib.push_back(static_cast<uint8_t>(inverse >> 8));
        zlib.insert(zlib.end(), raw.begin() + offset, raw.begin() + offset + block);
        offset += block;
    }
    uint32_t a = 1, b = 0;
    for (uint8_t byte : raw) { a = (a + byte) % 65521u; b = (b + a) % 65521u; }
    append_be32(zlib, (b << 16) | a);
    png_chunk(output, "IDAT", zlib);
    png_chunk(output, "IEND", {});
    return output;
}

} // namespace forge::server
