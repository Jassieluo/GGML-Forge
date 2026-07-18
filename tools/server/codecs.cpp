#include "codecs.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace forge::server {
namespace {

template <typename T>
void append_little(std::string& output, T value) {
    for (size_t i = 0; i < sizeof(T); ++i) output.push_back(static_cast<char>((value >> (i * 8)) & 0xff));
}

uint32_t read_u32(const char* value) {
    const auto* b = reinterpret_cast<const uint8_t*>(value);
    return static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) |
        (static_cast<uint32_t>(b[2]) << 16) | (static_cast<uint32_t>(b[3]) << 24);
}

uint16_t read_u16(const char* value) {
    const auto* b = reinterpret_cast<const uint8_t*>(value);
    return static_cast<uint16_t>(b[0] | (static_cast<uint16_t>(b[1]) << 8));
}

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

bool decode_wav(const std::string& bytes, std::vector<float>& audio, int32_t& sample_rate) {
    audio.clear();
    sample_rate = 0;
    if (bytes.size() < 44 || std::memcmp(bytes.data(), "RIFF", 4) ||
        std::memcmp(bytes.data() + 8, "WAVE", 4)) return false;
    uint16_t format = 0, channels = 0, bits = 0;
    const char* data = nullptr;
    size_t data_size = 0;
    for (size_t offset = 12; offset + 8 <= bytes.size();) {
        const uint32_t size = read_u32(bytes.data() + offset + 4);
        const size_t payload = offset + 8;
        if (payload + size > bytes.size()) return false;
        if (!std::memcmp(bytes.data() + offset, "fmt ", 4) && size >= 16) {
            format = read_u16(bytes.data() + payload);
            channels = read_u16(bytes.data() + payload + 2);
            sample_rate = static_cast<int32_t>(read_u32(bytes.data() + payload + 4));
            bits = read_u16(bytes.data() + payload + 14);
        } else if (!std::memcmp(bytes.data() + offset, "data", 4)) {
            data = bytes.data() + payload;
            data_size = size;
        }
        offset = payload + size + (size & 1u);
    }
    if (!data || !channels || sample_rate <= 0 ||
        !((format == 1 && bits == 16) || (format == 3 && bits == 32))) return false;
    const size_t bytes_per_sample = bits / 8;
    const size_t frames = data_size / (bytes_per_sample * channels);
    audio.resize(frames);
    for (size_t frame = 0; frame < frames; ++frame) {
        float mixed = 0.0f;
        for (uint16_t channel = 0; channel < channels; ++channel) {
            const char* sample = data + (frame * channels + channel) * bytes_per_sample;
            if (format == 1) {
                const int16_t value = static_cast<int16_t>(read_u16(sample));
                mixed += static_cast<float>(value) / 32768.0f;
            } else {
                float value = 0.0f;
                std::memcpy(&value, sample, sizeof(value));
                mixed += value;
            }
        }
        audio[frame] = mixed / channels;
    }
    return !audio.empty();
}

std::string encode_wav_pcm16(const std::vector<float>& audio, int32_t sample_rate) {
    const uint32_t data_size = static_cast<uint32_t>(audio.size() * sizeof(int16_t));
    std::string output;
    output.reserve(44 + data_size);
    output.append("RIFF", 4); append_little(output, 36u + data_size);
    output.append("WAVEfmt ", 8); append_little(output, 16u);
    append_little<uint16_t>(output, 1); append_little<uint16_t>(output, 1);
    append_little(output, static_cast<uint32_t>(sample_rate));
    append_little(output, static_cast<uint32_t>(sample_rate * 2));
    append_little<uint16_t>(output, 2); append_little<uint16_t>(output, 16);
    output.append("data", 4); append_little(output, data_size);
    for (float sample : audio) {
        const int16_t value = static_cast<int16_t>(std::lrint(
            std::clamp(sample, -1.0f, 1.0f) * 32767.0f));
        append_little<uint16_t>(output, static_cast<uint16_t>(value));
    }
    return output;
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
