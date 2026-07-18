#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace example {

struct Audio {
    int sample_rate = 0;
    std::vector<float> samples;
};

inline Audio load_wav(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    char id[4] = {};
    uint32_t size = 0;
    if (!file.read(id, 4) || std::memcmp(id, "RIFF", 4) != 0 ||
        !file.read(reinterpret_cast<char*>(&size), 4) ||
        !file.read(id, 4) || std::memcmp(id, "WAVE", 4) != 0) {
        return {};
    }

    uint16_t format = 0;
    uint16_t channels = 0;
    uint16_t bits = 0;
    uint32_t sample_rate = 0;
    std::vector<char> data;
    while (file.read(id, 4) && file.read(reinterpret_cast<char*>(&size), 4)) {
        if (std::memcmp(id, "fmt ", 4) == 0 && size >= 16) {
            file.read(reinterpret_cast<char*>(&format), 2);
            file.read(reinterpret_cast<char*>(&channels), 2);
            file.read(reinterpret_cast<char*>(&sample_rate), 4);
            file.seekg(6, std::ios::cur);
            file.read(reinterpret_cast<char*>(&bits), 2);
            file.seekg(size - 16, std::ios::cur);
        } else if (std::memcmp(id, "data", 4) == 0) {
            data.resize(size);
            if (!file.read(data.data(), static_cast<std::streamsize>(size))) return {};
            break;
        } else {
            file.seekg(size, std::ios::cur);
        }
        if (size & 1u) file.seekg(1, std::ios::cur);
    }

    if (channels == 0 || sample_rate == 0 || data.empty()) return {};
    const size_t sample_bytes = bits / 8;
    if (sample_bytes == 0 || data.size() % (sample_bytes * channels) != 0) return {};
    const size_t frames = data.size() / (sample_bytes * channels);
    Audio result{static_cast<int>(sample_rate), std::vector<float>(frames)};
    for (size_t frame = 0; frame < frames; ++frame) {
        float value = 0.0f;
        for (size_t channel = 0; channel < channels; ++channel) {
            const char* source = data.data() + (frame * channels + channel) * sample_bytes;
            if (format == 1 && bits == 16) {
                int16_t sample = 0;
                std::memcpy(&sample, source, sizeof(sample));
                value += static_cast<float>(sample) / 32768.0f;
            } else if (format == 3 && bits == 32) {
                float sample = 0.0f;
                std::memcpy(&sample, source, sizeof(sample));
                value += sample;
            } else {
                return {};
            }
        }
        result.samples[frame] = value / channels;
    }
    return result;
}

} // namespace example
