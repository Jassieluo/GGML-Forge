#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace forge::media {

// Interleaved normalized F32 samples. sample_count() includes all channels;
// frame_count() is the number of time-domain frames.
struct Audio {
    uint32_t sample_rate = 0;
    uint32_t channels = 0;
    std::vector<float> samples;

    size_t frame_count() const {
        return channels == 0 ? 0 : samples.size() / channels;
    }
};

bool decode_wav(const uint8_t* encoded, size_t encoded_size, Audio& output,
                std::string& error);
bool load_wav(const std::filesystem::path& path, Audio& output, std::string& error);
bool decode_wav_mono(const uint8_t* encoded, size_t encoded_size,
                     std::vector<float>& samples, uint32_t& sample_rate,
                     std::string& error);
bool load_wav_mono(const std::filesystem::path& path, std::vector<float>& samples,
                   uint32_t& sample_rate, std::string& error);
bool encode_wav_pcm16(const Audio& input, std::vector<uint8_t>& output,
                      std::string& error);
bool save_wav_pcm16(const std::filesystem::path& path, const Audio& input,
                    std::string& error);
bool encode_wav_pcm16_mono(const float* samples, size_t sample_count,
                           uint32_t sample_rate, std::vector<uint8_t>& output,
                           std::string& error);
bool save_wav_pcm16_mono(const std::filesystem::path& path, const float* samples,
                         size_t sample_count, uint32_t sample_rate,
                         std::string& error);
std::vector<float> mix_to_mono(const Audio& input);

} // namespace forge::media
