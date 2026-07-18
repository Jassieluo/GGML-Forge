#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace forge::server {

std::string base64_encode(const uint8_t* data, size_t size);
bool base64_decode(const std::string& text, std::vector<uint8_t>& output);
bool decode_wav(const std::string& bytes, std::vector<float>& audio, int32_t& sample_rate);
std::string encode_wav_pcm16(const std::vector<float>& audio, int32_t sample_rate);
std::vector<uint8_t> encode_png(
    const std::vector<uint8_t>& pixels,
    uint32_t width,
    uint32_t height,
    uint32_t channels);

} // namespace forge::server
