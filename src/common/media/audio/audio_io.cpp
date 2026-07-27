#include "audio/audio_io.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

namespace forge::media {
namespace {

uint16_t read_u16(const uint8_t* value) {
    return static_cast<uint16_t>(value[0] | (static_cast<uint16_t>(value[1]) << 8));
}

uint32_t read_u32(const uint8_t* value) {
    return static_cast<uint32_t>(value[0]) | (static_cast<uint32_t>(value[1]) << 8) |
           (static_cast<uint32_t>(value[2]) << 16) | (static_cast<uint32_t>(value[3]) << 24);
}

template <typename T>
void append_little(std::vector<uint8_t>& output, T value) {
    for (size_t i = 0; i < sizeof(T); ++i) {
        output.push_back(static_cast<uint8_t>((value >> (i * 8)) & 0xff));
    }
}

} // namespace

bool decode_wav(const uint8_t* encoded, size_t encoded_size, Audio& output,
                std::string& error) {
    output = {};
    error.clear();
    if (!encoded || encoded_size < 12 || std::memcmp(encoded, "RIFF", 4) != 0 ||
        std::memcmp(encoded + 8, "WAVE", 4) != 0) {
        error = "invalid RIFF/WAVE header";
        return false;
    }

    uint16_t format = 0;
    uint16_t channels = 0;
    uint16_t bits = 0;
    uint32_t sample_rate = 0;
    const uint8_t* sample_data = nullptr;
    size_t sample_bytes = 0;
    for (size_t offset = 12; offset + 8 <= encoded_size;) {
        const uint32_t chunk_size = read_u32(encoded + offset + 4);
        const size_t payload = offset + 8;
        if (chunk_size > encoded_size - payload) {
            error = "WAV chunk exceeds input buffer";
            return false;
        }
        if (std::memcmp(encoded + offset, "fmt ", 4) == 0 && chunk_size >= 16) {
            format = read_u16(encoded + payload);
            channels = read_u16(encoded + payload + 2);
            sample_rate = read_u32(encoded + payload + 4);
            bits = read_u16(encoded + payload + 14);
        } else if (std::memcmp(encoded + offset, "data", 4) == 0) {
            sample_data = encoded + payload;
            sample_bytes = chunk_size;
        }
        const size_t padded = static_cast<size_t>(chunk_size) + (chunk_size & 1u);
        if (padded > encoded_size - payload) break;
        offset = payload + padded;
    }

    if (!sample_data || channels == 0 || sample_rate == 0 ||
        !((format == 1 && bits == 16) || (format == 3 && bits == 32))) {
        error = "WAV must contain PCM16 or IEEE-float32 samples";
        return false;
    }
    const size_t bytes_per_sample = bits / 8;
    if (channels > std::numeric_limits<size_t>::max() / bytes_per_sample) {
        error = "invalid WAV channel count";
        return false;
    }
    const size_t frame_bytes = static_cast<size_t>(channels) * bytes_per_sample;
    if (sample_bytes == 0 || sample_bytes % frame_bytes != 0) {
        error = "WAV data chunk is not frame-aligned";
        return false;
    }
    const size_t value_count = sample_bytes / bytes_per_sample;
    output.sample_rate = sample_rate;
    output.channels = channels;
    output.samples.resize(value_count);
    for (size_t i = 0; i < value_count; ++i) {
        const uint8_t* source = sample_data + i * bytes_per_sample;
        if (format == 1) {
            output.samples[i] = static_cast<float>(static_cast<int16_t>(read_u16(source))) / 32768.0f;
        } else {
            std::memcpy(&output.samples[i], source, sizeof(float));
            if (!std::isfinite(output.samples[i])) {
                output = {};
                error = "WAV contains a non-finite sample";
                return false;
            }
        }
    }
    return true;
}

bool load_wav(const std::filesystem::path& path, Audio& output, std::string& error) {
    output = {};
    error.clear();
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        error = "failed to open WAV file";
        return false;
    }
    const std::streamoff length = input.tellg();
    if (length <= 0 || static_cast<uintmax_t>(length) >
            static_cast<uintmax_t>(std::numeric_limits<size_t>::max())) {
        error = "WAV file is empty or too large";
        return false;
    }
    std::vector<uint8_t> encoded(static_cast<size_t>(length));
    input.seekg(0, std::ios::beg);
    if (!input.read(reinterpret_cast<char*>(encoded.data()), length)) {
        error = "failed to read WAV file";
        return false;
    }
    return decode_wav(encoded.data(), encoded.size(), output, error);
}

bool decode_wav_mono(const uint8_t* encoded, size_t encoded_size,
                     std::vector<float>& samples, uint32_t& sample_rate,
                     std::string& error) {
    samples.clear();
    sample_rate = 0;
    Audio decoded;
    if (!decode_wav(encoded, encoded_size, decoded, error)) return false;
    samples = mix_to_mono(decoded);
    sample_rate = decoded.sample_rate;
    if (samples.empty()) {
        error = "WAV contains no complete audio frames";
        sample_rate = 0;
        return false;
    }
    return true;
}

bool load_wav_mono(const std::filesystem::path& path, std::vector<float>& samples,
                   uint32_t& sample_rate, std::string& error) {
    samples.clear();
    sample_rate = 0;
    Audio decoded;
    if (!load_wav(path, decoded, error)) return false;
    samples = mix_to_mono(decoded);
    sample_rate = decoded.sample_rate;
    if (samples.empty()) {
        error = "WAV contains no complete audio frames";
        sample_rate = 0;
        return false;
    }
    return true;
}

bool encode_wav_pcm16(const Audio& input, std::vector<uint8_t>& output,
                      std::string& error) {
    output.clear();
    error.clear();
    if (input.sample_rate == 0 || input.channels == 0 || input.samples.empty() ||
        input.samples.size() % input.channels != 0) {
        error = "audio buffer is empty or not frame-aligned";
        return false;
    }
    if (input.samples.size() >
            (std::numeric_limits<uint32_t>::max() - 36u) / sizeof(int16_t)) {
        error = "audio buffer is too large for a WAV data chunk";
        return false;
    }
    if (input.channels > std::numeric_limits<uint16_t>::max() / sizeof(int16_t) ||
        input.channels > std::numeric_limits<uint32_t>::max() / sizeof(int16_t)) {
        error = "audio format exceeds WAV limits";
        return false;
    }
    const uint32_t block_align_u32 = input.channels * sizeof(int16_t);
    if (input.sample_rate > std::numeric_limits<uint32_t>::max() / block_align_u32) {
        error = "audio format exceeds WAV limits";
        return false;
    }
    const uint32_t data_size = static_cast<uint32_t>(input.samples.size() * sizeof(int16_t));
    const uint16_t block_align = static_cast<uint16_t>(block_align_u32);
    output.reserve(44 + data_size);
    output.insert(output.end(), {'R','I','F','F'}); append_little(output, 36u + data_size);
    output.insert(output.end(), {'W','A','V','E','f','m','t',' '}); append_little(output, 16u);
    append_little<uint16_t>(output, 1); append_little<uint16_t>(output, static_cast<uint16_t>(input.channels));
    append_little(output, input.sample_rate); append_little(output, input.sample_rate * block_align);
    append_little(output, block_align); append_little<uint16_t>(output, 16);
    output.insert(output.end(), {'d','a','t','a'}); append_little(output, data_size);
    for (float sample : input.samples) {
        if (!std::isfinite(sample)) {
            output.clear();
            error = "audio contains a non-finite sample";
            return false;
        }
        const int16_t value = static_cast<int16_t>(std::lrint(
            std::clamp(sample, -1.0f, 1.0f) * 32767.0f));
        append_little<uint16_t>(output, static_cast<uint16_t>(value));
    }
    return true;
}

bool save_wav_pcm16(const std::filesystem::path& path, const Audio& input,
                    std::string& error) {
    std::vector<uint8_t> encoded;
    if (!encode_wav_pcm16(input, encoded, error)) return false;
    std::ofstream output(path, std::ios::binary);
    if (!output || !output.write(reinterpret_cast<const char*>(encoded.data()),
                                 static_cast<std::streamsize>(encoded.size()))) {
        error = "failed to write WAV file";
        return false;
    }
    return true;
}

bool encode_wav_pcm16_mono(const float* samples, size_t sample_count,
                           uint32_t sample_rate, std::vector<uint8_t>& output,
                           std::string& error) {
    if (!samples || sample_count == 0) {
        output.clear();
        error = "audio buffer is empty";
        return false;
    }
    Audio input{sample_rate, 1, std::vector<float>(samples, samples + sample_count)};
    return encode_wav_pcm16(input, output, error);
}

bool save_wav_pcm16_mono(const std::filesystem::path& path, const float* samples,
                         size_t sample_count, uint32_t sample_rate,
                         std::string& error) {
    if (!samples || sample_count == 0) {
        error = "audio buffer is empty";
        return false;
    }
    Audio input{sample_rate, 1, std::vector<float>(samples, samples + sample_count)};
    return save_wav_pcm16(path, input, error);
}

std::vector<float> mix_to_mono(const Audio& input) {
    if (input.channels == 0 || input.samples.empty() ||
        input.samples.size() % input.channels != 0) return {};
    if (input.channels == 1) return input.samples;
    std::vector<float> mono(input.frame_count());
    for (size_t frame = 0; frame < mono.size(); ++frame) {
        double sum = 0.0;
        for (uint32_t channel = 0; channel < input.channels; ++channel) {
            sum += input.samples[frame * input.channels + channel];
        }
        mono[frame] = static_cast<float>(sum / input.channels);
    }
    return mono;
}

} // namespace forge::media
