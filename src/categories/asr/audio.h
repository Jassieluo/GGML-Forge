#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace asr::audio {

bool valid(const float* samples, size_t sample_count, int32_t sample_rate);
std::vector<float> resample_mono(
    const float* samples,
    size_t sample_count,
    int32_t source_rate,
    int32_t target_rate);

} // namespace asr::audio
