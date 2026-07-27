#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace forge::media {

bool valid_mono_audio(const float* samples, size_t sample_count, int32_t sample_rate);

// Band-limited sinc/Hann resampling compatible with torchaudio's default
// sinc_interp_hann parameters.
std::vector<float> resample_mono(const float* samples, size_t sample_count,
                                 int32_t source_rate, int32_t target_rate);

} // namespace forge::media
