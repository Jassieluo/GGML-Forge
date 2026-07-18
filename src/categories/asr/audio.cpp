#include "audio.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace asr::audio {

bool valid(const float* samples, size_t sample_count, int32_t sample_rate) {
    if (!samples || sample_count == 0 || sample_rate <= 0) return false;
    for (size_t i = 0; i < sample_count; ++i) {
        if (!std::isfinite(samples[i])) return false;
    }
    return true;
}

std::vector<float> resample_mono(
    const float* samples,
    size_t sample_count,
    int32_t source_rate,
    int32_t target_rate
) {
    if (!valid(samples, sample_count, source_rate) || target_rate <= 0) return {};
    if (source_rate == target_rate) return {samples, samples + sample_count};

    const long double scaled = static_cast<long double>(sample_count) * target_rate / source_rate;
    if (scaled <= 0 || scaled > static_cast<long double>(std::numeric_limits<size_t>::max())) return {};
    const size_t output_count = std::max<size_t>(1, static_cast<size_t>(std::llround(scaled)));
    std::vector<float> output(output_count);
    const double step = static_cast<double>(source_rate) / target_rate;
    for (size_t i = 0; i < output_count; ++i) {
        const double position = std::min<double>(i * step, sample_count - 1);
        const size_t left = static_cast<size_t>(position);
        const size_t right = std::min(left + 1, sample_count - 1);
        const float fraction = static_cast<float>(position - left);
        output[i] = samples[left] + (samples[right] - samples[left]) * fraction;
    }
    return output;
}

} // namespace asr::audio
