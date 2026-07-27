#define _USE_MATH_DEFINES
#include "audio/resample.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace forge::media {

bool valid_mono_audio(const float* samples, size_t sample_count, int32_t sample_rate) {
    if (!samples || sample_count == 0 || sample_rate <= 0) return false;
    for (size_t i = 0; i < sample_count; ++i) {
        if (!std::isfinite(samples[i])) return false;
    }
    return true;
}

std::vector<float> resample_mono(const float* samples, size_t sample_count,
                                 int32_t source_rate, int32_t target_rate) {
    if (!valid_mono_audio(samples, sample_count, source_rate) || target_rate <= 0) return {};
    if (source_rate == target_rate) return {samples, samples + sample_count};

    const int divisor = std::gcd(source_rate, target_rate);
    const int original_frequency = source_rate / divisor;
    const int new_frequency = target_rate / divisor;
    constexpr int filter_width = 6;
    constexpr double rolloff = 0.99;
    const double base_frequency = std::min(original_frequency, new_frequency) * rolloff;
    const int width = static_cast<int>(std::ceil(filter_width * original_frequency / base_frequency));
    const int kernel_size = 2 * width + original_frequency;
    if (kernel_size <= 0 || static_cast<size_t>(new_frequency) >
            std::numeric_limits<size_t>::max() / static_cast<size_t>(kernel_size)) return {};

    std::vector<double> kernels(static_cast<size_t>(new_frequency) * kernel_size);
    for (int phase = 0; phase < new_frequency; ++phase) {
        for (int tap = 0; tap < kernel_size; ++tap) {
            const double position = static_cast<double>(tap - width) / original_frequency;
            double time = (-static_cast<double>(phase) / new_frequency + position) * base_frequency;
            time = std::clamp(time, -static_cast<double>(filter_width),
                              static_cast<double>(filter_width));
            const double window = std::pow(std::cos(time * M_PI / filter_width / 2.0), 2.0);
            const double angle = time * M_PI;
            const double sinc = std::abs(angle) < 1e-12 ? 1.0 : std::sin(angle) / angle;
            kernels[static_cast<size_t>(phase) * kernel_size + tap] =
                sinc * window * base_frequency / original_frequency;
        }
    }

    const long double scaled = std::ceil(
        static_cast<long double>(new_frequency) * sample_count / original_frequency);
    if (scaled <= 0 || scaled > static_cast<long double>(std::numeric_limits<size_t>::max())) return {};
    std::vector<float> output(static_cast<size_t>(scaled));
    for (size_t index = 0; index < output.size(); ++index) {
        const int phase = static_cast<int>(index % new_frequency);
        const size_t frame = index / new_frequency;
        double sum = 0.0;
        for (int tap = 0; tap < kernel_size; ++tap) {
            const int64_t source_index = static_cast<int64_t>(frame * original_frequency + tap) - width;
            if (source_index >= 0 && source_index < static_cast<int64_t>(sample_count)) {
                sum += samples[source_index] * kernels[static_cast<size_t>(phase) * kernel_size + tap];
            }
        }
        output[index] = static_cast<float>(sum);
    }
    return output;
}

} // namespace forge::media
