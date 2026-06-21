#define _USE_MATH_DEFINES
#include "dsp.h"
#include <cmath>
#include <algorithm>
#include <complex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace gpt_sovits {
namespace dsp {

void fft_inplace(std::vector<std::complex<float>>& x) {
    const int n = (int)x.size();
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(x[i], x[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * (float)M_PI / len;
        std::complex<float> wlen(std::cos(ang), std::sin(ang));
        for (int i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (int j = 0; j < len / 2; ++j) {
                std::complex<float> u = x[i + j];
                std::complex<float> v = x[i + j + len/2] * w;
                x[i + j]         = u + v;
                x[i + j + len/2] = u - v;
                w *= wlen;
            }
        }
    }
}

std::vector<float> compute_stft_spectrogram(
    const float* audio_data,
    size_t audio_len,
    int& out_frames
) {
    out_frames = 0;
    const int n_fft     = 2048;
    const int hop_len   = 640;   // hop length at 32kHz
    const int n_ref_enc = 704;   // first 704 bins

    // 1. Upsample 16kHz -> 32kHz via linear interpolation
    std::vector<float> audio32k(audio_len * 2);
    for (size_t i = 0; i < audio_len; ++i) {
        audio32k[2 * i]     = audio_data[i];
        audio32k[2 * i + 1] = (i + 1 < audio_len)
            ? 0.5f * (audio_data[i] + audio_data[i + 1])
            : audio_data[i];
    }

    // 2. Clamp to [-1, 1]
    float max_val = 0.0f;
    for (float v : audio32k) max_val = std::max(max_val, std::abs(v));
    if (max_val > 1.0f) {
        for (float& v : audio32k) v /= std::min(2.0f, max_val);
    }

    // 3. Compute STFT frames (center=False: no padding)
    const size_t n_samples32k = audio32k.size();
    if (n_samples32k < n_fft) {
        return {};
    }

    int n_frames = (int)((n_samples32k - n_fft) / hop_len) + 1;
    if (n_frames % 2 != 0) {
        n_frames -= 1;
    }
    if (n_frames <= 0) {
        return {};
    }

    // 4. Build Hann window
    std::vector<float> window(n_fft);
    for (int i = 0; i < n_fft; ++i) {
        window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / n_fft));
    }

    // 5. Compute magnitude spectrogram
    std::vector<std::complex<float>> fft_buf(n_fft);
    std::vector<float> spec_data(n_ref_enc * n_frames, 0.0f);

    for (int f = 0; f < n_frames; ++f) {
        int offset = f * hop_len;
        for (int i = 0; i < n_fft; ++i) {
            fft_buf[i] = std::complex<float>(audio32k[offset + i] * window[i], 0.0f);
        }

        fft_inplace(fft_buf);

        for (int k = 0; k < n_ref_enc; ++k) {
            float re = fft_buf[k].real();
            float im = fft_buf[k].imag();
            spec_data[f * n_ref_enc + k] = std::sqrt(re * re + im * im + 1e-8f);
        }
    }

    out_frames = n_frames;
    return spec_data;
}

} // namespace dsp
} // namespace gpt_sovits
