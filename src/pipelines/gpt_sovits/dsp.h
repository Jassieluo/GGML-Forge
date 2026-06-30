#pragma once

#include <vector>
#include <complex>

namespace gpt_sovits {
namespace dsp {

void fft_inplace(std::vector<std::complex<float>>& x);

// Computes the magnitude STFT spectrogram (first 704 bins) of the input audio.
// Upsamples from 16kHz to 32kHz on the fly if needed.
std::vector<float> compute_stft_spectrogram(
    const float* audio_data,
    size_t audio_len,
    int& out_frames
);

// Computes the log-Mel spectrogram (100 bins) of the input audio.
std::vector<float> compute_mel_spectrogram(
    const float* audio_data,
    size_t audio_len,
    int sampling_rate, // 24000 for v3, 32000 for v4
    int n_fft,         // 1024 for v3, 1280 for v4
    int hop_size,      // 256 for v3, 320 for v4
    int win_size,      // 1024 for v3, 1280 for v4
    int num_mels,      // 100
    int& out_frames
);

} // namespace dsp
} // namespace gpt_sovits
