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

} // namespace dsp
} // namespace gpt_sovits
