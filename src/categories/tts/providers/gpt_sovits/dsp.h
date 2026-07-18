#pragma once

// Provider-local audio preprocessing.

#include <vector>
#include <complex>

namespace gpt_sovits {
namespace dsp {

void fft_inplace(std::vector<std::complex<float>>& x);

// Matches torchaudio's default sinc_interp_hann resampler.
std::vector<float> resample_audio(
    const float* audio_data,
    size_t audio_len,
    int source_rate,
    int target_rate
);

// Computes the magnitude STFT spectrogram of the input audio.
// Input audio must already be at the model's reference sampling rate (32 kHz for classic VITS).
std::vector<float> compute_stft_spectrogram(
    const float* audio_data,
    size_t audio_len,
    int n_ref_enc,
    int& out_frames
);

// Computes the log-Mel spectrogram (100 bins) of the input audio.
std::vector<float> compute_mel_spectrogram(
    const float* audio_data,
    size_t audio_len,
    int sampling_rate, // input/model rate: 24000 for v3, 32000 for v4
    int n_fft,         // 1024 for v3, 1280 for v4
    int hop_size,      // 256 for v3, 320 for v4
    int win_size,      // 1024 for v3, 1280 for v4
    int num_mels,      // 100
    int& out_frames
);

} // namespace dsp
} // namespace gpt_sovits
