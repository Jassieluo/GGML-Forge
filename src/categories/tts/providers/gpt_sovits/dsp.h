#pragma once

// Provider-local audio preprocessing.

#include <vector>
#include <complex>

namespace gpt_sovits {
namespace dsp {

void fft_inplace(std::vector<std::complex<float>>& x);

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

// Kaldi-compatible 80-bin log filterbank used by ERes2NetV2. Input must be
// mono 16 kHz float audio. Output uses GGML [frames, mel] storage order, so
// element (frame, mel) is at mel * out_frames + frame.
std::vector<float> compute_kaldi_fbank_80(
    const float* audio,
    size_t sample_count,
    int& out_frames
);

} // namespace dsp
} // namespace gpt_sovits
