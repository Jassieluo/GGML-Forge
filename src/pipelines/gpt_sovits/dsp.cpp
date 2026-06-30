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

std::vector<float> compute_mel_spectrogram(
    const float* audio_data,
    size_t audio_len,
    int sampling_rate, // 24000 for v3, 32000 for v4
    int n_fft,         // 1024 for v3, 1280 for v4
    int hop_size,      // 256 for v3, 320 for v4
    int win_size,      // 1024 for v3, 1280 for v4
    int num_mels,      // 100
    int& out_frames
) {
    out_frames = 0;
    
    // 1. Resample input 16kHz audio to target sampling_rate
    std::vector<float> audio_resampled;
    int in_sr = 16000;
    if (in_sr == sampling_rate) {
        audio_resampled.assign(audio_data, audio_data + audio_len);
    } else {
        double scale = (double)sampling_rate / in_sr;
        size_t out_len = (size_t)(audio_len * scale);
        audio_resampled.resize(out_len);
        for (size_t i = 0; i < out_len; ++i) {
            double src_pos = i / scale;
            size_t idx0 = (size_t)src_pos;
            size_t idx1 = std::min(idx0 + 1, audio_len - 1);
            float t = (float)(src_pos - idx0);
            audio_resampled[i] = (1.0f - t) * audio_data[idx0] + t * audio_data[idx1];
        }
    }
    
    // 2. Reflected padding (n_fft - hop_size) / 2
    int pad = (n_fft - hop_size) / 2;
    size_t res_len = audio_resampled.size();
    std::vector<float> padded_audio(res_len + 2 * pad);
    for (int i = 0; i < pad; ++i) {
        padded_audio[pad - 1 - i] = audio_resampled[std::min((size_t)(i + 1), res_len - 1)];
    }
    std::memcpy(padded_audio.data() + pad, audio_resampled.data(), res_len * sizeof(float));
    for (int i = 0; i < pad; ++i) {
        padded_audio[pad + res_len + i] = audio_resampled[std::max((int)0, (int)(res_len - 2 - i))];
    }
    
    // 3. Compute STFT frames (center = False because we already padded)
    const size_t total_samples = padded_audio.size();
    if (total_samples < (size_t)n_fft) {
        return {};
    }
    
    int n_frames = (int)((total_samples - n_fft) / hop_size) + 1;
    if (n_frames <= 0) {
        return {};
    }
    
    // 4. Build Hann window of size win_size
    std::vector<float> window(win_size);
    for (int i = 0; i < win_size; ++i) {
        window[i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * i / win_size));
    }
    
    // 5. Build Mel Filterbank
    // If n_fft is not a power of 2 (e.g. 1280), we pad FFT to next power of 2 (e.g. 2048)
    int n_fft_actual = n_fft;
    int n_fft_pow2 = 1;
    while (n_fft_pow2 < n_fft_actual) {
        n_fft_pow2 <<= 1;
    }
    
    int fft_bins = n_fft_pow2 / 2 + 1;
    std::vector<float> mel_filters(num_mels * fft_bins, 0.0f);
    
    double fmin = 0.0;
    double fmax = (double)sampling_rate / 2.0;
    
    auto hz_to_mel = [](double hz) {
        return 2595.0 * std::log10(1.0 + hz / 700.0);
    };
    auto mel_to_hz = [](double mel) {
        return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
    };
    
    double mel_min = hz_to_mel(fmin);
    double mel_max = hz_to_mel(fmax);
    
    std::vector<double> mel_pts(num_mels + 2);
    std::vector<double> freq_pts(num_mels + 2);
    std::vector<int> bin_pts(num_mels + 2);
    for (int i = 0; i < num_mels + 2; ++i) {
        mel_pts[i] = mel_min + i * (mel_max - mel_min) / (num_mels + 1);
        freq_pts[i] = mel_to_hz(mel_pts[i]);
        bin_pts[i] = (int)std::floor((n_fft_pow2 + 1) * freq_pts[i] / sampling_rate);
    }
    
    for (int m = 0; m < num_mels; ++m) {
        int b0 = bin_pts[m];
        int b1 = bin_pts[m + 1];
        int b2 = bin_pts[m + 2];
        
        double fs_diff0 = freq_pts[m + 1] - freq_pts[m];
        double fs_diff1 = freq_pts[m + 2] - freq_pts[m + 1];
        
        // Slaney normalization scale
        double slaney_scale = 2.0 / (freq_pts[m + 2] - freq_pts[m]);
        
        for (int k = b0; k < b2; ++k) {
            if (k >= fft_bins) break;
            double freq_k = (double)k * sampling_rate / n_fft_pow2;
            double w = 0.0;
            if (k < b1) {
                w = (freq_k - freq_pts[m]) / fs_diff0;
            } else {
                w = (freq_pts[m + 2] - freq_k) / fs_diff1;
            }
            if (w > 0.0) {
                mel_filters[m * fft_bins + k] = (float)(w * slaney_scale);
            }
        }
    }
    
    // 6. Compute STFT and project to Mel scale
    std::vector<std::complex<float>> fft_buf(n_fft_pow2);
    std::vector<float> spec_data(num_mels * n_frames, 0.0f);
    
    for (int f = 0; f < n_frames; ++f) {
        int offset = f * hop_size;
        
        // Window and pad
        std::fill(fft_buf.begin(), fft_buf.end(), std::complex<float>(0.0f, 0.0f));
        int win_offset = (n_fft_actual - win_size) / 2;
        for (int i = 0; i < win_size; ++i) {
            fft_buf[i] = std::complex<float>(padded_audio[offset + win_offset + i] * window[i], 0.0f);
        }
        
        fft_inplace(fft_buf);
        
        // Compute magnitude spectrum and Mel projection
        std::vector<float> mag(fft_bins);
        for (int k = 0; k < fft_bins; ++k) {
            float re = fft_buf[k].real();
            float im = fft_buf[k].imag();
            mag[k] = std::sqrt(re * re + im * im + 1e-8f);
        }
        
        for (int m = 0; m < num_mels; ++m) {
            float sum = 0.0f;
            for (int k = 0; k < fft_bins; ++k) {
                sum += mel_filters[m * fft_bins + k] * mag[k];
            }
            // Dynamic range compression: ln(max(val, 1e-5))
            spec_data[f * num_mels + m] = std::log(std::max(sum, 1e-5f));
        }
    }
    
    out_frames = n_frames;
    return spec_data;
}

} // namespace dsp
} // namespace gpt_sovits
