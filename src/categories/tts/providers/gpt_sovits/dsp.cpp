#define _USE_MATH_DEFINES
// Provider-local audio preprocessing.
#include "dsp.h"
#include <cmath>
#include <algorithm>
#include <complex>
#include <numeric>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace gpt_sovits {
namespace dsp {

std::vector<float> resample_audio(
    const float* audio_data,
    size_t audio_len,
    int source_rate,
    int target_rate
) {
    if (!audio_data || audio_len == 0 || source_rate <= 0 || target_rate <= 0) return {};
    if (source_rate == target_rate) return {audio_data, audio_data + audio_len};

    const int divisor = std::gcd(source_rate, target_rate);
    const int orig_freq = source_rate / divisor;
    const int new_freq = target_rate / divisor;
    constexpr int lowpass_filter_width = 6;
    constexpr double rolloff = 0.99;
    const double base_freq = std::min(orig_freq, new_freq) * rolloff;
    const int width = static_cast<int>(std::ceil(lowpass_filter_width * orig_freq / base_freq));
    const int kernel_size = 2 * width + orig_freq;

    std::vector<double> kernels(static_cast<size_t>(new_freq) * kernel_size);
    for (int phase = 0; phase < new_freq; ++phase) {
        for (int tap = 0; tap < kernel_size; ++tap) {
            const double idx = static_cast<double>(tap - width) / orig_freq;
            double t = (-static_cast<double>(phase) / new_freq + idx) * base_freq;
            t = std::clamp(t, -static_cast<double>(lowpass_filter_width),
                           static_cast<double>(lowpass_filter_width));
            const double window = std::pow(std::cos(t * M_PI / lowpass_filter_width / 2.0), 2.0);
            const double angle = t * M_PI;
            const double sinc = std::abs(angle) < 1e-12 ? 1.0 : std::sin(angle) / angle;
            kernels[static_cast<size_t>(phase) * kernel_size + tap] =
                sinc * window * base_freq / orig_freq;
        }
    }

    const size_t target_len = static_cast<size_t>(
        std::ceil(static_cast<double>(new_freq) * audio_len / orig_freq));
    std::vector<float> output(target_len);
    for (size_t index = 0; index < target_len; ++index) {
        const int phase = static_cast<int>(index % new_freq);
        const size_t frame = index / new_freq;
        double sum = 0.0;
        for (int tap = 0; tap < kernel_size; ++tap) {
            const int64_t source_index = static_cast<int64_t>(frame * orig_freq + tap) - width;
            if (source_index >= 0 && source_index < static_cast<int64_t>(audio_len)) {
                sum += audio_data[source_index] * kernels[static_cast<size_t>(phase) * kernel_size + tap];
            }
        }
        output[index] = static_cast<float>(sum);
    }
    return output;
}

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

static void ifft_inplace(std::vector<std::complex<float>>& x) {
    for (auto& value : x) value = std::conj(value);
    fft_inplace(x);
    const float scale = 1.0f / static_cast<float>(x.size());
    for (auto& value : x) value = std::conj(value) * scale;
}

static std::vector<std::complex<float>> fft_any_length(const std::vector<std::complex<float>>& input) {
    const size_t n = input.size();
    if (n > 0 && (n & (n - 1)) == 0) {
        auto output = input;
        fft_inplace(output);
        return output;
    }

    size_t convolution_size = 1;
    while (convolution_size < 2 * n - 1) convolution_size <<= 1;
    std::vector<std::complex<float>> a(convolution_size, {0.0f, 0.0f});
    std::vector<std::complex<float>> b(convolution_size, {0.0f, 0.0f});
    for (size_t i = 0; i < n; ++i) {
        const double angle = M_PI * static_cast<double>(i) * static_cast<double>(i) / n;
        const std::complex<float> forward_chirp = std::polar(1.0f, static_cast<float>(-angle));
        const std::complex<float> inverse_chirp = std::polar(1.0f, static_cast<float>(angle));
        a[i] = input[i] * forward_chirp;
        b[i] = inverse_chirp;
        if (i != 0) b[convolution_size - i] = inverse_chirp;
    }
    fft_inplace(a);
    fft_inplace(b);
    for (size_t i = 0; i < convolution_size; ++i) a[i] *= b[i];
    ifft_inplace(a);

    a.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const double angle = M_PI * static_cast<double>(i) * static_cast<double>(i) / n;
        a[i] *= std::polar(1.0f, static_cast<float>(-angle));
    }
    return a;
}

std::vector<float> compute_stft_spectrogram(
    const float* audio_data,
    size_t audio_len,
    int n_ref_enc,
    int& out_frames
) {
    out_frames = 0;
    const int n_fft     = 2048;
    const int hop_len   = 640;   // hop length at 32kHz

    std::vector<float> audio32k(audio_data, audio_data + audio_len);

    // 2. Clamp to [-1, 1]
    float max_val = 0.0f;
    for (float v : audio32k) max_val = std::max(max_val, std::abs(v));
    if (max_val > 1.0f) {
        for (float& v : audio32k) v /= std::min(2.0f, max_val);
    }

    // Python pads explicitly before torch.stft(center=False).
    const int pad = (n_fft - hop_len) / 2;
    const size_t n_samples32k = audio32k.size();
    if (n_samples32k < 2) {
        return {};
    }
    std::vector<float> padded_audio(n_samples32k + 2 * pad);
    for (int i = 0; i < pad; ++i) {
        padded_audio[pad - 1 - i] = audio32k[std::min(static_cast<size_t>(i + 1), n_samples32k - 1)];
    }
    std::memcpy(padded_audio.data() + pad, audio32k.data(), n_samples32k * sizeof(float));
    for (int i = 0; i < pad; ++i) {
        padded_audio[pad + n_samples32k + i] = audio32k[std::max(0, static_cast<int>(n_samples32k) - 2 - i)];
    }

    int n_frames = static_cast<int>((padded_audio.size() - n_fft) / hop_len) + 1;
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
            fft_buf[i] = std::complex<float>(padded_audio[offset + i] * window[i], 0.0f);
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

std::vector<float> compute_kaldi_fbank_80(
    const float* audio,
    size_t sample_count,
    int& out_frames
) {
    constexpr int sample_rate = 16000;
    constexpr int frame_size = 400;
    constexpr int frame_shift = 160;
    constexpr int fft_size = 512;
    constexpr int fft_bins = fft_size / 2;
    constexpr int mel_bins = 80;
    constexpr float preemphasis = 0.97f;
    out_frames = sample_count >= frame_size
        ? 1 + static_cast<int>((sample_count - frame_size) / frame_shift)
        : 0;
    if (!audio || out_frames <= 0) return {};

    std::vector<float> window(frame_size);
    for (int i = 0; i < frame_size; ++i) {
        const float hann = 0.5f - 0.5f * std::cos(
            2.0f * static_cast<float>(M_PI) * i / (frame_size - 1));
        window[i] = std::pow(std::max(0.0f, hann), 0.85f);
    }

    const auto mel_scale = [](double hz) {
        return 1127.0 * std::log(1.0 + hz / 700.0);
    };
    const double mel_low = mel_scale(20.0);
    const double mel_high = mel_scale(sample_rate * 0.5);
    const double mel_delta = (mel_high - mel_low) / (mel_bins + 1);
    std::vector<float> filters(static_cast<size_t>(mel_bins) * fft_bins, 0.0f);
    for (int mel = 0; mel < mel_bins; ++mel) {
        const double left = mel_low + mel * mel_delta;
        const double center = left + mel_delta;
        const double right = center + mel_delta;
        for (int bin = 0; bin < fft_bins; ++bin) {
            const double frequency = static_cast<double>(bin) * sample_rate / fft_size;
            const double value = mel_scale(frequency);
            const double up = (value - left) / (center - left);
            const double down = (right - value) / (right - center);
            filters[static_cast<size_t>(mel) * fft_bins + bin] =
                static_cast<float>(std::max(0.0, std::min(up, down)));
        }
    }

    std::vector<float> result(static_cast<size_t>(out_frames) * mel_bins);
    std::vector<float> frame(frame_size);
    std::vector<std::complex<float>> spectrum(fft_size);
    for (int index = 0; index < out_frames; ++index) {
        const float* source = audio + static_cast<size_t>(index) * frame_shift;
        double mean = 0.0;
        for (int i = 0; i < frame_size; ++i) mean += source[i];
        mean /= frame_size;
        for (int i = 0; i < frame_size; ++i) frame[i] = source[i] - static_cast<float>(mean);
        std::fill(spectrum.begin(), spectrum.end(), std::complex<float>(0.0f, 0.0f));
        for (int i = 0; i < frame_size; ++i) {
            const float previous = frame[i > 0 ? i - 1 : 0];
            spectrum[i] = std::complex<float>(
                (frame[i] - preemphasis * previous) * window[i], 0.0f);
        }
        fft_inplace(spectrum);
        for (int mel = 0; mel < mel_bins; ++mel) {
            double energy = 0.0;
            const float* filter = filters.data() + static_cast<size_t>(mel) * fft_bins;
            for (int bin = 0; bin < fft_bins; ++bin) {
                energy += filter[bin] * std::norm(spectrum[bin]);
            }
            result[static_cast<size_t>(mel) * out_frames + index] = std::log(
                std::max(energy, static_cast<double>(std::numeric_limits<float>::epsilon())));
        }
    }
    return result;
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
    
    std::vector<float> audio_resampled(audio_data, audio_data + audio_len);
    
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
    
    // 5. Build librosa-compatible Slaney Mel filterbank.
    int fft_bins = n_fft / 2 + 1;
    std::vector<float> mel_filters(num_mels * fft_bins, 0.0f);
    
    double fmin = 0.0;
    double fmax = (double)sampling_rate / 2.0;
    
    auto hz_to_mel = [](double hz) {
        constexpr double f_sp = 200.0 / 3.0;
        if (hz < 1000.0) return hz / f_sp;
        constexpr double min_log_mel = 15.0;
        const double logstep = std::log(6.4) / 27.0;
        return min_log_mel + std::log(hz / 1000.0) / logstep;
    };
    auto mel_to_hz = [](double mel) {
        constexpr double f_sp = 200.0 / 3.0;
        if (mel < 15.0) return mel * f_sp;
        const double logstep = std::log(6.4) / 27.0;
        return 1000.0 * std::exp(logstep * (mel - 15.0));
    };
    
    double mel_min = hz_to_mel(fmin);
    double mel_max = hz_to_mel(fmax);
    
    std::vector<double> mel_pts(num_mels + 2);
    std::vector<double> freq_pts(num_mels + 2);
    for (int i = 0; i < num_mels + 2; ++i) {
        mel_pts[i] = mel_min + i * (mel_max - mel_min) / (num_mels + 1);
        freq_pts[i] = mel_to_hz(mel_pts[i]);
    }
    
    for (int m = 0; m < num_mels; ++m) {
        double fs_diff0 = freq_pts[m + 1] - freq_pts[m];
        double fs_diff1 = freq_pts[m + 2] - freq_pts[m + 1];
        
        // Slaney normalization scale
        double slaney_scale = 2.0 / (freq_pts[m + 2] - freq_pts[m]);
        
        for (int k = 0; k < fft_bins; ++k) {
            const double freq_k = static_cast<double>(k) * sampling_rate / n_fft;
            const double lower = (freq_k - freq_pts[m]) / fs_diff0;
            const double upper = (freq_pts[m + 2] - freq_k) / fs_diff1;
            const double weight = std::max(0.0, std::min(lower, upper));
            mel_filters[m * fft_bins + k] = static_cast<float>(weight * slaney_scale);
        }
    }
    
    // 6. Compute STFT and project to Mel scale
    std::vector<std::complex<float>> fft_input(n_fft);
    std::vector<float> spec_data(num_mels * n_frames, 0.0f);
    
    for (int f = 0; f < n_frames; ++f) {
        int offset = f * hop_size;
        
        // Window and pad
        std::fill(fft_input.begin(), fft_input.end(), std::complex<float>(0.0f, 0.0f));
        int win_offset = (n_fft - win_size) / 2;
        for (int i = 0; i < win_size; ++i) {
            fft_input[win_offset + i] = std::complex<float>(padded_audio[offset + win_offset + i] * window[i], 0.0f);
        }
        const std::vector<std::complex<float>> fft_buf = fft_any_length(fft_input);
        
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
