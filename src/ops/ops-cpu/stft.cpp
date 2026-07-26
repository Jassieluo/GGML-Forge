#include "ops/cpu.h"
#include "ops/ops.h"
#include "ops_cpu_common.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace ggml_ops_ext::cpu {
namespace {

// In-place iterative radix-2 complex FFT (size is a power of two; enforced by
// the contract). inverse=true computes the unscaled inverse transform; the
// caller divides by size.
void fft_radix2(float* re, float* im, int size, bool inverse) {
    for (int i = 1, j = 0; i < size; ++i) {
        int bit = size >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j |= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    const float sign = inverse ? 1.0f : -1.0f;
    for (int len = 2; len <= size; len <<= 1) {
        const double angle = sign * 2.0 * 3.14159265358979323846 / len;
        const float wr = static_cast<float>(std::cos(angle));
        const float wi = static_cast<float>(std::sin(angle));
        for (int start = 0; start < size; start += len) {
            float cr = 1.0f, ci = 0.0f;
            for (int k = 0; k < len / 2; ++k) {
                const int a = start + k;
                const int b = start + k + len / 2;
                const float tr = re[b] * cr - im[b] * ci;
                const float ti = re[b] * ci + im[b] * cr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                const float next_cr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = next_cr;
            }
        }
    }
}

} // namespace

bool ops_cpu_op_stft(ggml_backend_t backend, ggml_tensor* node) {
    const ggml_tensor* signal = node->src[0];
    const ggml_tensor* window = node->src[1];
    ops_stft_params params;
    std::memcpy(&params, node->op_params, sizeof(params));

    const int n_fft = params.n_fft;
    const int hop = params.hop;
    const int64_t bins = n_fft / 2 + 1;
    const int64_t samples = signal->ne[0];
    const int64_t batch = signal->ne[1];
    const int64_t frames = (samples - n_fft) / hop + 1;
    const float* x = static_cast<const float*>(signal->data);
    const float* w = static_cast<const float*>(window->data);
    float* out = static_cast<float*>(node->data);
    const int threads = backend_thread_count(backend);

#pragma omp parallel for collapse(2) num_threads(threads) schedule(static)
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t f = 0; f < frames; ++f) {
            std::vector<float> re(static_cast<size_t>(n_fft));
            std::vector<float> im(static_cast<size_t>(n_fft), 0.0f);
            const float* frame = x + b * samples + f * hop;
            for (int i = 0; i < n_fft; ++i) re[i] = frame[i] * w[i];
            fft_radix2(re.data(), im.data(), n_fft, false);
            float* out_re = out + ((b * 2 + 0) * frames + f) * bins;
            float* out_im = out + ((b * 2 + 1) * frames + f) * bins;
            for (int64_t k = 0; k < bins; ++k) {
                out_re[k] = re[k];
                out_im[k] = im[k];
            }
        }
    }
    return true;
}

bool ops_cpu_op_fft(ggml_backend_t backend, ggml_tensor* node) {
    const ggml_tensor* x = node->src[0];
    ops_fft_params params;
    std::memcpy(&params, node->op_params, sizeof(params));

    const int n = static_cast<int>(x->ne[0]);
    const int64_t rows = x->ne[1];
    const int64_t batch = x->ne[3];
    const float* in = static_cast<const float*>(x->data);
    float* out = static_cast<float*>(node->data);
    const int threads = backend_thread_count(backend);
    const bool inverse = params.inverse != 0;
    const float scale = inverse ? 1.0f / static_cast<float>(n) : 1.0f;

#pragma omp parallel for collapse(2) num_threads(threads) schedule(static)
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t r = 0; r < rows; ++r) {
            std::vector<float> re(static_cast<size_t>(n));
            std::vector<float> im(static_cast<size_t>(n));
            const float* in_re = in + ((b * 2 + 0) * rows + r) * n;
            const float* in_im = in + ((b * 2 + 1) * rows + r) * n;
            std::memcpy(re.data(), in_re, static_cast<size_t>(n) * sizeof(float));
            std::memcpy(im.data(), in_im, static_cast<size_t>(n) * sizeof(float));
            fft_radix2(re.data(), im.data(), n, inverse);
            float* out_re = out + ((b * 2 + 0) * rows + r) * n;
            float* out_im = out + ((b * 2 + 1) * rows + r) * n;
            for (int i = 0; i < n; ++i) {
                out_re[i] = re[i] * scale;
                out_im[i] = im[i] * scale;
            }
        }
    }
    return true;
}

bool ops_cpu_op_istft(ggml_backend_t backend, ggml_tensor* node) {
    const ggml_tensor* spectrum = node->src[0];
    const ggml_tensor* window = node->src[1];
    ops_stft_params params;
    std::memcpy(&params, node->op_params, sizeof(params));

    const int n_fft = params.n_fft;
    const int hop = params.hop;
    const int64_t bins = n_fft / 2 + 1;
    const int64_t frames = spectrum->ne[1];
    const int64_t batch = spectrum->ne[3];
    const int64_t samples = (frames - 1) * hop + n_fft;
    const float* spec = static_cast<const float*>(spectrum->data);
    const float* w = static_cast<const float*>(window->data);
    float* out = static_cast<float*>(node->data);
    const int threads = backend_thread_count(backend);

    // Squared-window overlap envelope (NOLA normalization), shared by batches.
    std::vector<float> envelope(static_cast<size_t>(samples), 0.0f);
    for (int64_t f = 0; f < frames; ++f) {
        for (int i = 0; i < n_fft; ++i) {
            envelope[f * hop + i] += w[i] * w[i];
        }
    }

#pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t b = 0; b < batch; ++b) {
        float* out_row = out + b * samples;
        std::memset(out_row, 0, static_cast<size_t>(samples) * sizeof(float));
        std::vector<float> re(static_cast<size_t>(n_fft));
        std::vector<float> im(static_cast<size_t>(n_fft));
        const float* spec_re = spec + (b * 2 + 0) * frames * bins;
        const float* spec_im = spec + (b * 2 + 1) * frames * bins;
        for (int64_t f = 0; f < frames; ++f) {
            const float* row_re = spec_re + f * bins;
            const float* row_im = spec_im + f * bins;
            // Rebuild the Hermitian-symmetric spectrum of the real frame.
            for (int64_t k = 0; k < bins; ++k) {
                re[k] = row_re[k];
                im[k] = row_im[k];
            }
            for (int64_t k = bins; k < n_fft; ++k) {
                re[k] = row_re[n_fft - k];
                im[k] = -row_im[n_fft - k];
            }
            fft_radix2(re.data(), im.data(), n_fft, true);
            const float inv_size = 1.0f / static_cast<float>(n_fft);
            float* target = out_row + f * hop;
            for (int i = 0; i < n_fft; ++i) {
                target[i] += re[i] * inv_size * w[i];
            }
        }
        for (int64_t t = 0; t < samples; ++t) {
            out_row[t] = envelope[t] > 1e-10f ? out_row[t] / envelope[t] : 0.0f;
        }
    }
    return true;
}

} // namespace ggml_ops_ext::cpu
