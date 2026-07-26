#include "common.hpp"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "ops/ops.h"
#include "ops_sycl.h"

#include <cstdio>
#include <cstring>

namespace ggml_ops_ext {
namespace sycl {

class SpectralFFTSYCLKernel;
class SpectralSTFTSYCLKernel;
class SpectralISTFTOverlapSYCLKernel;
class SpectralISTFTNormalizeSYCLKernel;

// Hand-rolled local-memory radix-2 FFT for GGML_OP_OPS_VIRT_FFT / STFT /
// ISTFT. Mirrors src/ops/ops-cpu/stft.cpp: one frame (or one complex row) per
// work-group, two float[n_fft] local-memory arrays (re, im). n_fft is a power
// of two by contract; the supports_ probe in ops_sycl.cpp additionally rejects
// n_fft > ops_sycl_spectral_max_n_fft so the local allocation
// (2 * 4096 * 4 B = 32 KiB) stays inside the 64 KiB SLM of the target iGPU.
// No oneMKL DFT: we do not want its runtime DLLs. Float-only math: the target
// iGPU has no fp64 support.

static inline unsigned spectral_bit_reverse(unsigned v, int log2n) {
    unsigned r = 0;
    for (int i = 0; i < log2n; ++i) {
        r = (r << 1) | (v & 1u);
        v >>= 1;
    }
    return r;
}

// Work-group cooperative in-place radix-2 FFT over local arrays already loaded
// in bit-reversed order. Butterfly pairs within one stage are disjoint, so
// only the inter-stage barrier is needed; the loop bounds are uniform across
// the work-group, so every work-item reaches every barrier. Twiddles come from
// sinpi/cospi on the exact float ratio pos/len (len is a power of two, so the
// division is exact), matching the CPU reference within rounding.
// sign = -1 forward, +1 inverse.
template <typename LocalAcc>
static inline void spectral_workgroup_fft(const ::sycl::nd_item<1>& item, LocalAcc re, LocalAcc im,
                                          int n, float sign) {
    const int tid = static_cast<int>(item.get_local_linear_id());
    const int local_size = static_cast<int>(item.get_local_range(0));
    for (int len = 2; len <= n; len <<= 1) {
        const int half = len >> 1;
        for (int butterfly = tid; butterfly < n / 2; butterfly += local_size) {
            const int pos = butterfly % half;
            const int a = (butterfly / half) * len + pos;
            const int b = a + half;
            const float frac = sign * 2.0f * static_cast<float>(pos) / static_cast<float>(len);
            const float tw_re = ::sycl::cospi(frac);
            const float tw_im = ::sycl::sinpi(frac);
            const float tr = re[b] * tw_re - im[b] * tw_im;
            const float ti = re[b] * tw_im + im[b] * tw_re;
            re[b] = re[a] - tr;
            im[b] = im[a] - ti;
            re[a] += tr;
            im[a] += ti;
        }
        item.barrier(::sycl::access::fence_space::local_space);
    }
}

bool ggml_sycl_op_fft_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    try {
        if (!node || !node->src[0]) {
            return false;
        }
        const ggml_tensor* x = node->src[0];

        ops_fft_params params;
        std::memcpy(&params, node->op_params, sizeof(params));

        ::sycl::queue* queue =
            static_cast<::sycl::queue*>(ggml_ops_ext_bridge_sycl_get_queue(backend));
        if (!queue) {
            return false;
        }

        const int n = static_cast<int>(x->ne[0]);
        const int64_t rows = x->ne[1];
        const int64_t batch = x->ne[3];
        if (n < 2 || n > ops_sycl_spectral_max_n_fft || rows <= 0 || batch <= 0) {
            return false;
        }
        int log2n = 0;
        while ((1 << log2n) < n) {
            ++log2n;
        }
        const bool inverse = params.inverse != 0;
        const float sign = inverse ? 1.0f : -1.0f;
        const float scale = inverse ? 1.0f / static_cast<float>(n) : 1.0f;

        constexpr size_t workgroup_size = 256;
        const size_t workgroups = static_cast<size_t>(rows * batch);
        const float* x_data = static_cast<const float*>(x->data);
        float* dst_data = static_cast<float*>(node->data);

        queue->submit([&](::sycl::handler& handler) {
            ::sycl::local_accessor<float, 1> re_l(::sycl::range<1>(n), handler);
            ::sycl::local_accessor<float, 1> im_l(::sycl::range<1>(n), handler);
            handler.parallel_for<SpectralFFTSYCLKernel>(
                ::sycl::nd_range<1>(workgroups * workgroup_size, workgroup_size),
                [=](::sycl::nd_item<1> item) {
                    const int64_t group = static_cast<int64_t>(item.get_group_linear_id());
                    const int64_t r = group % rows;
                    const int64_t b = group / rows;
                    const int tid = static_cast<int>(item.get_local_linear_id());
                    const float* in_re = x_data + ((b * 2 + 0) * rows + r) * n;
                    const float* in_im = x_data + ((b * 2 + 1) * rows + r) * n;
                    for (int i = tid; i < n; i += static_cast<int>(workgroup_size)) {
                        const unsigned j = spectral_bit_reverse(static_cast<unsigned>(i), log2n);
                        re_l[i] = in_re[j];
                        im_l[i] = in_im[j];
                    }
                    item.barrier(::sycl::access::fence_space::local_space);
                    spectral_workgroup_fft(item, re_l, im_l, n, sign);
                    float* out_re = dst_data + ((b * 2 + 0) * rows + r) * n;
                    float* out_im = dst_data + ((b * 2 + 1) * rows + r) * n;
                    for (int i = tid; i < n; i += static_cast<int>(workgroup_size)) {
                        out_re[i] = re_l[i] * scale;
                        out_im[i] = im_l[i] * scale;
                    }
                });
        });
        return true;
    } catch (const ::sycl::exception& e) {
        std::fprintf(stderr, "SYCL FFT Exception: %s\n", e.what());
        return false;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "SYCL FFT Exception: %s\n", e.what());
        return false;
    }
}

bool ggml_sycl_op_stft_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    try {
        if (!node || !node->src[0] || !node->src[1]) {
            return false;
        }
        const ggml_tensor* signal = node->src[0];
        const ggml_tensor* window = node->src[1];

        ops_stft_params params;
        std::memcpy(&params, node->op_params, sizeof(params));

        ::sycl::queue* queue =
            static_cast<::sycl::queue*>(ggml_ops_ext_bridge_sycl_get_queue(backend));
        if (!queue) {
            return false;
        }

        const int n_fft = params.n_fft;
        const int hop = params.hop;
        const int64_t samples = signal->ne[0];
        const int64_t batch = signal->ne[1];
        const int64_t frames = (samples - n_fft) / hop + 1;
        const int64_t bins = n_fft / 2 + 1;
        if (n_fft < 4 || n_fft > ops_sycl_spectral_max_n_fft || frames <= 0 || batch <= 0) {
            return false;
        }
        int log2n = 0;
        while ((1 << log2n) < n_fft) {
            ++log2n;
        }

        constexpr size_t workgroup_size = 256;
        const size_t workgroups = static_cast<size_t>(frames * batch);
        const float* signal_data = static_cast<const float*>(signal->data);
        const float* window_data = static_cast<const float*>(window->data);
        float* dst_data = static_cast<float*>(node->data);

        queue->submit([&](::sycl::handler& handler) {
            ::sycl::local_accessor<float, 1> re_l(::sycl::range<1>(n_fft), handler);
            ::sycl::local_accessor<float, 1> im_l(::sycl::range<1>(n_fft), handler);
            handler.parallel_for<SpectralSTFTSYCLKernel>(
                ::sycl::nd_range<1>(workgroups * workgroup_size, workgroup_size),
                [=](::sycl::nd_item<1> item) {
                    const int64_t group = static_cast<int64_t>(item.get_group_linear_id());
                    const int64_t f = group % frames;
                    const int64_t b = group / frames;
                    const int tid = static_cast<int>(item.get_local_linear_id());
                    const float* frame = signal_data + b * samples + f * hop;
                    for (int i = tid; i < n_fft; i += static_cast<int>(workgroup_size)) {
                        const unsigned j = spectral_bit_reverse(static_cast<unsigned>(i), log2n);
                        re_l[i] = frame[j] * window_data[j];
                        im_l[i] = 0.0f;
                    }
                    item.barrier(::sycl::access::fence_space::local_space);
                    spectral_workgroup_fft(item, re_l, im_l, n_fft, -1.0f);
                    float* out_re = dst_data + ((b * 2 + 0) * frames + f) * bins;
                    float* out_im = dst_data + ((b * 2 + 1) * frames + f) * bins;
                    for (int k = tid; k < static_cast<int>(bins);
                         k += static_cast<int>(workgroup_size)) {
                        out_re[k] = re_l[k];
                        out_im[k] = im_l[k];
                    }
                });
        });
        return true;
    } catch (const ::sycl::exception& e) {
        std::fprintf(stderr, "SYCL STFT Exception: %s\n", e.what());
        return false;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "SYCL STFT Exception: %s\n", e.what());
        return false;
    }
}

bool ggml_sycl_op_istft_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    try {
        if (!node || !node->src[0] || !node->src[1]) {
            return false;
        }
        const ggml_tensor* spectrum = node->src[0];
        const ggml_tensor* window = node->src[1];

        ops_stft_params params;
        std::memcpy(&params, node->op_params, sizeof(params));

        ::sycl::queue* queue =
            static_cast<::sycl::queue*>(ggml_ops_ext_bridge_sycl_get_queue(backend));
        if (!queue) {
            return false;
        }

        const int n_fft = params.n_fft;
        const int hop = params.hop;
        const int64_t bins = n_fft / 2 + 1;
        const int64_t frames = spectrum->ne[1];
        const int64_t batch = spectrum->ne[3];
        const int64_t samples = (frames - 1) * hop + n_fft;
        if (n_fft < 4 || n_fft > ops_sycl_spectral_max_n_fft || frames <= 0 || batch <= 0) {
            return false;
        }
        int log2n = 0;
        while ((1 << log2n) < n_fft) {
            ++log2n;
        }

        constexpr size_t workgroup_size = 256;
        const size_t workgroups = static_cast<size_t>(frames * batch);
        const int64_t total = samples * batch;
        const float* spec_data = static_cast<const float*>(spectrum->data);
        const float* window_data = static_cast<const float*>(window->data);
        float* dst_data = static_cast<float*>(node->data);

        // Pass 0: zero the output so the overlap-add can accumulate with
        // atomics. The queue is in-order, so the passes serialize.
        queue->fill(dst_data, 0.0f, static_cast<size_t>(total));

        // Pass 1: Hermitian rebuild + inverse FFT + windowed overlap-add.
        queue->submit([&](::sycl::handler& handler) {
            ::sycl::local_accessor<float, 1> re_l(::sycl::range<1>(n_fft), handler);
            ::sycl::local_accessor<float, 1> im_l(::sycl::range<1>(n_fft), handler);
            handler.parallel_for<SpectralISTFTOverlapSYCLKernel>(
                ::sycl::nd_range<1>(workgroups * workgroup_size, workgroup_size),
                [=](::sycl::nd_item<1> item) {
                    const int64_t group = static_cast<int64_t>(item.get_group_linear_id());
                    const int64_t f = group % frames;
                    const int64_t b = group / frames;
                    const int tid = static_cast<int>(item.get_local_linear_id());
                    const float* row_re = spec_data + ((b * 2 + 0) * frames + f) * bins;
                    const float* row_im = spec_data + ((b * 2 + 1) * frames + f) * bins;
                    for (int i = tid; i < n_fft; i += static_cast<int>(workgroup_size)) {
                        const unsigned j = spectral_bit_reverse(static_cast<unsigned>(i), log2n);
                        if (j < static_cast<unsigned>(bins)) {
                            re_l[i] = row_re[j];
                            im_l[i] = row_im[j];
                        } else {
                            re_l[i] = row_re[n_fft - j];
                            im_l[i] = -row_im[n_fft - j];
                        }
                    }
                    item.barrier(::sycl::access::fence_space::local_space);
                    spectral_workgroup_fft(item, re_l, im_l, n_fft, 1.0f);
                    const float inv_size = 1.0f / static_cast<float>(n_fft);
                    float* target = dst_data + b * samples + f * hop;
                    for (int i = tid; i < n_fft; i += static_cast<int>(workgroup_size)) {
                        ::sycl::atomic_ref<float, ::sycl::memory_order::relaxed,
                                           ::sycl::memory_scope::device,
                                           ::sycl::access::address_space::global_space>
                            accumulator(target[i]);
                        accumulator.fetch_add(re_l[i] * inv_size * window_data[i]);
                    }
                });
        });

        // Pass 2: divide by the squared-window overlap envelope (NOLA),
        // computed on the fly per sample. The frame loop runs in ascending
        // order so the float summation matches the CPU reference exactly.
        constexpr size_t norm_local_size = 256;
        const size_t norm_global_size =
            (static_cast<size_t>(total) + norm_local_size - 1) / norm_local_size *
            norm_local_size;
        queue->submit([&](::sycl::handler& handler) {
            handler.parallel_for<SpectralISTFTNormalizeSYCLKernel>(
                ::sycl::nd_range<1>(norm_global_size, norm_local_size),
                [=](::sycl::nd_item<1> item) {
                    const int64_t idx = static_cast<int64_t>(item.get_global_linear_id());
                    if (idx >= total) {
                        return;
                    }
                    const int64_t t = idx % samples;
                    const int64_t f_lo = t >= n_fft ? (t - n_fft) / hop + 1 : 0;
                    int64_t f_hi = t / hop;
                    if (f_hi > frames - 1) {
                        f_hi = frames - 1;
                    }
                    float envelope = 0.0f;
                    for (int64_t f = f_lo; f <= f_hi; ++f) {
                        const float w = window_data[t - f * hop];
                        envelope += w * w;
                    }
                    dst_data[idx] = envelope > 1e-10f ? dst_data[idx] / envelope : 0.0f;
                });
        });
        return true;
    } catch (const ::sycl::exception& e) {
        std::fprintf(stderr, "SYCL iSTFT Exception: %s\n", e.what());
        return false;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "SYCL iSTFT Exception: %s\n", e.what());
        return false;
    }
}

} // namespace sycl
} // namespace ggml_ops_ext
