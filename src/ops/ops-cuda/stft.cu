#include "ops_cuda_common.cuh"

#include <cstdint>
#include <cstring>

namespace ggml_ops_ext {
namespace cuda {

// Hand-rolled shared-memory radix-2 FFT for GGML_OP_OPS_VIRT_FFT / STFT /
// ISTFT. Mirrors src/ops/ops-cpu/stft.cpp: one frame (or one complex row) per
// block, two float[n_fft] arrays (re, im) in dynamic shared memory. n_fft is a
// power of two by contract; the supports_ probe in ops_cuda.cu additionally
// rejects n_fft > OPS_SPECTRAL_MAX_N_FFT so the shared allocation
// (2 * 4096 * 4 B = 32 KiB) stays inside the 48 KiB default limit. No cuFFT:
// we do not want its runtime DLLs.
constexpr int64_t OPS_SPECTRAL_MAX_N_FFT = 4096;

static __device__ __forceinline__ unsigned spectral_bit_reverse(unsigned v, int log2n) {
    return __brev(v) >> (32 - log2n);
}

// Block-cooperative in-place radix-2 FFT over shared arrays already loaded in
// bit-reversed order. Butterfly pairs within one stage are disjoint, so only
// the inter-stage barrier is needed. Twiddles come from sincospif on the exact
// float ratio pos/len (len is a power of two, so the division is exact),
// matching the CPU reference within rounding. sign = -1 forward, +1 inverse.
static __device__ void spectral_block_fft(float* re, float* im, int n, float sign) {
    for (int len = 2; len <= n; len <<= 1) {
        const int half = len >> 1;
        for (int butterfly = threadIdx.x; butterfly < n / 2; butterfly += blockDim.x) {
            const int pos = butterfly % half;
            const int a = (butterfly / half) * len + pos;
            const int b = a + half;
            float tw_im, tw_re;
            sincospif(sign * 2.0f * (float)pos / (float)len, &tw_im, &tw_re);
            const float tr = re[b] * tw_re - im[b] * tw_im;
            const float ti = re[b] * tw_im + im[b] * tw_re;
            re[b] = re[a] - tr;
            im[b] = im[a] - ti;
            re[a] += tr;
            im[a] += ti;
        }
        __syncthreads();
    }
}

// x, dst F32 [n, rows, 2, batch]; one block per (row, batch), transform along
// dim 0, dim 2 indexes {0 = real, 1 = imag}. scale is 1/n for the inverse.
__global__ void spectral_fft_kernel(const float* x, float* dst, int n, int log2n, int64_t rows,
                                    float sign, float scale) {
    extern __shared__ float smem[];
    float* re = smem;
    float* im = smem + n;
    const int64_t r = blockIdx.x;
    const int64_t b = blockIdx.y;
    const float* in_re = x + ((b * 2 + 0) * rows + r) * n;
    const float* in_im = x + ((b * 2 + 1) * rows + r) * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const unsigned j = spectral_bit_reverse((unsigned)i, log2n);
        re[i] = in_re[j];
        im[i] = in_im[j];
    }
    __syncthreads();
    spectral_block_fft(re, im, n, sign);
    float* out_re = dst + ((b * 2 + 0) * rows + r) * n;
    float* out_im = dst + ((b * 2 + 1) * rows + r) * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out_re[i] = re[i] * scale;
        out_im[i] = im[i] * scale;
    }
}

// signal F32 [samples, batch] + window F32 [n_fft] ->
// spectrum F32 [bins, frames, 2, batch]; one block per (frame, batch). The
// window is applied before the forward transform.
__global__ void spectral_stft_kernel(const float* signal, const float* window, float* dst,
                                     int n_fft, int log2n, int hop, int64_t samples,
                                     int64_t frames, int64_t bins) {
    extern __shared__ float smem[];
    float* re = smem;
    float* im = smem + n_fft;
    const int64_t f = blockIdx.x;
    const int64_t b = blockIdx.y;
    const float* frame = signal + b * samples + f * hop;
    for (int i = threadIdx.x; i < n_fft; i += blockDim.x) {
        const unsigned j = spectral_bit_reverse((unsigned)i, log2n);
        re[i] = frame[j] * window[j];
        im[i] = 0.0f;
    }
    __syncthreads();
    spectral_block_fft(re, im, n_fft, -1.0f);
    float* out_re = dst + ((b * 2 + 0) * frames + f) * bins;
    float* out_im = dst + ((b * 2 + 1) * frames + f) * bins;
    for (int k = threadIdx.x; k < (int)bins; k += blockDim.x) {
        out_re[k] = re[k];
        out_im[k] = im[k];
    }
}

// One block per (frame, batch): rebuild the Hermitian-symmetric spectrum,
// inverse FFT, apply the synthesis window and overlap-add into dst (zeroed
// beforehand) with atomicAdd for the cross-frame accumulation.
__global__ void spectral_istft_overlap_kernel(const float* spec, const float* window, float* dst,
                                              int n_fft, int log2n, int hop, int64_t bins,
                                              int64_t frames, int64_t samples) {
    extern __shared__ float smem[];
    float* re = smem;
    float* im = smem + n_fft;
    const int64_t f = blockIdx.x;
    const int64_t b = blockIdx.y;
    const float* row_re = spec + ((b * 2 + 0) * frames + f) * bins;
    const float* row_im = spec + ((b * 2 + 1) * frames + f) * bins;
    for (int i = threadIdx.x; i < n_fft; i += blockDim.x) {
        const unsigned j = spectral_bit_reverse((unsigned)i, log2n);
        if (j < (unsigned)bins) {
            re[i] = row_re[j];
            im[i] = row_im[j];
        } else {
            re[i] = row_re[n_fft - j];
            im[i] = -row_im[n_fft - j];
        }
    }
    __syncthreads();
    spectral_block_fft(re, im, n_fft, 1.0f);
    const float inv_size = 1.0f / (float)n_fft;
    float* target = dst + b * samples + f * hop;
    for (int i = threadIdx.x; i < n_fft; i += blockDim.x) {
        atomicAdd(&target[i], re[i] * inv_size * window[i]);
    }
}

// Grid-stride pass over [samples, batch]: divide by the squared-window overlap
// envelope (NOLA), computed on the fly per sample. The frame loop runs in
// ascending order so the float summation matches the CPU reference exactly.
__global__ void spectral_istft_normalize_kernel(const float* window, float* dst, int n_fft,
                                                int hop, int64_t frames, int64_t samples,
                                                int64_t total) {
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += stride) {
        const int64_t t = idx % samples;
        const int64_t f_lo = t >= n_fft ? (t - n_fft) / hop + 1 : 0;
        int64_t f_hi = t / hop;
        if (f_hi > frames - 1) {
            f_hi = frames - 1;
        }
        float envelope = 0.0f;
        for (int64_t f = f_lo; f <= f_hi; ++f) {
            const float w = window[t - f * hop];
            envelope += w * w;
        }
        dst[idx] = envelope > 1e-10f ? dst[idx] / envelope : 0.0f;
    }
}

bool ggml_cuda_op_fft_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    const ggml_tensor* x = node->src[0];
    ops_fft_params params;
    std::memcpy(&params, node->op_params, sizeof(params));

    int device = ggml_ops_ext_bridge_cuda_get_device(backend);
    cudaStream_t stream = (cudaStream_t)ggml_ops_ext_bridge_cuda_get_stream(backend);
    CUDA_CHECK(cudaSetDevice(device));

    const int n = (int)x->ne[0];
    const int64_t rows = x->ne[1];
    const int64_t batch = x->ne[3];
    if (n < 2 || n > OPS_SPECTRAL_MAX_N_FFT || rows <= 0 || batch <= 0) {
        fprintf(stderr, "CUDA FFT: unsupported size n=%d\n", n);
        return false;
    }
    int log2n = 0;
    while ((1 << log2n) < n) {
        ++log2n;
    }
    const bool inverse = params.inverse != 0;
    const float sign = inverse ? 1.0f : -1.0f;
    const float scale = inverse ? 1.0f / (float)n : 1.0f;

    constexpr int threads = 256;
    const size_t shared_size = 2 * (size_t)n * sizeof(float);
    const dim3 blocks((unsigned)rows, (unsigned)batch);
    spectral_fft_kernel<<<blocks, threads, shared_size, stream>>>(
        (const float*)x->data, (float*)node->data, n, log2n, rows, sign, scale);

    return cudaGetLastError() == cudaSuccess;
}

bool ggml_cuda_op_stft_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    const ggml_tensor* signal = node->src[0];
    const ggml_tensor* window = node->src[1];
    ops_stft_params params;
    std::memcpy(&params, node->op_params, sizeof(params));

    int device = ggml_ops_ext_bridge_cuda_get_device(backend);
    cudaStream_t stream = (cudaStream_t)ggml_ops_ext_bridge_cuda_get_stream(backend);
    CUDA_CHECK(cudaSetDevice(device));

    const int n_fft = params.n_fft;
    const int hop = params.hop;
    const int64_t samples = signal->ne[0];
    const int64_t batch = signal->ne[1];
    const int64_t frames = (samples - n_fft) / hop + 1;
    const int64_t bins = n_fft / 2 + 1;
    if (n_fft < 4 || n_fft > OPS_SPECTRAL_MAX_N_FFT || frames <= 0 || batch <= 0) {
        fprintf(stderr, "CUDA STFT: unsupported n_fft=%d\n", n_fft);
        return false;
    }
    int log2n = 0;
    while ((1 << log2n) < n_fft) {
        ++log2n;
    }

    constexpr int threads = 256;
    const size_t shared_size = 2 * (size_t)n_fft * sizeof(float);
    const dim3 blocks((unsigned)frames, (unsigned)batch);
    spectral_stft_kernel<<<blocks, threads, shared_size, stream>>>(
        (const float*)signal->data, (const float*)window->data, (float*)node->data, n_fft, log2n,
        hop, samples, frames, bins);

    return cudaGetLastError() == cudaSuccess;
}

bool ggml_cuda_op_istft_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    const ggml_tensor* spectrum = node->src[0];
    const ggml_tensor* window = node->src[1];
    ops_stft_params params;
    std::memcpy(&params, node->op_params, sizeof(params));

    int device = ggml_ops_ext_bridge_cuda_get_device(backend);
    cudaStream_t stream = (cudaStream_t)ggml_ops_ext_bridge_cuda_get_stream(backend);
    CUDA_CHECK(cudaSetDevice(device));

    const int n_fft = params.n_fft;
    const int hop = params.hop;
    const int64_t bins = n_fft / 2 + 1;
    const int64_t frames = spectrum->ne[1];
    const int64_t batch = spectrum->ne[3];
    const int64_t samples = (frames - 1) * hop + n_fft;
    if (n_fft < 4 || n_fft > OPS_SPECTRAL_MAX_N_FFT || frames <= 0 || batch <= 0) {
        fprintf(stderr, "CUDA iSTFT: unsupported n_fft=%d\n", n_fft);
        return false;
    }
    int log2n = 0;
    while ((1 << log2n) < n_fft) {
        ++log2n;
    }

    // Pass 0: zero the output so the overlap-add can accumulate with atomics.
    const int64_t total = samples * batch;
    if (cudaMemsetAsync(node->data, 0, (size_t)total * sizeof(float), stream) != cudaSuccess) {
        fprintf(stderr, "CUDA iSTFT: failed to zero the output\n");
        return false;
    }

    constexpr int threads = 256;
    const size_t shared_size = 2 * (size_t)n_fft * sizeof(float);
    const dim3 blocks((unsigned)frames, (unsigned)batch);
    spectral_istft_overlap_kernel<<<blocks, threads, shared_size, stream>>>(
        (const float*)spectrum->data, (const float*)window->data, (float*)node->data, n_fft,
        log2n, hop, bins, frames, samples);
    if (cudaGetLastError() != cudaSuccess) {
        return false;
    }

    const int64_t want_blocks = (total + threads - 1) / threads;
    const unsigned norm_blocks = (unsigned)(want_blocks > 65535 ? 65535 : want_blocks);
    spectral_istft_normalize_kernel<<<norm_blocks, threads, 0, stream>>>(
        (const float*)window->data, (float*)node->data, n_fft, hop, frames, samples, total);

    return cudaGetLastError() == cudaSuccess;
}

} // namespace cuda
} // namespace ggml_ops_ext
