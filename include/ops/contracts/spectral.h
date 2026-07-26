#pragma once

#include "ops/types.h"
#include <cstdint>
#include <cstring>

namespace ggml_ops_ext {

// ---- STFT / iSTFT -----------------------------------------------------------
// Complex spectra are stored as two F32 planes (ggml has no complex type):
// dim 2 indexes {0 = real, 1 = imag}.
//
// STFT  srcs: [signal, window]
//   signal F32 [samples, batch]   (contiguous, samples >= n_fft)
//   window F32 [n_fft]
//   output F32 [n_bins, frames, 2, batch]
//     n_bins = n_fft / 2 + 1, frames = (samples - n_fft) / hop + 1.
//   No implicit center padding: callers pad the signal themselves so the
//   contract stays a pure frame transform.
//
// ISTFT srcs: [spectrum, window]
//   spectrum F32 [n_bins, frames, 2, batch]
//   window   F32 [n_fft]
//   output   F32 [samples, batch], samples = (frames - 1) * hop + n_fft.
//   Kernels apply the synthesis window and normalize by the squared-window
//   overlap envelope (NOLA), matching torch.istft(center=false).
//
// op_params: { int32 n_fft; int32 hop }. n_fft must be a power of two so
// every backend can run an in-place radix-2 transform; hop in [1, n_fft].

struct ops_stft_params {
    int32_t n_fft = 0;
    int32_t hop = 0;
};
static_assert(sizeof(ops_stft_params) == 2 * sizeof(int32_t));

// ---- Batched 1-D complex FFT ------------------------------------------------
// srcs: [x] with x F32 [n, rows, 2, batch]; the transform runs along dim 0
// for every (row, batch) pair, dim 2 indexes {0 = real, 1 = imag}. n must be
// a power of two. inverse != 0 computes the inverse transform including the
// 1/n scale. The DFT of real input is the STFT special case
// (window = ones, hop = n_fft).
// op_params: { int32 inverse }.

struct ops_fft_params {
    int32_t inverse = 0;
};
static_assert(sizeof(ops_fft_params) == sizeof(int32_t));

inline bool ops_validate_fft(const ops_request& request) {
    if (!request.srcs || request.n_srcs < 1 || !request.srcs[0] || !request.params ||
        request.params_size < sizeof(ops_fft_params)) return false;
    const ggml_tensor* x = request.srcs[0];
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x)) return false;
    const int64_t n = x->ne[0];
    if (n < 2 || (n & (n - 1)) != 0 || x->ne[2] != 2) return false;
    if (!request.output) return true;
    return request.output->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(request.output, x);
}

inline bool ops_stft_params_valid(const ops_stft_params& params) {
    return params.n_fft >= 4 && (params.n_fft & (params.n_fft - 1)) == 0 &&
           params.hop >= 1 && params.hop <= params.n_fft;
}

inline bool ops_validate_stft(const ops_request& request) {
    if (!request.srcs || request.n_srcs < 2 || !request.srcs[0] || !request.srcs[1] ||
        !request.params || request.params_size < sizeof(ops_stft_params)) return false;
    ops_stft_params params;
    std::memcpy(&params, request.params, sizeof(params));
    if (!ops_stft_params_valid(params)) return false;
    const ggml_tensor* signal = request.srcs[0];
    const ggml_tensor* window = request.srcs[1];
    if (signal->type != GGML_TYPE_F32 || window->type != GGML_TYPE_F32) return false;
    if (!ggml_is_contiguous(signal) || !ggml_is_contiguous(window)) return false;
    if (signal->ne[2] != 1 || signal->ne[3] != 1) return false;
    if (window->ne[0] != params.n_fft || ggml_nelements(window) != params.n_fft) return false;
    const int64_t samples = signal->ne[0];
    if (samples < params.n_fft) return false;
    const int64_t frames = (samples - params.n_fft) / params.hop + 1;
    const int64_t bins = params.n_fft / 2 + 1;
    if (!request.output) return true;
    return request.output->type == GGML_TYPE_F32 &&
           request.output->ne[0] == bins && request.output->ne[1] == frames &&
           request.output->ne[2] == 2 && request.output->ne[3] == signal->ne[1];
}

inline bool ops_validate_istft(const ops_request& request) {
    if (!request.srcs || request.n_srcs < 2 || !request.srcs[0] || !request.srcs[1] ||
        !request.params || request.params_size < sizeof(ops_stft_params)) return false;
    ops_stft_params params;
    std::memcpy(&params, request.params, sizeof(params));
    if (!ops_stft_params_valid(params)) return false;
    const ggml_tensor* spectrum = request.srcs[0];
    const ggml_tensor* window = request.srcs[1];
    if (spectrum->type != GGML_TYPE_F32 || window->type != GGML_TYPE_F32) return false;
    if (!ggml_is_contiguous(spectrum) || !ggml_is_contiguous(window)) return false;
    const int64_t bins = params.n_fft / 2 + 1;
    if (spectrum->ne[0] != bins || spectrum->ne[2] != 2) return false;
    if (window->ne[0] != params.n_fft || ggml_nelements(window) != params.n_fft) return false;
    const int64_t frames = spectrum->ne[1];
    if (frames < 1) return false;
    const int64_t samples = (frames - 1) * params.hop + params.n_fft;
    if (!request.output) return true;
    return request.output->type == GGML_TYPE_F32 &&
           request.output->ne[0] == samples && request.output->ne[1] == spectrum->ne[3] &&
           request.output->ne[2] == 1 && request.output->ne[3] == 1;
}

} // namespace ggml_ops_ext
