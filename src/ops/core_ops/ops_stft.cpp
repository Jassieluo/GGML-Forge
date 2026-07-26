#include "ops/ops.h"
#include "ggml-impl.h"
#include <cstring>

// Kernel-required builders: composing a DFT from native nodes needs the basis
// matrix as bound constant data, which graph builders cannot allocate. Callers
// that want a composed fallback build it at the nn::functional tier where a
// Context is available to hold the basis weights.

struct ggml_tensor* ggml_ops_stft(
    struct ggml_context* ctx,
    struct ggml_tensor* signal,
    struct ggml_tensor* window,
    int32_t n_fft,
    int32_t hop,
    ggml_backend_t backend
) {
    if (!signal || !window) return nullptr;
    ggml_ops_ext::ops_stft_params params;
    params.n_fft = n_fft;
    params.hop = hop;
    if (!ggml_ops_ext::ops_stft_params_valid(params)) return nullptr;
    struct ggml_tensor* srcs[] = { signal, window };

    if (!ggml_ops_backend_supports_op(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_STFT,
                                      srcs, 2, &params, sizeof(params))) {
        return nullptr;
    }

    const int64_t frames = (signal->ne[0] - n_fft) / hop + 1;
    const int64_t ne[4] = { n_fft / 2 + 1, frames, 2, signal->ne[1] };
    struct ggml_tensor* result = ggml_ops_ext::ops_new_virtual_node(
        ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_STFT, GGML_TYPE_F32, 4, ne, 2, srcs);
    ggml_set_op_params(result, &params, sizeof(params));
    return result;
}

struct ggml_tensor* ggml_ops_istft(
    struct ggml_context* ctx,
    struct ggml_tensor* spectrum,
    struct ggml_tensor* window,
    int32_t n_fft,
    int32_t hop,
    ggml_backend_t backend
) {
    if (!spectrum || !window) return nullptr;
    ggml_ops_ext::ops_stft_params params;
    params.n_fft = n_fft;
    params.hop = hop;
    if (!ggml_ops_ext::ops_stft_params_valid(params)) return nullptr;
    struct ggml_tensor* srcs[] = { spectrum, window };

    if (!ggml_ops_backend_supports_op(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_ISTFT,
                                      srcs, 2, &params, sizeof(params))) {
        return nullptr;
    }

    const int64_t samples = (spectrum->ne[1] - 1) * hop + n_fft;
    const int64_t ne[2] = { samples, spectrum->ne[3] };
    struct ggml_tensor* result = ggml_ops_ext::ops_new_virtual_node(
        ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_ISTFT, GGML_TYPE_F32, 2, ne, 2, srcs);
    ggml_set_op_params(result, &params, sizeof(params));
    return result;
}

struct ggml_tensor* ggml_ops_fft(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    bool inverse,
    ggml_backend_t backend
) {
    if (!x) return nullptr;
    ggml_ops_ext::ops_fft_params params;
    params.inverse = inverse ? 1 : 0;
    struct ggml_tensor* srcs[] = { x };

    if (!ggml_ops_backend_supports_op(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_FFT,
                                      srcs, 1, &params, sizeof(params))) {
        return nullptr;
    }

    struct ggml_tensor* result = ggml_ops_ext::ops_new_virtual_node(
        ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_FFT, GGML_TYPE_F32, 4, x->ne, 1, srcs);
    ggml_set_op_params(result, &params, sizeof(params));
    return result;
}
