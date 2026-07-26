#pragma once

#include "ops/contracts/activation.h"
#include <cmath>
#include <cstring>

namespace ggml_ops_ext {

// ---- Fused layer norm + activation (optional residual) ----------------------
// srcs: [x, gamma, beta, residual?]; y = act(norm(x [+ residual]) * gamma + beta)
// op_params: { float eps; int32 activation } with activation matching
// ggml_ops_gate_activation (0 = silu, 1 = gelu, 2 = relu, 3 = identity).

struct ops_fused_norm_act_params {
    float eps = 1e-5f;
    int32_t activation = 3;
};
static_assert(sizeof(ops_fused_norm_act_params) == 2 * sizeof(int32_t));

inline bool ops_validate_fused_norm_act(const ops_request& request) {
    if (!request.srcs || request.n_srcs < 3 || !request.srcs[0] || !request.srcs[1] ||
        !request.srcs[2] || !request.params ||
        request.params_size < sizeof(ops_fused_norm_act_params)) return false;
    const ggml_tensor* x = request.srcs[0];
    const ggml_tensor* residual = request.n_srcs > 3 ? request.srcs[3] : nullptr;
    if (!ops_is_float_activation_type(x->type) || request.srcs[1]->type != x->type ||
        request.srcs[2]->type != x->type) return false;
    if (!ggml_is_contiguous(x) || !ggml_is_contiguous(request.srcs[1]) ||
        !ggml_is_contiguous(request.srcs[2])) return false;
    if (residual && (residual->type != x->type || !ggml_are_same_shape(residual, x) ||
                     !ggml_is_contiguous(residual))) return false;
    ops_fused_norm_act_params params;
    std::memcpy(&params, request.params, sizeof(params));
    return std::isfinite(params.eps) && params.eps > 0.0f &&
           params.activation >= 0 && params.activation <= 3 &&
           request.srcs[1]->ne[0] == x->ne[0] && request.srcs[2]->ne[0] == x->ne[0] &&
           ggml_nelements(request.srcs[1]) == ggml_nelements(request.srcs[2]) &&
           (!request.output || (request.output->type == x->type &&
                                ggml_are_same_shape(request.output, x)));
}

// ---- Fused sinusoidal position encoding -------------------------------------
// srcs: [x, position?]; y[2i, t] = x + sin(p / base^(2i/d)),
// y[2i+1, t] = x + cos(p / base^(2i/d)) with p = offset + (*position) + t.
// Applied over dims (0 = feature, 1 = time), broadcast over dims 2 and 3.
// op_params: { float base; int32 offset }.

struct ops_pos_encoding_params {
    float base = 10000.0f;
    int32_t offset = 0;
};
static_assert(sizeof(ops_pos_encoding_params) == 2 * sizeof(int32_t));

inline bool ops_validate_pos_encoding(const ops_request& request) {
    if (!request.srcs || request.n_srcs < 1 || !request.srcs[0] || !request.params ||
        request.params_size < sizeof(ops_pos_encoding_params)) return false;
    const ggml_tensor* x = request.srcs[0];
    const ggml_tensor* position = request.n_srcs > 1 ? request.srcs[1] : nullptr;
    if (!ops_is_float_activation_type(x->type) || !ggml_is_contiguous(x)) return false;
    if (position && (position->type != GGML_TYPE_I32 || ggml_nelements(position) != 1)) return false;
    ops_pos_encoding_params params;
    std::memcpy(&params, request.params, sizeof(params));
    return std::isfinite(params.base) && params.base > 1.0f && params.offset >= 0 &&
           x->ne[0] > 0 && x->ne[1] > 0 &&
           (!request.output || (request.output->type == x->type &&
                                ggml_are_same_shape(request.output, x)));
}

// ---- Fused temperature softmax + top-k/top-p sampling -----------------------
// srcs: [logits (F32 [vocab], contiguous, single row), uniform (F32, 1 element
// in [0, 1) supplied by the host so sampling stays reproducible)].
// Output: I32 [1] sampled index.
// op_params: { int32 top_k (<= 0 disables); float top_p (>= 1 disables);
//              float temperature (> 0) }.

struct ops_sample_dist_params {
    int32_t top_k = 0;
    float top_p = 1.0f;
    float temperature = 1.0f;
};
static_assert(sizeof(ops_sample_dist_params) == 3 * sizeof(int32_t));

inline bool ops_validate_sample_dist(const ops_request& request) {
    if (!request.srcs || request.n_srcs < 2 || !request.srcs[0] || !request.srcs[1] ||
        !request.params || request.params_size < sizeof(ops_sample_dist_params)) return false;
    const ggml_tensor* logits = request.srcs[0];
    const ggml_tensor* uniform = request.srcs[1];
    if (logits->type != GGML_TYPE_F32 || !ggml_is_contiguous(logits) ||
        logits->ne[0] <= 0 || logits->ne[1] != 1 || logits->ne[2] != 1 || logits->ne[3] != 1) {
        return false;
    }
    if (uniform->type != GGML_TYPE_F32 || ggml_nelements(uniform) != 1) return false;
    ops_sample_dist_params params;
    std::memcpy(&params, request.params, sizeof(params));
    return std::isfinite(params.top_p) && std::isfinite(params.temperature) &&
           params.temperature > 0.0f &&
           (!request.output || (request.output->type == GGML_TYPE_I32 &&
                                ggml_nelements(request.output) == 1));
}

} // namespace ggml_ops_ext
