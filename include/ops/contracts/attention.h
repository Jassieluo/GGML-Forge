#pragma once

#include "ops/contracts/activation.h"
#include <cmath>
#include <cstring>

namespace ggml_ops_ext {

inline bool ops_validate_fused_attention(const ops_request& request, bool gpu) {
    if (!request.srcs || request.n_srcs < 5 || !request.srcs[0] || !request.srcs[1] || !request.srcs[2] ||
        !request.params ||
        request.params_size < 2 * sizeof(int32_t)) return false;
    const ggml_tensor* q = request.srcs[0];
    const ggml_tensor* k = request.srcs[1];
    const ggml_tensor* v = request.srcs[2];
    const ggml_tensor* weights = request.srcs[4];
    const auto cache_type = [](ggml_type type) {
        return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 ||
               type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0;
    };
    if (!ops_is_float_activation_type(q->type) || !cache_type(k->type) || !cache_type(v->type)) return false;
    if (q->ne[0] % ggml_blck_size(k->type) != 0 || q->ne[0] % ggml_blck_size(v->type) != 0) return false;
    // The materialized path uses GEMM and currently requires one uniform float type.
    if (weights && (q->type != k->type || q->type != v->type || (gpu && q->type != GGML_TYPE_F32))) return false;
    float scale;
    std::memcpy(&scale, request.params, sizeof(scale));
    if (!std::isfinite(scale) || q->ne[0] <= 0 || q->ne[1] <= 0 || q->ne[2] <= 0 ||
        k->ne[0] != q->ne[0] || v->ne[0] != q->ne[0] || k->ne[1] != v->ne[1] ||
        k->ne[2] <= 0 || k->ne[2] != v->ne[2] || q->ne[2] % k->ne[2] != 0 ||
        q->ne[3] != k->ne[3] || q->ne[3] != v->ne[3]) return false;
    if (request.srcs[3] && ((request.srcs[3]->type != GGML_TYPE_F32 &&
                             (gpu || request.srcs[3]->type != GGML_TYPE_F16)) ||
                            request.srcs[3]->ne[0] != k->ne[1] ||
                            request.srcs[3]->ne[1] != q->ne[1] || request.srcs[3]->ne[2] != q->ne[2])) return false;
    return !weights || (weights->type == q->type && weights->ne[0] == k->ne[1] &&
           weights->ne[1] == q->ne[1] && weights->ne[2] == q->ne[2]);
}

inline bool ops_validate_relative_pe_keys(const ops_request& request) {
    if (!request.srcs || request.n_srcs < 2 || !request.srcs[0] || !request.srcs[1] ||
        !request.params || request.params_size < 2 * sizeof(int32_t)) return false;
    int32_t window_size;
    std::memcpy(&window_size, static_cast<const int32_t*>(request.params) + 1, sizeof(window_size));
    const ggml_tensor* q = request.srcs[0];
    const ggml_tensor* emb = request.srcs[1];
    return ops_is_float_activation_type(q->type) && ops_is_float_activation_type(emb->type) &&
           window_size >= 0 && q->ne[0] == emb->ne[0] && emb->ne[1] >= 2LL * window_size + 1 &&
           ggml_is_contiguous(q) && ggml_is_contiguous(emb);
}

inline bool ops_validate_relative_pe_values(const ops_request& request) {
    if (!request.srcs || request.n_srcs < 2 || !request.srcs[0] || !request.srcs[1] ||
        !request.params || request.params_size < sizeof(int32_t)) return false;
    int32_t window_size;
    std::memcpy(&window_size, request.params, sizeof(window_size));
    const ggml_tensor* weights = request.srcs[0];
    const ggml_tensor* emb = request.srcs[1];
    return ops_is_float_activation_type(weights->type) && ops_is_float_activation_type(emb->type) &&
           window_size >= 0 && weights->ne[0] == weights->ne[1] &&
           emb->ne[1] >= 2LL * window_size + 1 && ggml_is_contiguous(weights) && ggml_is_contiguous(emb);
}

} // namespace ggml_ops_ext
