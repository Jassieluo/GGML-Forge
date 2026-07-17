#pragma once

#include "ops/types.h"
#include <cmath>
#include <cstring>

namespace ggml_ops_ext {

inline bool ops_is_float_activation_type(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16;
}

inline bool ops_validate_unary_activation(const ops_request& request) {
    return request.srcs && request.n_srcs >= 1 && request.srcs[0] &&
           ops_is_float_activation_type(request.srcs[0]->type);
}

inline bool ops_validate_snake(const ops_request& request) {
    if (!ops_validate_unary_activation(request) || !request.params || request.params_size < sizeof(float)) return false;
    float alpha;
    std::memcpy(&alpha, request.params, sizeof(alpha));
    return std::isfinite(alpha);
}

inline bool ops_validate_gated_tanh_sigmoid(const ops_request& request) {
    if (!ops_validate_unary_activation(request) || !request.params || request.params_size < sizeof(int32_t)) return false;
    int32_t hidden_channels;
    std::memcpy(&hidden_channels, request.params, sizeof(hidden_channels));
    return hidden_channels > 0 && request.srcs[0]->ne[0] == 2LL * hidden_channels;
}

inline bool ops_validate_glu(const ops_request& request) {
    if (!ops_validate_unary_activation(request)) return false;
    const ggml_tensor* x = request.srcs[0];
    return x->ne[0] > 0 && x->ne[0] % 2 == 0 && ggml_is_contiguous(x);
}

inline bool ops_validate_snake_beta(const ops_request& request) {
    if (!request.srcs || request.n_srcs < 3 || !request.srcs[0] || !request.srcs[1] || !request.srcs[2]) return false;
    const ggml_tensor* x = request.srcs[0];
    return ops_is_float_activation_type(x->type) &&
           ops_is_float_activation_type(request.srcs[1]->type) &&
           ops_is_float_activation_type(request.srcs[2]->type) &&
           ggml_nelements(request.srcs[1]) >= x->ne[1] &&
           ggml_nelements(request.srcs[2]) >= x->ne[1];
}

} // namespace ggml_ops_ext
