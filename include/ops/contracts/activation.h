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

struct ops_gated_activation_params {
    int32_t activation = 0;
    int32_t axis = 0;
};

inline bool ops_validate_gated_activation(const ops_request& request) {
    if (!ops_validate_unary_activation(request) || !request.params ||
        request.params_size < sizeof(ops_gated_activation_params)) return false;
    ops_gated_activation_params params;
    std::memcpy(&params, request.params, sizeof(params));
    const ggml_tensor* input = request.srcs[0];
    if (params.activation < 0 || params.activation > 3 || params.axis < 0 || params.axis >= GGML_MAX_DIMS ||
        input->ne[params.axis] <= 0 || input->ne[params.axis] % 2 != 0 || !ggml_is_contiguous(input)) return false;
    if (!request.output) return true;
    if (request.output->type != input->type) return false;
    for (int axis = 0; axis < GGML_MAX_DIMS; ++axis) {
        const int64_t expected = axis == params.axis ? input->ne[axis] / 2 : input->ne[axis];
        if (request.output->ne[axis] != expected) return false;
    }
    return true;
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

inline bool ops_validate_alias_free_activation(const ops_request& request) {
    if (!request.srcs || request.n_srcs < 5) return false;
    const ggml_tensor* x = request.srcs[0];
    const ggml_tensor* up = request.srcs[1];
    const ggml_tensor* down = request.srcs[2];
    const ggml_tensor* alpha = request.srcs[3];
    const ggml_tensor* beta = request.srcs[4];
    if (!x || !up || !down || !alpha || !beta || !ops_is_float_activation_type(x->type)) return false;
    if ((up->type != GGML_TYPE_F16 && up->type != GGML_TYPE_F32) ||
        (down->type != GGML_TYPE_F16 && down->type != GGML_TYPE_F32) ||
        !ops_is_float_activation_type(alpha->type) || !ops_is_float_activation_type(beta->type)) return false;

    const int64_t channels = x->ne[1];
    return x->ne[0] > 0 && channels > 0 && up->ne[0] == 12 && down->ne[0] == 12 &&
           up->ne[1] == 1 && down->ne[1] == 1 && up->ne[2] == channels && down->ne[2] == channels &&
           ggml_nelements(alpha) >= channels && ggml_nelements(beta) >= channels &&
           (!request.output || (request.output->type == x->type &&
                                ggml_are_same_shape(request.output, x)));
}

} // namespace ggml_ops_ext
