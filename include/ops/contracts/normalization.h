#pragma once

#include "ops/contracts/activation.h"
#include <cmath>
#include <cstring>

namespace ggml_ops_ext {

inline bool ops_validate_affine_norm(const ops_request& request) {
    if (!request.srcs || request.n_srcs < 3 || !request.srcs[0] || !request.srcs[1] || !request.srcs[2] ||
        !request.params || request.params_size < sizeof(float)) return false;
    const ggml_tensor* x = request.srcs[0];
    if (!ops_is_float_activation_type(x->type) || request.srcs[1]->type != x->type ||
        request.srcs[2]->type != x->type) return false;
    if (!ggml_is_contiguous(x) || !ggml_is_contiguous(request.srcs[1]) ||
        !ggml_is_contiguous(request.srcs[2])) return false;
    float eps;
    std::memcpy(&eps, request.params, sizeof(eps));
    return std::isfinite(eps) && eps > 0.0f && request.srcs[1]->ne[0] == x->ne[0] &&
           request.srcs[2]->ne[0] == x->ne[0] &&
           ggml_nelements(request.srcs[1]) == ggml_nelements(request.srcs[2]) &&
           (!request.output || request.output->type == x->type);
}

inline bool ops_validate_instance_norm(const ops_request& request) {
    if (!request.srcs || request.n_srcs < 3 || !request.srcs[0] ||
        !request.params || request.params_size < sizeof(float)) return false;
    const ggml_tensor* x = request.srcs[0];
    if (x->ne[0] <= 0 || x->ne[1] <= 0 || !ggml_is_contiguous(x) ||
        !ops_is_float_activation_type(x->type)) return false;
    float eps;
    std::memcpy(&eps, request.params, sizeof(eps));
    if (!std::isfinite(eps) || eps <= 0.0f) return false;
    for (int i = 1; i <= 2; ++i) {
        if (request.srcs[i] && (!ops_is_float_activation_type(request.srcs[i]->type) ||
                                ggml_nelements(request.srcs[i]) < x->ne[1])) return false;
    }
    return !request.output || request.output->type == x->type;
}

} // namespace ggml_ops_ext
