#pragma once

#include "ops/types.h"

#include <cstdint>
#include <cstring>

namespace ggml_ops_ext {

enum class ops_compare_mode : uint8_t { equal, not_equal, less, less_equal, greater, greater_equal };
enum class ops_logical_mode : uint8_t { logical_and, logical_or, logical_xor, logical_not };

struct ops_selection_params {
    uint8_t mode = 0;
    uint8_t reserved[3] = {};
};
static_assert(sizeof(ops_selection_params) <= 64);

inline bool ops_same_shape(const ggml_tensor* a, const ggml_tensor* b) {
    if (!a || !b) return false;
    for (int axis = 0; axis < 4; ++axis) if (a->ne[axis] != b->ne[axis]) return false;
    return true;
}

inline ops_status ops_validate_compare(const ops_request& request) {
    if (!request.srcs || request.n_srcs < 2 || !request.srcs[0] || !request.srcs[1] ||
        !request.params || request.params_size < sizeof(ops_selection_params)) {
        return ops_status::error(ops_status_code::invalid_request, "Compare request is incomplete");
    }
    ops_selection_params params{};
    std::memcpy(&params, request.params, sizeof(params));
    if (params.mode > static_cast<uint8_t>(ops_compare_mode::greater_equal) ||
        request.srcs[0]->type != request.srcs[1]->type ||
        !ops_same_shape(request.srcs[0], request.srcs[1])) {
        return ops_status::error(ops_status_code::invalid_request, "Compare inputs or mode are invalid");
    }
    if (request.output &&
        (request.output->type != GGML_TYPE_I32 || !ops_same_shape(request.output, request.srcs[0]))) {
        return ops_status::error(ops_status_code::invalid_request, "Compare output must be same-shape I32");
    }
    return ops_status::ok();
}

inline ops_status ops_validate_logical(const ops_request& request) {
    if (!request.srcs || request.n_srcs < 1 || !request.srcs[0] || !request.params ||
        request.params_size < sizeof(ops_selection_params)) {
        return ops_status::error(ops_status_code::invalid_request, "Logical request is incomplete");
    }
    ops_selection_params params{};
    std::memcpy(&params, request.params, sizeof(params));
    if (params.mode > static_cast<uint8_t>(ops_logical_mode::logical_not) ||
        request.srcs[0]->type != GGML_TYPE_I32) {
        return ops_status::error(ops_status_code::invalid_request, "Logical mode or input type is invalid");
    }
    const bool unary = params.mode == static_cast<uint8_t>(ops_logical_mode::logical_not);
    if (!unary && (request.n_srcs < 2 || !request.srcs[1] ||
                   request.srcs[1]->type != GGML_TYPE_I32 ||
                   !ops_same_shape(request.srcs[0], request.srcs[1]))) {
        return ops_status::error(ops_status_code::invalid_request, "Logical binary input is invalid");
    }
    if (request.output &&
        (request.output->type != GGML_TYPE_I32 || !ops_same_shape(request.output, request.srcs[0]))) {
        return ops_status::error(ops_status_code::invalid_request, "Logical output must be same-shape I32");
    }
    return ops_status::ok();
}

inline ops_status ops_validate_where(const ops_request& request) {
    if (!request.srcs || request.n_srcs < 3 || !request.srcs[0] || !request.srcs[1] ||
        !request.srcs[2] || request.srcs[0]->type != GGML_TYPE_I32 ||
        request.srcs[1]->type != request.srcs[2]->type ||
        !ops_same_shape(request.srcs[0], request.srcs[1]) ||
        !ops_same_shape(request.srcs[1], request.srcs[2])) {
        return ops_status::error(ops_status_code::invalid_request, "Where inputs are invalid");
    }
    if (request.output &&
        (request.output->type != request.srcs[1]->type || !ops_same_shape(request.output, request.srcs[1]))) {
        return ops_status::error(ops_status_code::invalid_request, "Where output shape or type mismatch");
    }
    return ops_status::ok();
}

} // namespace ggml_ops_ext
