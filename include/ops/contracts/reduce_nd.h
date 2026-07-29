#pragma once

#include "ops/types.h"

#include <cstdint>
#include <cstring>
#include <limits>

namespace ggml_ops_ext {

enum class ops_reduce_mode : uint8_t {
    sum = 0,
    mean = 1,
    maximum = 2,
    minimum = 3,
    product = 4,
    variance = 5,
    standard_deviation = 6,
    logsumexp = 7,
};

// Axes use GGML's physical order (ne[0] .. ne[3]).  When keep_dims is false,
// retained axes are compacted towards ne[0] while preserving their order.
struct ops_reduce_nd_config {
    uint8_t axis_mask = 0x01;
    ops_reduce_mode mode = ops_reduce_mode::sum;
    bool keep_dims = true;
    // Matches torch.var/std correction. Ignored by other modes.
    int32_t correction = 1;
};

struct ops_reduce_nd_encoded_params {
    uint8_t axis_mask = 0x01;
    uint8_t mode = static_cast<uint8_t>(ops_reduce_mode::sum);
    uint8_t keep_dims = 1;
    uint8_t reserved = 0;
    int32_t correction = 1;
};
static_assert(sizeof(ops_reduce_nd_encoded_params) <= 64);

struct ops_reduce_nd_desc {
    uint8_t axis_mask = 0;
    ops_reduce_mode mode = ops_reduce_mode::sum;
    bool keep_dims = true;
    int32_t correction = 1;
    int64_t input_shape[4] = {1, 1, 1, 1};
    int64_t output_shape[4] = {1, 1, 1, 1};
    int64_t reduction_count = 1;
    int64_t output_count = 1;
};

enum class ops_arg_reduce_mode : uint8_t { maximum = 0, minimum = 1 };

struct ops_arg_reduce_nd_config {
    uint8_t axis_mask = 0x01;
    ops_arg_reduce_mode mode = ops_arg_reduce_mode::maximum;
    bool keep_dims = true;
};

struct ops_arg_reduce_nd_encoded_params {
    uint8_t axis_mask = 0x01;
    uint8_t mode = static_cast<uint8_t>(ops_arg_reduce_mode::maximum);
    uint8_t keep_dims = 1;
    uint8_t reserved = 0;
};
static_assert(sizeof(ops_arg_reduce_nd_encoded_params) <= 64);

struct ops_arg_reduce_nd_desc {
    uint8_t axis_mask = 0;
    ops_arg_reduce_mode mode = ops_arg_reduce_mode::maximum;
    bool keep_dims = true;
    int64_t input_shape[4] = {1, 1, 1, 1};
    int64_t output_shape[4] = {1, 1, 1, 1};
    int64_t reduction_count = 1;
    int64_t output_count = 1;
};

inline bool ops_encode_reduce_nd_params(const ops_reduce_nd_config& config,
                                        ops_reduce_nd_encoded_params& encoded) {
    if ((config.axis_mask & 0x0f) == 0 || (config.axis_mask & 0xf0) != 0 ||
        config.mode > ops_reduce_mode::logsumexp || config.correction < 0) {
        return false;
    }
    encoded = {};
    encoded.axis_mask = config.axis_mask;
    encoded.mode = static_cast<uint8_t>(config.mode);
    encoded.keep_dims = config.keep_dims ? 1 : 0;
    encoded.correction = config.correction;
    return true;
}

inline ops_status ops_validate_reduce_nd_contract(const ops_request& request,
                                                  ops_reduce_nd_desc* output_desc = nullptr) {
    if (!request.srcs || request.n_srcs < 1 || !request.srcs[0] || !request.params ||
        request.params_size < sizeof(ops_reduce_nd_encoded_params)) {
        return ops_status::error(ops_status_code::invalid_request,
                                 "ReduceND request is incomplete");
    }
    ops_reduce_nd_encoded_params params{};
    std::memcpy(&params, request.params, sizeof(params));
    if ((params.axis_mask & 0x0f) == 0 || (params.axis_mask & 0xf0) != 0 ||
        params.mode > static_cast<uint8_t>(ops_reduce_mode::logsumexp) ||
        params.keep_dims > 1 || params.correction < 0) {
        return ops_status::error(ops_status_code::invalid_request,
                                 "ReduceND parameters are invalid");
    }

    const ggml_tensor* input = request.srcs[0];
    ops_reduce_nd_desc desc;
    desc.axis_mask = params.axis_mask;
    desc.mode = static_cast<ops_reduce_mode>(params.mode);
    desc.keep_dims = params.keep_dims != 0;
    desc.correction = params.correction;
    int compact_axis = 0;
    for (int axis = 0; axis < 4; ++axis) {
        if (input->ne[axis] <= 0 ||
            desc.reduction_count > std::numeric_limits<int64_t>::max() / input->ne[axis]) {
            return ops_status::error(ops_status_code::invalid_request,
                                     "ReduceND input shape is invalid or too large");
        }
        desc.input_shape[axis] = input->ne[axis];
        if (desc.axis_mask & (1u << axis)) {
            desc.reduction_count *= input->ne[axis];
            if (desc.keep_dims) desc.output_shape[axis] = 1;
        } else {
            const int output_axis = desc.keep_dims ? axis : compact_axis++;
            desc.output_shape[output_axis] = input->ne[axis];
        }
    }
    desc.output_count = 1;
    for (int axis = 0; axis < 4; ++axis) {
        if (desc.output_shape[axis] <= 0 ||
            desc.output_count > std::numeric_limits<int64_t>::max() / desc.output_shape[axis]) {
            return ops_status::error(ops_status_code::invalid_request,
                                     "ReduceND output shape is invalid or too large");
        }
        desc.output_count *= desc.output_shape[axis];
    }
    if ((desc.mode == ops_reduce_mode::variance ||
         desc.mode == ops_reduce_mode::standard_deviation) &&
        desc.reduction_count <= desc.correction) {
        return ops_status::error(ops_status_code::invalid_request,
                                 "ReduceND correction must be smaller than the reduction size");
    }
    if (request.output) {
        if (request.output->type != input->type) {
            return ops_status::error(ops_status_code::invalid_request,
                                     "ReduceND output type mismatch");
        }
        for (int axis = 0; axis < 4; ++axis) {
            if (request.output->ne[axis] != desc.output_shape[axis]) {
                return ops_status::error(ops_status_code::invalid_request,
                                         "ReduceND output shape mismatch");
            }
        }
    }
    if (output_desc) *output_desc = desc;
    return ops_status::ok();
}

inline bool ops_encode_arg_reduce_nd_params(const ops_arg_reduce_nd_config& config,
                                            ops_arg_reduce_nd_encoded_params& encoded) {
    if ((config.axis_mask & 0x0f) == 0 || (config.axis_mask & 0xf0) != 0 ||
        config.mode > ops_arg_reduce_mode::minimum) return false;
    encoded = {};
    encoded.axis_mask = config.axis_mask;
    encoded.mode = static_cast<uint8_t>(config.mode);
    encoded.keep_dims = config.keep_dims ? 1 : 0;
    return true;
}

inline ops_status ops_validate_arg_reduce_nd_contract(
    const ops_request& request, ops_arg_reduce_nd_desc* output_desc = nullptr) {
    if (!request.srcs || request.n_srcs < 1 || !request.srcs[0] || !request.params ||
        request.params_size < sizeof(ops_arg_reduce_nd_encoded_params)) {
        return ops_status::error(ops_status_code::invalid_request,
                                 "ArgReduceND request is incomplete");
    }
    ops_arg_reduce_nd_encoded_params params{};
    std::memcpy(&params, request.params, sizeof(params));
    if ((params.axis_mask & 0x0f) == 0 || (params.axis_mask & 0xf0) != 0 ||
        params.mode > static_cast<uint8_t>(ops_arg_reduce_mode::minimum) || params.keep_dims > 1) {
        return ops_status::error(ops_status_code::invalid_request,
                                 "ArgReduceND parameters are invalid");
    }
    const ggml_tensor* input = request.srcs[0];
    ops_arg_reduce_nd_desc desc;
    desc.axis_mask = params.axis_mask;
    desc.mode = static_cast<ops_arg_reduce_mode>(params.mode);
    desc.keep_dims = params.keep_dims != 0;
    int compact_axis = 0;
    for (int axis = 0; axis < 4; ++axis) {
        if (input->ne[axis] <= 0) {
            return ops_status::error(ops_status_code::invalid_request,
                                     "ArgReduceND input shape is invalid");
        }
        desc.input_shape[axis] = input->ne[axis];
        if (desc.axis_mask & (1u << axis)) {
            if (desc.reduction_count > std::numeric_limits<int64_t>::max() / input->ne[axis]) {
                return ops_status::error(ops_status_code::invalid_request,
                                         "ArgReduceND reduction is too large");
            }
            desc.reduction_count *= input->ne[axis];
        } else {
            desc.output_shape[desc.keep_dims ? axis : compact_axis++] = input->ne[axis];
        }
    }
    for (int axis = 0; axis < 4; ++axis) desc.output_count *= desc.output_shape[axis];
    if (request.output) {
        if (request.output->type != GGML_TYPE_I32) {
            return ops_status::error(ops_status_code::invalid_request,
                                     "ArgReduceND output must be I32");
        }
        for (int axis = 0; axis < 4; ++axis) {
            if (request.output->ne[axis] != desc.output_shape[axis]) {
                return ops_status::error(ops_status_code::invalid_request,
                                         "ArgReduceND output shape mismatch");
            }
        }
    }
    if (output_desc) *output_desc = desc;
    return ops_status::ok();
}

} // namespace ggml_ops_ext
