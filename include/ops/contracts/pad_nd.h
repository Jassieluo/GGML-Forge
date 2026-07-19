#pragma once

#include "ops/types.h"

#include <cstdint>
#include <cstring>
#include <limits>

namespace ggml_ops_ext {

enum class ops_pad_mode : uint8_t {
    constant = 0,
    reflect = 1,
    replicate = 2,
    circular = 3,
};

// Pad1D: [W,C,N], Pad2D: [W,H,C,N], Pad3D: [W*H*D,C,N].
struct ops_pad_nd_config {
    int32_t spatial_dims = 1;
    int32_t input_size[3] = {0, 1, 1};
    int32_t padding_before[3] = {0, 0, 0};
    int32_t padding_after[3] = {0, 0, 0};
    ops_pad_mode mode = ops_pad_mode::constant;
    float value = 0.0f;
};

struct ops_pad_nd_encoded_params {
    uint8_t spatial_dims = 1;
    uint8_t mode = static_cast<uint8_t>(ops_pad_mode::constant);
    uint16_t reserved = 0;
    int32_t input_size[3] = {0, 1, 1};
    int32_t padding_before[3] = {0, 0, 0};
    int32_t padding_after[3] = {0, 0, 0};
    float value = 0.0f;
};
static_assert(sizeof(ops_pad_nd_encoded_params) <= 64);

struct ops_pad_nd_desc {
    int spatial_dims = 0;
    ops_pad_mode mode = ops_pad_mode::constant;
    float value = 0.0f;
    int64_t input_size[3] = {};
    int64_t output_size[3] = {};
    int64_t channels = 0;
    int64_t batch = 0;
};

inline bool ops_encode_pad_nd_params(const ops_pad_nd_config& config,
                                     ops_pad_nd_encoded_params& encoded) {
    if (config.spatial_dims < 1 || config.spatial_dims > 3 ||
        config.mode < ops_pad_mode::constant || config.mode > ops_pad_mode::circular) {
        return false;
    }
    encoded = {};
    encoded.spatial_dims = static_cast<uint8_t>(config.spatial_dims);
    encoded.mode = static_cast<uint8_t>(config.mode);
    encoded.value = config.value;
    for (int axis = 0; axis < 3; ++axis) {
        const bool active = axis < config.spatial_dims;
        const int32_t input_size = active ? config.input_size[axis] : 1;
        const int32_t padding_before = active ? config.padding_before[axis] : 0;
        const int32_t padding_after = active ? config.padding_after[axis] : 0;
        if (input_size <= 0 || padding_before < 0 || padding_after < 0 ||
            int64_t(input_size) + padding_before + padding_after >
                std::numeric_limits<int32_t>::max()) {
            return false;
        }
        if (config.mode == ops_pad_mode::reflect && active &&
            (padding_before >= input_size || padding_after >= input_size)) {
            return false;
        }
        encoded.input_size[axis] = input_size;
        encoded.padding_before[axis] = padding_before;
        encoded.padding_after[axis] = padding_after;
    }
    return true;
}

inline ops_status ops_validate_pad_nd_contract(const ops_request& request,
                                               int expected_spatial_dims,
                                               ops_pad_nd_desc* output_desc = nullptr) {
    if (!request.srcs || request.n_srcs < 1 || !request.srcs[0] || !request.params ||
        request.params_size < sizeof(ops_pad_nd_encoded_params)) {
        return ops_status::error(ops_status_code::invalid_request, "PadND request is incomplete");
    }
    ops_pad_nd_encoded_params params;
    std::memcpy(&params, request.params, sizeof(params));
    if (params.spatial_dims != expected_spatial_dims ||
        params.mode > static_cast<uint8_t>(ops_pad_mode::circular)) {
        return ops_status::error(ops_status_code::invalid_request, "PadND rank or mode is invalid");
    }
    const ggml_tensor* input = request.srcs[0];
    ops_pad_nd_desc desc;
    desc.spatial_dims = expected_spatial_dims;
    desc.mode = static_cast<ops_pad_mode>(params.mode);
    desc.value = params.value;
    int64_t input_volume = 1;
    int64_t output_volume = 1;
    for (int axis = 0; axis < 3; ++axis) {
        if (params.input_size[axis] <= 0 || params.padding_before[axis] < 0 ||
            params.padding_after[axis] < 0) {
            return ops_status::error(ops_status_code::invalid_request, "PadND geometry is invalid");
        }
        if (desc.mode == ops_pad_mode::reflect && axis < expected_spatial_dims &&
            (params.padding_before[axis] >= params.input_size[axis] ||
             params.padding_after[axis] >= params.input_size[axis])) {
            return ops_status::error(ops_status_code::invalid_request,
                                     "Reflect padding must be smaller than input");
        }
        desc.input_size[axis] = params.input_size[axis];
        desc.output_size[axis] = int64_t(params.input_size[axis]) + params.padding_before[axis] +
                                 params.padding_after[axis];
        input_volume *= desc.input_size[axis];
        output_volume *= desc.output_size[axis];
    }
    if (expected_spatial_dims == 1) {
        if (input->ne[0] != desc.input_size[0]) {
            return ops_status::error(ops_status_code::invalid_request,
                                     "Pad1D input shape mismatch");
        }
        desc.channels = input->ne[1];
        desc.batch = input->ne[2] * input->ne[3];
    } else if (expected_spatial_dims == 2) {
        if (input->ne[0] != desc.input_size[0] || input->ne[1] != desc.input_size[1]) {
            return ops_status::error(ops_status_code::invalid_request,
                                     "Pad2D input shape mismatch");
        }
        desc.channels = input->ne[2];
        desc.batch = input->ne[3];
    } else {
        if (input->ne[0] != input_volume) {
            return ops_status::error(ops_status_code::invalid_request,
                                     "Pad3D packed volume mismatch");
        }
        desc.channels = input->ne[1];
        desc.batch = input->ne[2] * input->ne[3];
    }
    if (desc.channels <= 0 || desc.batch <= 0) {
        return ops_status::error(ops_status_code::invalid_request,
                                 "PadND channels or batch is invalid");
    }
    if (request.output) {
        const bool shape_matches =
            expected_spatial_dims == 1
                ? request.output->ne[0] == desc.output_size[0] &&
                      request.output->ne[1] == desc.channels && request.output->ne[2] == desc.batch
            : expected_spatial_dims == 2
                ? request.output->ne[0] == desc.output_size[0] &&
                      request.output->ne[1] == desc.output_size[1] &&
                      request.output->ne[2] == desc.channels && request.output->ne[3] == desc.batch
                : request.output->ne[0] == output_volume &&
                      request.output->ne[1] == desc.channels && request.output->ne[2] == desc.batch;
        if (!shape_matches || request.output->type != input->type) {
            return ops_status::error(ops_status_code::invalid_request,
                                     "PadND output shape or type mismatch");
        }
    }
    if (output_desc) {
        *output_desc = desc;
    }
    return ops_status::ok();
}

} // namespace ggml_ops_ext
