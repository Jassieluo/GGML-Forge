#pragma once

#include "ops/types.h"

#include <cstdint>
#include <cstring>
#include <limits>

namespace ggml_ops_ext {

enum class ops_pool_mode : uint8_t {
    maximum = 0,
    average = 1,
};

// Pool1D: [W,C,N], Pool2D: [W,H,C,N], Pool3D: [W*H*D,C,N].
struct ops_pool_nd_config {
    int32_t spatial_dims = 1;
    int32_t input_size[3] = {0, 1, 1};
    int32_t kernel_size[3] = {1, 1, 1};
    int32_t stride[3] = {1, 1, 1};
    int32_t padding_before[3] = {0, 0, 0};
    int32_t padding_after[3] = {0, 0, 0};
    int32_t dilation[3] = {1, 1, 1};
    ops_pool_mode mode = ops_pool_mode::maximum;
    bool ceil_mode = false;
    bool count_include_pad = true;
};

struct ops_pool_nd_encoded_params {
    uint8_t spatial_dims = 1;
    uint8_t mode = static_cast<uint8_t>(ops_pool_mode::maximum);
    uint8_t ceil_mode = 0;
    uint8_t count_include_pad = 1;
    int32_t input_size[3] = {0, 1, 1};
    int16_t kernel_size[3] = {1, 1, 1};
    int16_t stride[3] = {1, 1, 1};
    int16_t padding_before[3] = {0, 0, 0};
    int16_t padding_after[3] = {0, 0, 0};
    int16_t dilation[3] = {1, 1, 1};
};
static_assert(sizeof(ops_pool_nd_encoded_params) <= 64);

struct ops_pool_nd_desc {
    int spatial_dims = 0;
    ops_pool_mode mode = ops_pool_mode::maximum;
    bool ceil_mode = false;
    bool count_include_pad = true;
    int64_t input_size[3] = {};
    int64_t output_size[3] = {};
    int64_t kernel_size[3] = {};
    int64_t channels = 0;
    int64_t batch = 0;
};

inline bool ops_pool_small_parameter(int32_t value) {
    return value >= std::numeric_limits<int16_t>::min() &&
           value <= std::numeric_limits<int16_t>::max();
}

inline bool ops_encode_pool_nd_params(const ops_pool_nd_config& config,
                                      ops_pool_nd_encoded_params& encoded) {
    if (config.spatial_dims < 1 || config.spatial_dims > 3 ||
        (config.mode != ops_pool_mode::maximum && config.mode != ops_pool_mode::average)) {
        return false;
    }
    encoded = {};
    encoded.spatial_dims = static_cast<uint8_t>(config.spatial_dims);
    encoded.mode = static_cast<uint8_t>(config.mode);
    encoded.ceil_mode = config.ceil_mode ? 1 : 0;
    encoded.count_include_pad = config.count_include_pad ? 1 : 0;
    for (int axis = 0; axis < 3; ++axis) {
        const bool active = axis < config.spatial_dims;
        const int32_t input = active ? config.input_size[axis] : 1;
        const int32_t kernel = active ? config.kernel_size[axis] : 1;
        const int32_t stride = active ? config.stride[axis] : 1;
        const int32_t before = active ? config.padding_before[axis] : 0;
        const int32_t after = active ? config.padding_after[axis] : 0;
        const int32_t dilation = active ? config.dilation[axis] : 1;
        if (input <= 0 || kernel <= 0 || stride <= 0 || before < 0 || after < 0 || dilation <= 0 ||
            !ops_pool_small_parameter(kernel) || !ops_pool_small_parameter(stride) ||
            !ops_pool_small_parameter(before) || !ops_pool_small_parameter(after) ||
            !ops_pool_small_parameter(dilation)) {
            return false;
        }
        encoded.input_size[axis] = input;
        encoded.kernel_size[axis] = static_cast<int16_t>(kernel);
        encoded.stride[axis] = static_cast<int16_t>(stride);
        encoded.padding_before[axis] = static_cast<int16_t>(before);
        encoded.padding_after[axis] = static_cast<int16_t>(after);
        encoded.dilation[axis] = static_cast<int16_t>(dilation);
    }
    return true;
}

inline ops_status ops_validate_pool_nd_contract(const ops_request& request,
                                                int expected_spatial_dims,
                                                ops_pool_nd_desc* output_desc = nullptr) {
    if (!request.srcs || request.n_srcs < 1 || !request.srcs[0] || !request.params ||
        request.params_size < sizeof(ops_pool_nd_encoded_params)) {
        return ops_status::error(ops_status_code::invalid_request, "PoolND request is incomplete");
    }
    ops_pool_nd_encoded_params params;
    std::memcpy(&params, request.params, sizeof(params));
    if (params.spatial_dims != expected_spatial_dims ||
        params.mode > static_cast<uint8_t>(ops_pool_mode::average)) {
        return ops_status::error(ops_status_code::invalid_request,
                                 "PoolND rank or mode is invalid");
    }
    const ggml_tensor* input = request.srcs[0];
    ops_pool_nd_desc desc;
    desc.spatial_dims = expected_spatial_dims;
    desc.mode = static_cast<ops_pool_mode>(params.mode);
    desc.ceil_mode = params.ceil_mode != 0;
    desc.count_include_pad = params.count_include_pad != 0;
    int64_t input_volume = 1;
    int64_t output_volume = 1;
    for (int axis = 0; axis < 3; ++axis) {
        desc.input_size[axis] = params.input_size[axis];
        desc.kernel_size[axis] = params.kernel_size[axis];
        if (params.input_size[axis] <= 0 || params.kernel_size[axis] <= 0 ||
            params.stride[axis] <= 0 || params.padding_before[axis] < 0 ||
            params.padding_after[axis] < 0 || params.dilation[axis] <= 0) {
            return ops_status::error(ops_status_code::invalid_request,
                                     "PoolND geometry is invalid");
        }
        const int64_t effective =
            params.dilation[axis] * (int64_t(params.kernel_size[axis]) - 1) + 1;
        const int64_t numerator = int64_t(params.input_size[axis]) + params.padding_before[axis] +
                                  params.padding_after[axis] - effective;
        if (numerator < 0) {
            return ops_status::error(ops_status_code::invalid_request,
                                     "PoolND kernel exceeds padded input");
        }
        int64_t size =
            (numerator + (desc.ceil_mode ? params.stride[axis] - 1 : 0)) / params.stride[axis] + 1;
        if (desc.ceil_mode && size > 1 &&
            (size - 1) * params.stride[axis] >=
                int64_t(params.input_size[axis]) + params.padding_before[axis]) {
            --size;
        }
        if (size <= 0) {
            return ops_status::error(ops_status_code::invalid_request, "PoolND output is empty");
        }
        desc.output_size[axis] = size;
        input_volume *= desc.input_size[axis];
        output_volume *= size;
    }
    if (expected_spatial_dims == 1) {
        if (input->ne[0] != desc.input_size[0]) {
            return ops_status::error(ops_status_code::invalid_request,
                                     "Pool1D input shape mismatch");
        }
        desc.channels = input->ne[1];
        desc.batch = input->ne[2] * input->ne[3];
    } else if (expected_spatial_dims == 2) {
        if (input->ne[0] != desc.input_size[0] || input->ne[1] != desc.input_size[1]) {
            return ops_status::error(ops_status_code::invalid_request,
                                     "Pool2D input shape mismatch");
        }
        desc.channels = input->ne[2];
        desc.batch = input->ne[3];
    } else {
        if (input->ne[0] != input_volume) {
            return ops_status::error(ops_status_code::invalid_request,
                                     "Pool3D packed volume mismatch");
        }
        desc.channels = input->ne[1];
        desc.batch = input->ne[2] * input->ne[3];
    }
    if (desc.channels <= 0 || desc.batch <= 0) {
        return ops_status::error(ops_status_code::invalid_request,
                                 "PoolND channels or batch is invalid");
    }
    if (expected_spatial_dims != 2 &&
        (!ops_batch_fold_packed(input) ||
         (request.output && !ops_batch_fold_packed(request.output)))) {
        return ops_status::error(ops_status_code::unsupported,
                                 "PoolND batch dims must be packed");
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
                                     "PoolND output shape or type mismatch");
        }
    }
    if (output_desc) {
        *output_desc = desc;
    }
    return ops_status::ok();
}

} // namespace ggml_ops_ext
