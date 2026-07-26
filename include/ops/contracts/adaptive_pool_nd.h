#pragma once

#include "ops/contracts/pool_nd.h"

#include <cstdint>
#include <cstring>

namespace ggml_ops_ext {

struct ops_adaptive_pool_nd_config {
    int32_t spatial_dims = 1;
    int32_t input_size[3] = {0, 1, 1};
    int32_t output_size[3] = {1, 1, 1};
    ops_pool_mode mode = ops_pool_mode::average;
};

struct ops_adaptive_pool_nd_encoded_params {
    uint8_t spatial_dims = 1;
    uint8_t mode = static_cast<uint8_t>(ops_pool_mode::average);
    uint16_t reserved = 0;
    int32_t input_size[3] = {0, 1, 1};
    int32_t output_size[3] = {1, 1, 1};
};
static_assert(sizeof(ops_adaptive_pool_nd_encoded_params) <= 64);

struct ops_adaptive_pool_nd_desc {
    int spatial_dims = 0;
    ops_pool_mode mode = ops_pool_mode::average;
    int64_t input_size[3] = {};
    int64_t output_size[3] = {};
    int64_t channels = 0;
    int64_t batch = 0;
};

inline bool ops_encode_adaptive_pool_nd_params(const ops_adaptive_pool_nd_config& config,
                                               ops_adaptive_pool_nd_encoded_params& encoded) {
    if (config.spatial_dims < 1 || config.spatial_dims > 3 ||
        (config.mode != ops_pool_mode::maximum && config.mode != ops_pool_mode::average)) {
        return false;
    }
    encoded = {};
    encoded.spatial_dims = static_cast<uint8_t>(config.spatial_dims);
    encoded.mode = static_cast<uint8_t>(config.mode);
    for (int axis = 0; axis < 3; ++axis) {
        const bool active = axis < config.spatial_dims;
        const int32_t input_size = active ? config.input_size[axis] : 1;
        const int32_t output_size = active ? config.output_size[axis] : 1;
        if (input_size <= 0 || output_size <= 0) {
            return false;
        }
        encoded.input_size[axis] = input_size;
        encoded.output_size[axis] = output_size;
    }
    return true;
}

inline ops_status
ops_validate_adaptive_pool_nd_contract(const ops_request& request, int expected_spatial_dims,
                                       ops_adaptive_pool_nd_desc* output_desc = nullptr) {
    if (!request.srcs || request.n_srcs < 1 || !request.srcs[0] || !request.params ||
        request.params_size < sizeof(ops_adaptive_pool_nd_encoded_params)) {
        return ops_status::error(ops_status_code::invalid_request,
                                 "AdaptivePoolND request is incomplete");
    }
    ops_adaptive_pool_nd_encoded_params params;
    std::memcpy(&params, request.params, sizeof(params));
    if (params.spatial_dims != expected_spatial_dims ||
        params.mode > static_cast<uint8_t>(ops_pool_mode::average)) {
        return ops_status::error(ops_status_code::invalid_request,
                                 "AdaptivePoolND rank or mode is invalid");
    }
    const ggml_tensor* input = request.srcs[0];
    ops_adaptive_pool_nd_desc desc;
    desc.spatial_dims = expected_spatial_dims;
    desc.mode = static_cast<ops_pool_mode>(params.mode);
    int64_t input_volume = 1;
    int64_t output_volume = 1;
    for (int axis = 0; axis < 3; ++axis) {
        if (params.input_size[axis] <= 0 || params.output_size[axis] <= 0) {
            return ops_status::error(ops_status_code::invalid_request,
                                     "AdaptivePoolND geometry is invalid");
        }
        desc.input_size[axis] = params.input_size[axis];
        desc.output_size[axis] = params.output_size[axis];
        input_volume *= desc.input_size[axis];
        output_volume *= desc.output_size[axis];
    }
    if (expected_spatial_dims == 1) {
        if (input->ne[0] != desc.input_size[0]) {
            return ops_status::error(ops_status_code::invalid_request,
                                     "AdaptivePool1D input shape mismatch");
        }
        desc.channels = input->ne[1];
        desc.batch = input->ne[2] * input->ne[3];
    } else if (expected_spatial_dims == 2) {
        if (input->ne[0] != desc.input_size[0] || input->ne[1] != desc.input_size[1]) {
            return ops_status::error(ops_status_code::invalid_request,
                                     "AdaptivePool2D input shape mismatch");
        }
        desc.channels = input->ne[2];
        desc.batch = input->ne[3];
    } else {
        if (input->ne[0] != input_volume) {
            return ops_status::error(ops_status_code::invalid_request,
                                     "AdaptivePool3D packed volume mismatch");
        }
        desc.channels = input->ne[1];
        desc.batch = input->ne[2] * input->ne[3];
    }
    if (desc.channels <= 0 || desc.batch <= 0) {
        return ops_status::error(ops_status_code::invalid_request,
                                 "AdaptivePoolND channels or batch is invalid");
    }
    if (expected_spatial_dims != 2 &&
        (!ops_batch_fold_packed(input) ||
         (request.output && !ops_batch_fold_packed(request.output)))) {
        return ops_status::error(ops_status_code::unsupported,
                                 "AdaptivePoolND batch dims must be packed");
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
                                     "AdaptivePoolND output shape or type mismatch");
        }
    }
    if (output_desc) {
        *output_desc = desc;
    }
    return ops_status::ok();
}

} // namespace ggml_ops_ext
