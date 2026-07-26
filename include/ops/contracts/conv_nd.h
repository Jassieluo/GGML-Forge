#pragma once

#include "ops/types.h"

#include <cstdint>
#include <cstring>
#include <limits>

namespace ggml_ops_ext {

// Public, unpacked ConvND description. Spatial dimensions are ordered W, H, D.
// Conv2D activations use [W, H, C, N]. Conv3D activations use the GGML-safe
// packed shape [W*H*D, C, N, 1], with input_size preserving logical geometry.
struct ops_conv_nd_config {
    int32_t spatial_dims = 1;
    int32_t input_size[3] = { 0, 1, 1 };
    int32_t kernel_size[3] = { 1, 1, 1 };
    int32_t stride[3] = { 1, 1, 1 };
    int32_t padding_before[3] = { 0, 0, 0 };
    int32_t padding_after[3] = { 0, 0, 0 };
    int32_t dilation[3] = { 1, 1, 1 };
    int32_t output_padding[3] = { 0, 0, 0 };
    int32_t groups = 1;
    ops_weight_layout weight_layout = ops_weight_layout::channel_rows;
};

// GGML limits op_params to 64 bytes. Geometry remains 32-bit while the small
// convolution hyperparameters use a checked 16-bit representation.
struct ops_conv_nd_encoded_params {
    uint8_t spatial_dims = 1;
    uint8_t transposed = 0;
    uint8_t weight_layout = static_cast<uint8_t>(ops_weight_layout::channel_rows);
    uint8_t reserved = 0;
    int32_t groups = 1;
    int32_t input_size[3] = { 0, 1, 1 };
    int32_t kernel_size[3] = { 1, 1, 1 };
    int16_t stride[3] = { 1, 1, 1 };
    int16_t padding_before[3] = { 0, 0, 0 };
    int16_t padding_after[3] = { 0, 0, 0 };
    int16_t dilation[3] = { 1, 1, 1 };
    int16_t output_padding[3] = { 0, 0, 0 };
};
static_assert(sizeof(ops_conv_nd_encoded_params) <= 64);

struct ops_conv_nd_desc {
    int spatial_dims = 0;
    bool transposed = false;
    int64_t input_size[3] = {};
    int64_t kernel_size[3] = {};
    int64_t output_size[3] = {};
    int64_t kernel_volume = 0;
    int64_t input_channels = 0;
    int64_t input_channels_per_group = 0;
    int64_t output_channels = 0;
    int64_t output_channels_per_group = 0;
    int64_t batch = 0;
    int64_t groups = 0;
    ops_weight_layout weight_layout = ops_weight_layout::channel_rows;
};

inline bool ops_conv_nd_small_parameter(int32_t value) {
    return value >= std::numeric_limits<int16_t>::min() &&
           value <= std::numeric_limits<int16_t>::max();
}

inline bool ops_encode_conv_nd_params(
    const ops_conv_nd_config& config,
    bool transposed,
    ops_conv_nd_encoded_params& encoded
) {
    if (config.spatial_dims < 1 || config.spatial_dims > 3 || config.groups <= 0) return false;
    if (config.weight_layout != ops_weight_layout::channel_rows &&
        config.weight_layout != ops_weight_layout::flattened_rows) return false;
    encoded = {};
    encoded.spatial_dims = static_cast<uint8_t>(config.spatial_dims);
    encoded.transposed = transposed ? 1 : 0;
    encoded.weight_layout = static_cast<uint8_t>(config.weight_layout);
    encoded.groups = config.groups;
    for (int axis = 0; axis < 3; ++axis) {
        const bool active = axis < config.spatial_dims;
        const int32_t input = active ? config.input_size[axis] : 1;
        const int32_t kernel = active ? config.kernel_size[axis] : 1;
        const int32_t stride = active ? config.stride[axis] : 1;
        const int32_t before = active ? config.padding_before[axis] : 0;
        const int32_t after = active ? config.padding_after[axis] : 0;
        const int32_t dilation = active ? config.dilation[axis] : 1;
        const int32_t output_padding = active ? config.output_padding[axis] : 0;
        if (input <= 0 || kernel <= 0 || stride <= 0 || before < 0 || after < 0 ||
            dilation <= 0 || output_padding < 0 || output_padding >= stride ||
            !ops_conv_nd_small_parameter(stride) || !ops_conv_nd_small_parameter(before) ||
            !ops_conv_nd_small_parameter(after) || !ops_conv_nd_small_parameter(dilation) ||
            !ops_conv_nd_small_parameter(output_padding)) return false;
        encoded.input_size[axis] = input;
        encoded.kernel_size[axis] = kernel;
        encoded.stride[axis] = static_cast<int16_t>(stride);
        encoded.padding_before[axis] = static_cast<int16_t>(before);
        encoded.padding_after[axis] = static_cast<int16_t>(after);
        encoded.dilation[axis] = static_cast<int16_t>(dilation);
        encoded.output_padding[axis] = static_cast<int16_t>(output_padding);
    }
    return true;
}

inline ops_status ops_validate_conv_nd_contract(
    const ops_request& request,
    int expected_spatial_dims,
    bool expected_transposed,
    ops_conv_nd_desc* output_desc = nullptr
) {
    if (!request.srcs || request.n_srcs < 2 || !request.srcs[0] || !request.srcs[1] ||
        !request.params || request.params_size < sizeof(ops_conv_nd_encoded_params)) {
        return ops_status::error(ops_status_code::invalid_request, "ConvND request is missing tensors or parameters");
    }
    ops_conv_nd_encoded_params params;
    std::memcpy(&params, request.params, sizeof(params));
    if (params.spatial_dims != expected_spatial_dims || (params.transposed != 0) != expected_transposed ||
        params.groups <= 0) {
        return ops_status::error(ops_status_code::invalid_request, "ConvND rank, direction, or groups is invalid");
    }

    const ggml_tensor* weight = request.srcs[0];
    const ggml_tensor* input = request.srcs[1];
    const ggml_tensor* bias = request.n_srcs >= 3 ? request.srcs[2] : nullptr;
    ops_conv_nd_desc desc;
    desc.spatial_dims = expected_spatial_dims;
    desc.transposed = expected_transposed;
    desc.groups = params.groups;
    desc.weight_layout = static_cast<ops_weight_layout>(params.weight_layout);
    if (desc.weight_layout != ops_weight_layout::channel_rows &&
        desc.weight_layout != ops_weight_layout::flattened_rows) {
        return ops_status::error(ops_status_code::invalid_request, "ConvND weight layout is invalid");
    }
    // Quantized decode paths compute flat packed block indices and ignore the
    // weight's byte strides — a non-contiguous quantized view would be read
    // wrongly, so require contiguity.
    if (ggml_is_quantized(weight->type) && !ggml_is_contiguous(weight)) {
        return ops_status::error(ops_status_code::unsupported, "ConvND quantized weight must be contiguous");
    }
    desc.kernel_volume = 1;
    int64_t input_volume = 1;
    int64_t output_volume = 1;
    for (int axis = 0; axis < 3; ++axis) {
        desc.input_size[axis] = params.input_size[axis];
        desc.kernel_size[axis] = params.kernel_size[axis];
        if (axis < expected_spatial_dims &&
            (params.input_size[axis] <= 0 || params.kernel_size[axis] <= 0 || params.stride[axis] <= 0 ||
             params.padding_before[axis] < 0 || params.padding_after[axis] < 0 || params.dilation[axis] <= 0 ||
             params.output_padding[axis] < 0 || params.output_padding[axis] >= params.stride[axis])) {
            return ops_status::error(ops_status_code::invalid_request, "ConvND geometry is invalid");
        }
        desc.kernel_volume *= desc.kernel_size[axis];
        input_volume *= desc.input_size[axis];
        const int64_t effective_kernel = params.dilation[axis] * (desc.kernel_size[axis] - 1) + 1;
        desc.output_size[axis] = expected_transposed
            ? (desc.input_size[axis] - 1) * params.stride[axis] - params.padding_before[axis] -
                  params.padding_after[axis] + effective_kernel + params.output_padding[axis]
            : (desc.input_size[axis] + params.padding_before[axis] + params.padding_after[axis] -
                  effective_kernel) / params.stride[axis] + 1;
        if (desc.output_size[axis] <= 0) {
            return ops_status::error(ops_status_code::invalid_request, "ConvND output geometry is empty");
        }
        output_volume *= desc.output_size[axis];
    }

    if (expected_spatial_dims == 2) {
        if (input->ne[0] != desc.input_size[0] || input->ne[1] != desc.input_size[1]) {
            return ops_status::error(ops_status_code::invalid_request, "Conv2D input shape does not match its logical geometry");
        }
        desc.input_channels = input->ne[2];
        desc.batch = input->ne[3];
    } else {
        if (input->ne[0] != input_volume) {
            return ops_status::error(ops_status_code::invalid_request, "Packed Conv3D volume does not match its logical geometry");
        }
        desc.input_channels = input->ne[1];
        desc.batch = input->ne[2];
    }
    if (desc.input_channels <= 0 || desc.batch <= 0 || desc.input_channels % desc.groups != 0) {
        return ops_status::error(ops_status_code::invalid_request, "ConvND input channels or batch is invalid");
    }
    desc.input_channels_per_group = desc.input_channels / desc.groups;

    // Packed channel-row weights: Conv [Cin/G, KVolume, Cout], transpose
    // [Cout/G, KVolume, Cin]. This layout works for float and GGML block types.
    if (desc.weight_layout == ops_weight_layout::flattened_rows && !expected_transposed) {
        if (weight->ne[0] != desc.input_channels_per_group * desc.kernel_volume ||
            weight->ne[2] != 1 || weight->ne[3] != 1) {
            return ops_status::error(ops_status_code::invalid_request, "Flattened ConvND weight row is invalid");
        }
        desc.output_channels = weight->ne[1];
        if (desc.output_channels <= 0 || desc.output_channels % desc.groups != 0) {
            return ops_status::error(ops_status_code::invalid_request, "Flattened ConvND output channels are invalid");
        }
        desc.output_channels_per_group = desc.output_channels / desc.groups;
    } else if (desc.weight_layout == ops_weight_layout::flattened_rows) {
        if (weight->ne[1] != desc.input_channels || weight->ne[0] % desc.kernel_volume != 0 ||
            weight->ne[2] != 1 || weight->ne[3] != 1) {
            return ops_status::error(ops_status_code::invalid_request, "Flattened ConvTransposeND weight row is invalid");
        }
        desc.output_channels_per_group = weight->ne[0] / desc.kernel_volume;
        desc.output_channels = desc.output_channels_per_group * desc.groups;
    } else if (!expected_transposed) {
        if (weight->ne[0] != desc.input_channels_per_group || weight->ne[1] != desc.kernel_volume) {
            return ops_status::error(ops_status_code::invalid_request, "ConvND weight shape is invalid");
        }
        desc.output_channels = weight->ne[2];
        if (desc.output_channels <= 0 || desc.output_channels % desc.groups != 0) {
            return ops_status::error(ops_status_code::invalid_request, "ConvND output channels are not divisible by groups");
        }
        desc.output_channels_per_group = desc.output_channels / desc.groups;
    } else {
        if (weight->ne[1] != desc.kernel_volume || weight->ne[2] != desc.input_channels) {
            return ops_status::error(ops_status_code::invalid_request, "ConvTransposeND weight shape is invalid");
        }
        desc.output_channels_per_group = weight->ne[0];
        desc.output_channels = desc.output_channels_per_group * desc.groups;
    }
    if (bias && (bias->ne[0] != desc.output_channels || ggml_nelements(bias) != desc.output_channels)) {
        return ops_status::error(ops_status_code::invalid_request, "ConvND bias shape is invalid");
    }
    if (request.output) {
        const bool shape_ok = expected_spatial_dims == 2
            ? request.output->ne[0] == desc.output_size[0] && request.output->ne[1] == desc.output_size[1] &&
                  request.output->ne[2] == desc.output_channels && request.output->ne[3] == desc.batch
            : request.output->ne[0] == output_volume && request.output->ne[1] == desc.output_channels &&
                  request.output->ne[2] == desc.batch;
        if (!shape_ok || request.output->type != input->type) {
            return ops_status::error(ops_status_code::invalid_request, "ConvND output tensor is incompatible");
        }
    }
    if (output_desc) *output_desc = desc;
    return ops_status::ok();
}

} // namespace ggml_ops_ext
