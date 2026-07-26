#pragma once

#include "ops/types.h"
#include <cstring>

namespace ggml_ops_ext {

struct ops_conv_1d_contract_params {
    int32_t stride = 1;
    int32_t padding = 0;
    int32_t dilation = 1;
    int32_t groups = 1;
};
static_assert(sizeof(ops_conv_1d_contract_params) == 4 * sizeof(int32_t));

struct ops_conv_weight_desc {
    int64_t kernel = 0;
    int64_t input_channels_per_group = 0;
    int64_t output_channels_per_group = 0;
    int64_t output_channels = 0;
    bool packed = false;
};

inline bool ops_describe_conv_weight(
    int op_id,
    int conv_op_id,
    int conv_transpose_op_id,
    const ggml_tensor* w,
    const ggml_tensor* x,
    int64_t groups,
    ops_conv_weight_desc& desc
) {
    // Kernels and the builder treat activations as 3D [L, C, N]; a fourth
    // dimension would be silently dropped, so reject it here.
    if (!w || !x || groups <= 0 || x->ne[1] <= 0 || x->ne[1] % groups != 0 || x->ne[3] > 1) return false;
    desc.packed = ggml_is_quantized(w->type);
    desc.input_channels_per_group = x->ne[1] / groups;

    if (op_id == conv_op_id) {
        desc.kernel = desc.packed ? w->ne[1] : w->ne[0];
        const int64_t stored_input_channels = desc.packed ? w->ne[0] : w->ne[1];
        desc.output_channels = w->ne[2];
        if (stored_input_channels != desc.input_channels_per_group ||
            desc.output_channels <= 0 || desc.output_channels % groups != 0) return false;
        desc.output_channels_per_group = desc.output_channels / groups;
        return desc.kernel > 0;
    }

    if (op_id == conv_transpose_op_id) {
        desc.kernel = desc.packed ? w->ne[1] : w->ne[0];
        desc.output_channels_per_group = desc.packed ? w->ne[0] : w->ne[1];
        const int64_t stored_input_channels = w->ne[2];
        desc.output_channels = desc.output_channels_per_group * groups;
        return desc.kernel > 0 && desc.output_channels_per_group > 0 &&
               stored_input_channels == x->ne[1];
    }

    return false;
}

inline ops_status ops_validate_conv_contract(
    int op_id,
    int conv_op_id,
    int conv_transpose_op_id,
    ggml_tensor* const* srcs,
    int n_srcs,
    const void* raw_params,
    size_t params_size,
    int64_t* output_length = nullptr,
    ops_conv_weight_desc* output_weight_desc = nullptr
) {
    if (!srcs || n_srcs < 2 || !srcs[0] || !srcs[1]) {
        return ops_status::error(ops_status_code::invalid_request, "Conv requires weight and activation tensors");
    }
    if (!raw_params || params_size < sizeof(ops_conv_1d_contract_params)) {
        return ops_status::error(ops_status_code::invalid_request, "Conv parameters are missing or truncated");
    }

    ops_conv_1d_contract_params params;
    std::memcpy(&params, raw_params, sizeof(params));
    if (params.stride <= 0 || params.padding < 0 || params.dilation <= 0 || params.groups <= 0) {
        return ops_status::error(ops_status_code::invalid_request, "Conv stride, padding, dilation, or groups is invalid");
    }

    const ggml_tensor* w = srcs[0];
    const ggml_tensor* x = srcs[1];
    const ggml_tensor* bias = n_srcs >= 3 ? srcs[2] : nullptr;
    ops_conv_weight_desc weight_desc;
    if (x->ne[0] <= 0 || !ops_describe_conv_weight(
            op_id, conv_op_id, conv_transpose_op_id, w, x, params.groups, weight_desc)) {
        return ops_status::error(ops_status_code::invalid_request, "Conv weight layout or channel grouping is invalid");
    }

    int64_t length = 0;
    if (op_id == conv_op_id) {
        length = (x->ne[0] + 2LL * params.padding - params.dilation * (weight_desc.kernel - 1) - 1) /
                 params.stride + 1;
    } else if (op_id == conv_transpose_op_id) {
        length = (x->ne[0] - 1) * params.stride - 2LL * params.padding +
                 params.dilation * (weight_desc.kernel - 1) + 1;
    } else {
        return ops_status::error(ops_status_code::invalid_request, "Unknown Conv contract operation");
    }

    if (length <= 0 || (bias && bias->ne[0] != weight_desc.output_channels)) {
        return ops_status::error(ops_status_code::invalid_request, "Conv output shape or bias shape is invalid");
    }
    if (output_length) *output_length = length;
    if (output_weight_desc) *output_weight_desc = weight_desc;
    return ops_status::ok();
}

} // namespace ggml_ops_ext
