#include "nn/layers/convolution.h"

#include "nn/core/context.h"
#include "nn/functional/convolution.h"

#include <stdexcept>
#include <string>

namespace nn {

static void check_weight_rank(const Shape& shape, size_t expected, const char* layer) {
    if (shape.size() != expected) {
        throw std::logic_error(
            std::string("nn::") + layer + ": weight must have rank " +
            std::to_string(expected) + ", got " + std::to_string(shape.size()));
    }
}

ggml_tensor* Conv1d::forward(
    Context& context, ggml_tensor* input, ggml_backend_t selected_backend
) {
    ggml_context* ctx = context.native_handle();
    ggml_backend_t target = selected_backend ? selected_backend : backend;
    return functional::conv1d(
        ctx, input, weight.tensor(), bias.local_tensor(),
        stride, padding, dilation, groups, target);
}

ggml_tensor* ConvTranspose1d::forward(
    Context& context, ggml_tensor* input, ggml_backend_t selected_backend
) {
    ggml_context* ctx = context.native_handle();
    ggml_backend_t target = selected_backend ? selected_backend : backend;
    return functional::conv_transpose1d(
        ctx, input, weight.tensor(), bias.local_tensor(), stride, padding, groups, target);
}

ggml_tensor* Conv2d::forward(
    Context& context, ggml_tensor* input, ggml_backend_t selected_backend
) {
    ggml_context* ctx = context.native_handle();
    const Shape& shape = weight.logical_shape();
    check_weight_rank(shape, 4, "Conv2d");
    return functional::conv2d(
        ctx, input, weight.tensor(), bias.local_tensor(),
        stride_width, stride_height, padding_width, padding_height, dilation_width, dilation_height,
        shape[0], shape[1], shape[2], shape[3], groups,
        selected_backend ? selected_backend : backend);
}

ggml_tensor* ConvTranspose2d::forward(
    Context& context, ggml_tensor* input, ggml_backend_t selected_backend
) {
    ggml_context* ctx = context.native_handle();
    const Shape& shape = weight.logical_shape();
    check_weight_rank(shape, 4, "ConvTranspose2d");
    ggml_ops_ext::ops_conv_nd_config config;
    config.spatial_dims = 2;
    config.input_size[0] = static_cast<int32_t>(input->ne[0]);
    config.input_size[1] = static_cast<int32_t>(input->ne[1]);
    config.kernel_size[0] = static_cast<int32_t>(shape[0]);
    config.kernel_size[1] = static_cast<int32_t>(shape[1]);
    config.groups = groups;
    for (int axis = 0; axis < 2; ++axis) {
        config.stride[axis] = stride[axis];
        config.padding_before[axis] = config.padding_after[axis] = padding[axis];
        config.output_padding[axis] = output_padding[axis];
        config.dilation[axis] = dilation[axis];
    }
    ggml_tensor* direct_weight = weight.tensor();
    if (ggml_n_dims(direct_weight) == 4 && !ggml_is_quantized(direct_weight->type)) {
        direct_weight = ggml_reshape_2d(
            ctx, direct_weight, shape[0] * shape[1] * shape[2], shape[3]);
        config.weight_layout = ggml_ops_ext::ops_weight_layout::flattened_rows;
    } else if (ggml_n_dims(direct_weight) == 2) {
        config.weight_layout = ggml_ops_ext::ops_weight_layout::flattened_rows;
    }
    return functional::conv_transpose2d(
        ctx, input, direct_weight, config, bias.local_tensor(),
        selected_backend ? selected_backend : backend);
}

static ggml_ops_ext::ops_conv_nd_config make_conv3d_config(
    const Shape& shape, const int input_size[3], const int stride[3], const int padding[3],
    const int dilation[3], const int* output_padding, int groups
) {
    ggml_ops_ext::ops_conv_nd_config config;
    config.spatial_dims = 3;
    config.groups = groups;
    for (int axis = 0; axis < 3; ++axis) {
        config.input_size[axis] = input_size[axis];
        config.kernel_size[axis] = static_cast<int32_t>(shape[axis]);
        config.stride[axis] = stride[axis];
        config.padding_before[axis] = config.padding_after[axis] = padding[axis];
        config.dilation[axis] = dilation[axis];
        config.output_padding[axis] = output_padding ? output_padding[axis] : 0;
    }
    return config;
}

ggml_tensor* Conv3d::forward(
    Context& context, ggml_tensor* input, ggml_backend_t selected_backend
) {
    ggml_context* ctx = context.native_handle();
    const Shape& shape = weight.logical_shape();
    check_weight_rank(shape, 5, "Conv3d");
    const auto config = make_conv3d_config(shape, input_size, stride, padding, dilation, nullptr, groups);
    return functional::conv3d(
        ctx, input, weight.tensor(), config, bias.local_tensor(),
        selected_backend ? selected_backend : backend);
}

ggml_tensor* ConvTranspose3d::forward(
    Context& context, ggml_tensor* input, ggml_backend_t selected_backend
) {
    ggml_context* ctx = context.native_handle();
    const Shape& shape = weight.logical_shape();
    check_weight_rank(shape, 5, "ConvTranspose3d");
    const auto config = make_conv3d_config(shape, input_size, stride, padding, dilation, output_padding, groups);
    return functional::conv_transpose3d(
        ctx, input, weight.tensor(), config, bias.local_tensor(),
        selected_backend ? selected_backend : backend);
}

} // namespace nn
