#include "nn/layers/convolution.h"

#include "nn/functional/convolution.h"

namespace nn {

ggml_tensor* Conv1d::forward(
    ggml_context* ctx, ggml_tensor* input, ggml_backend_t selected_backend
) {
    ggml_backend_t target = selected_backend ? selected_backend : backend;
    return functional::conv1d(
        ctx, input, weight.tensor(), bias.local_tensor(),
        stride, padding, dilation, groups, target);
}

ggml_tensor* ConvTranspose1d::forward(
    ggml_context* ctx, ggml_tensor* input, ggml_backend_t selected_backend
) {
    ggml_backend_t target = selected_backend ? selected_backend : backend;
    return functional::conv_transpose1d(
        ctx, input, weight.tensor(), bias.local_tensor(), stride, padding, groups, target);
}

ggml_tensor* Conv2d::forward(
    ggml_context* ctx, ggml_tensor* input, ggml_backend_t selected_backend
) {
    const Shape& shape = weight.logical_shape();
    if (shape.size() != 4) return nullptr;
    return functional::conv2d(
        ctx, input, weight.tensor(), bias.local_tensor(),
        stride_width, stride_height, padding_width, padding_height, dilation_width, dilation_height,
        shape[0], shape[1], shape[2], shape[3], groups,
        selected_backend ? selected_backend : backend);
}

ggml_tensor* ConvTranspose2d::forward(
    ggml_context* ctx, ggml_tensor* input, ggml_backend_t selected_backend
) {
    const Shape& shape = weight.logical_shape();
    if (shape.size() != 4) return nullptr;
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
    return functional::conv_transpose2d(
        ctx, input, weight.tensor(), config, bias.local_tensor(),
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
    ggml_context* ctx, ggml_tensor* input, ggml_backend_t selected_backend
) {
    const Shape& shape = weight.logical_shape();
    if (shape.size() != 5) return nullptr;
    const auto config = make_conv3d_config(shape, input_size, stride, padding, dilation, nullptr, groups);
    return functional::conv3d(
        ctx, input, weight.tensor(), config, bias.local_tensor(),
        selected_backend ? selected_backend : backend);
}

ggml_tensor* ConvTranspose3d::forward(
    ggml_context* ctx, ggml_tensor* input, ggml_backend_t selected_backend
) {
    const Shape& shape = weight.logical_shape();
    if (shape.size() != 5) return nullptr;
    const auto config = make_conv3d_config(shape, input_size, stride, padding, dilation, output_padding, groups);
    return functional::conv_transpose3d(
        ctx, input, weight.tensor(), config, bias.local_tensor(),
        selected_backend ? selected_backend : backend);
}

} // namespace nn
