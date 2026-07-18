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

ggml_tensor* Conv2d::forward(ggml_context* ctx, ggml_tensor* input) {
    const Shape& shape = weight.logical_shape();
    if (shape.size() != 4) return nullptr;
    return functional::conv2d(
        ctx, input, weight.tensor(), bias.tensor(),
        stride_width, stride_height,
        padding_width, padding_height,
        dilation_width, dilation_height,
        shape[0], shape[1], shape[2], shape[3]);
}

} // namespace nn
