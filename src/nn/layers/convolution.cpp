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

} // namespace nn
