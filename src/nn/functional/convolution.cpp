#include "nn/functional/activation.h"
#include "nn/functional/convolution.h"

#include "ops/ops.h"

namespace nn::functional {

ggml_tensor* conv1d(
    ggml_context* ctx, ggml_tensor* input, ggml_tensor* weight, ggml_tensor* bias,
    int stride, int padding, int dilation, int groups, ggml_backend_t backend
) {
    ggml_tensor* transposed = ggml_cont(ctx, ggml_transpose(ctx, input));
    ggml_tensor* output = ggml_ops_conv_1d(
        ctx, weight, transposed, stride, padding, dilation, groups, backend, bias);
    return output ? ggml_cont(ctx, ggml_transpose(ctx, output)) : nullptr;
}

ggml_tensor* conv1d_no_transpose(
    ggml_context* ctx, ggml_tensor* input, ggml_tensor* weight, ggml_tensor* bias,
    int stride, int padding, int dilation, int groups, ggml_backend_t backend
) {
    return ggml_ops_conv_1d(
        ctx, weight, input, stride, padding, dilation, groups, backend, bias);
}

ggml_tensor* conv_transpose1d(
    ggml_context* ctx, ggml_tensor* input, ggml_tensor* weight, ggml_tensor* bias,
    int stride, int padding, int groups, ggml_backend_t backend
) {
    ggml_tensor* transposed = ggml_cont(ctx, ggml_transpose(ctx, input));
    ggml_tensor* output = ggml_ops_conv_transpose_1d(
        ctx, weight, transposed, stride, padding, 1, groups, backend, bias);
    return output ? ggml_cont(ctx, ggml_transpose(ctx, output)) : nullptr;
}

ggml_tensor* conv_transpose1d_no_transpose(
    ggml_context* ctx, ggml_tensor* input, ggml_tensor* weight, ggml_tensor* bias,
    int stride, int padding, ggml_backend_t backend
) {
    return ggml_ops_conv_transpose_1d(
        ctx, weight, input, stride, padding, 1, 1, backend, bias);
}

ggml_tensor* alias_free_activation1d(
    ggml_context* ctx, ggml_tensor* input, ggml_tensor* up_filter,
    ggml_tensor* down_filter, ggml_tensor* alpha, ggml_tensor* beta,
    ggml_backend_t backend
) {
    if (ggml_tensor* fused = ggml_ops_alias_free_activation(
            ctx, input, up_filter, down_filter, alpha, beta, backend)) {
        return fused;
    }
    const int channels = static_cast<int>(input->ne[1]);
    ggml_tensor* upsampled = ggml_ops_conv_transpose_1d(
        ctx, up_filter, input, 2, 5, 1, channels, backend);
    if (!upsampled) return nullptr;
    upsampled = ggml_scale(ctx, upsampled, 2.0f);
    ggml_tensor* activated = ggml_ops_snake_beta(ctx, upsampled, alpha, beta, backend);
    if (!activated) return nullptr;
    return ggml_ops_conv_1d(
        ctx, down_filter, activated, 2, 5, 1, channels, backend);
}

} // namespace nn::functional
