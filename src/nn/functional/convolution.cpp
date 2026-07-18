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

ggml_tensor* conv2d(
    ggml_context* ctx, ggml_tensor* input, ggml_tensor* weight, ggml_tensor* bias,
    int stride_width, int stride_height,
    int padding_width, int padding_height,
    int dilation_width, int dilation_height,
    int64_t kernel_width, int64_t kernel_height,
    int64_t input_channels, int64_t output_channels
) {
    ggml_tensor* output = nullptr;
    if (ggml_n_dims(weight) == 2 && ggml_is_quantized(weight->type)) {
        if (kernel_width <= 0 || kernel_height <= 0 || input_channels <= 0 ||
            output_channels <= 0 || weight->ne[0] != kernel_width * kernel_height * input_channels ||
            weight->ne[1] != output_channels) return nullptr;
        // IM2COL consumes the kernel tensor only as a shape descriptor. The
        // actual flattened Q4/Q8 rows are consumed directly by MUL_MAT.
        ggml_tensor* kernel_shape = ggml_new_tensor_4d(
            ctx, GGML_TYPE_F16, kernel_width, kernel_height, input_channels, output_channels);
        ggml_tensor* columns = ggml_im2col(
            ctx, kernel_shape, input,
            stride_width, stride_height,
            padding_width, padding_height,
            dilation_width, dilation_height, true, GGML_TYPE_F16);
        ggml_tensor* product = ggml_mul_mat(
            ctx, weight,
            ggml_reshape_2d(
                ctx, columns, columns->ne[0],
                columns->ne[3] * columns->ne[2] * columns->ne[1]));
        output = ggml_reshape_4d(
            ctx, product, output_channels, columns->ne[1], columns->ne[2], columns->ne[3]);
        output = ggml_cont(ctx, ggml_permute(ctx, output, 2, 0, 1, 3));
    } else {
        output = ggml_conv_2d(
            ctx, weight, input,
            stride_width, stride_height,
            padding_width, padding_height,
            dilation_width, dilation_height);
    }
    if (!output || !bias) return output;
    ggml_tensor* shaped_bias = ggml_reshape_4d(ctx, bias, 1, 1, bias->ne[0], 1);
    return ggml_add(ctx, output, ggml_repeat(ctx, shaped_bias, output));
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
