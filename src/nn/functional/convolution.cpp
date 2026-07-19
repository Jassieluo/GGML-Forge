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
    int64_t input_channels, int64_t output_channels,
    int groups, ggml_backend_t backend
) {
    const bool geometry_ok = kernel_width > 0 && kernel_height > 0 && groups > 0 &&
        input_channels > 0 && input_channels % groups == 0 &&
        output_channels > 0 && output_channels % groups == 0;
    const bool channel_rows = geometry_ok &&
        weight->ne[0] == input_channels / groups &&
        weight->ne[1] == kernel_width * kernel_height && weight->ne[2] == output_channels;
    const bool flattened_rows = geometry_ok && ggml_n_dims(weight) == 2 &&
        weight->ne[0] == kernel_width * kernel_height * (input_channels / groups) &&
        weight->ne[1] == output_channels;
    const bool native_4d = geometry_ok && !ggml_is_quantized(weight->type) &&
        weight->ne[0] == kernel_width && weight->ne[1] == kernel_height &&
        weight->ne[2] == input_channels / groups && weight->ne[3] == output_channels;
    if (channel_rows || flattened_rows || native_4d) {
        ggml_ops_ext::ops_conv_nd_config config;
        config.spatial_dims = 2;
        config.input_size[0] = static_cast<int32_t>(input->ne[0]);
        config.input_size[1] = static_cast<int32_t>(input->ne[1]);
        config.kernel_size[0] = static_cast<int32_t>(kernel_width);
        config.kernel_size[1] = static_cast<int32_t>(kernel_height);
        config.stride[0] = stride_width;
        config.stride[1] = stride_height;
        config.padding_before[0] = config.padding_after[0] = padding_width;
        config.padding_before[1] = config.padding_after[1] = padding_height;
        config.dilation[0] = dilation_width;
        config.dilation[1] = dilation_height;
        config.groups = groups;
        config.weight_layout = channel_rows
            ? ggml_ops_ext::ops_weight_layout::channel_rows
            : ggml_ops_ext::ops_weight_layout::flattened_rows;
        ggml_tensor* direct_weight = native_4d
            ? ggml_reshape_2d(ctx, weight, kernel_width * kernel_height * (input_channels / groups), output_channels)
            : weight;
        return ggml_ops_conv_2d(ctx, direct_weight, input, config, backend, bias);
    }

    // Conv2d is intentionally direct-only. Silently falling back to GGML's
    // im2col implementation would reintroduce an activation-sized workspace
    // for malformed or unsupported weight layouts.
    return nullptr;
}

ggml_tensor* conv_transpose2d(
    ggml_context* ctx, ggml_tensor* input, ggml_tensor* weight,
    const ggml_ops_ext::ops_conv_nd_config& config,
    ggml_tensor* bias, ggml_backend_t backend
) {
    return ggml_ops_conv_transpose_2d(ctx, weight, input, config, backend, bias);
}

ggml_tensor* conv3d(
    ggml_context* ctx, ggml_tensor* input, ggml_tensor* weight,
    const ggml_ops_ext::ops_conv_nd_config& config,
    ggml_tensor* bias, ggml_backend_t backend
) {
    return ggml_ops_conv_3d(ctx, weight, input, config, backend, bias);
}

ggml_tensor* conv_transpose3d(
    ggml_context* ctx, ggml_tensor* input, ggml_tensor* weight,
    const ggml_ops_ext::ops_conv_nd_config& config,
    ggml_tensor* bias, ggml_backend_t backend
) {
    return ggml_ops_conv_transpose_3d(ctx, weight, input, config, backend, bias);
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
