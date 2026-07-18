#pragma once

#include "ggml-backend.h"
#include "ggml.h"

namespace nn::functional {

ggml_tensor* conv1d(
    ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, ggml_tensor* bias = nullptr,
    int stride = 1, int padding = 0, int dilation = 1, int groups = 1,
    ggml_backend_t backend = nullptr);

ggml_tensor* conv1d_no_transpose(
    ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, ggml_tensor* bias = nullptr,
    int stride = 1, int padding = 0, int dilation = 1, int groups = 1,
    ggml_backend_t backend = nullptr);

ggml_tensor* conv_transpose1d(
    ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, ggml_tensor* bias = nullptr,
    int stride = 1, int padding = 0, int groups = 1,
    ggml_backend_t backend = nullptr);

ggml_tensor* conv_transpose1d_no_transpose(
    ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, ggml_tensor* bias = nullptr,
    int stride = 1, int padding = 0,
    ggml_backend_t backend = nullptr);

// Input is [width, height, channels, batch] in GGML dimension order. Kernel
// storage is [kernel_width, kernel_height, input_channels, output_channels].
ggml_tensor* conv2d(
    ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, ggml_tensor* bias = nullptr,
    int stride_width = 1, int stride_height = 1,
    int padding_width = 0, int padding_height = 0,
    int dilation_width = 1, int dilation_height = 1,
    int64_t kernel_width = 0, int64_t kernel_height = 0,
    int64_t input_channels = 0, int64_t output_channels = 0);

} // namespace nn::functional
