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

} // namespace nn::functional
