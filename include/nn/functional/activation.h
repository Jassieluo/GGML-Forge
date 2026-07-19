#pragma once

#include "ggml-backend.h"
#include "ggml.h"

namespace nn::functional {

ggml_tensor* mish(ggml_context* ctx, ggml_tensor* x, ggml_backend_t backend = nullptr);

ggml_tensor* gated_tanh_sigmoid(
    ggml_context* ctx,
    ggml_tensor* x,
    int channels,
    ggml_backend_t backend = nullptr);

ggml_tensor* alias_free_activation1d(
    ggml_context* ctx,
    ggml_tensor* x,
    ggml_tensor* up_filter,
    ggml_tensor* down_filter,
    ggml_tensor* alpha,
    ggml_tensor* beta,
    ggml_backend_t backend = nullptr);

ggml_tensor* swiglu(ggml_context* ctx, ggml_tensor* x, int axis = 0);
ggml_tensor* geglu(ggml_context* ctx, ggml_tensor* x, int axis = 0);
ggml_tensor* reglu(ggml_context* ctx, ggml_tensor* x, int axis = 0);

} // namespace nn::functional
