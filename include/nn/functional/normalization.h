#pragma once

#include "ggml-backend.h"
#include "ggml.h"

namespace nn::functional {

ggml_tensor* layer_norm(
    ggml_context* ctx,
    ggml_tensor* x,
    ggml_tensor* gamma,
    ggml_tensor* beta,
    float eps = 1e-5f,
    ggml_backend_t backend = nullptr);

ggml_tensor* rms_norm(
    ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight = nullptr,
    float eps = 1e-5f);

ggml_tensor* group_norm(
    ggml_context* ctx, ggml_tensor* x, int groups,
    ggml_tensor* weight = nullptr, ggml_tensor* bias = nullptr,
    float eps = 1e-5f);

ggml_tensor* l2_normalize(
    ggml_context* ctx, ggml_tensor* x, int axis = 0, float eps = 1e-12f);

} // namespace nn::functional
