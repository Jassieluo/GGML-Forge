#pragma once

#include "ggml-backend.h"
#include "ggml.h"

namespace nn::functional {

ggml_tensor* attention(
    ggml_context* ctx,
    ggml_tensor* q,
    ggml_tensor* k,
    ggml_tensor* v,
    ggml_tensor* mask = nullptr,
    ggml_tensor* weights = nullptr,
    float scale = 1.0f,
    int sliding_window = -1,
    ggml_backend_t backend = nullptr);

ggml_tensor* relative_position_keys(
    ggml_context* ctx,
    ggml_tensor* q,
    ggml_tensor* embedding,
    float scale,
    int window,
    ggml_backend_t backend = nullptr);

ggml_tensor* relative_position_values(
    ggml_context* ctx,
    ggml_tensor* weights,
    ggml_tensor* embedding,
    ggml_tensor* attention_output,
    int window,
    ggml_backend_t backend = nullptr);

} // namespace nn::functional
