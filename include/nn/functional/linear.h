#pragma once

#include "ggml-backend.h"
#include "ggml.h"

namespace nn::functional {

ggml_tensor* linear(
    ggml_context* ctx,
    ggml_tensor* x,
    ggml_tensor* weight,
    ggml_tensor* bias = nullptr,
    ggml_backend_t backend = nullptr);

} // namespace nn::functional
