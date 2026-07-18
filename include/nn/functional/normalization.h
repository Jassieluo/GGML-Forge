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

} // namespace nn::functional
