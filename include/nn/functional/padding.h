#pragma once

#include "ggml-backend.h"
#include "ggml.h"
#include "ops/contracts/pad_nd.h"

namespace nn::functional {

ggml_tensor* pad1d(ggml_context* ctx, ggml_tensor* input,
                   const ggml_ops_ext::ops_pad_nd_config& config, ggml_backend_t backend = nullptr);
ggml_tensor* pad2d(ggml_context* ctx, ggml_tensor* input,
                   const ggml_ops_ext::ops_pad_nd_config& config, ggml_backend_t backend = nullptr);
ggml_tensor* pad3d(ggml_context* ctx, ggml_tensor* input,
                   const ggml_ops_ext::ops_pad_nd_config& config, ggml_backend_t backend = nullptr);

} // namespace nn::functional
