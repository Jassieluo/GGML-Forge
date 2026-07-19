#pragma once

#include "ggml-backend.h"
#include "ggml.h"
#include "nn/core/context.h"
#include "ops/contracts/resize_nd.h"

#include <cstdint>

namespace nn::functional {

ggml_tensor* interpolate_nearest_2x(ggml_context* ctx, ggml_tensor* input);
ggml_tensor* interpolate_nearest(Context& context, ggml_tensor* input, int64_t target_length,
                                 double scale_factor = 0.0);
ggml_tensor* interpolate_linear(Context& context, ggml_tensor* input, int64_t target_length);

// Native ResizeND uses [W,C,N], [W,H,C,N], and packed [W*H*D,C,N].
ggml_tensor* resize1d(ggml_context* ctx, ggml_tensor* input,
                      const ggml_ops_ext::ops_resize_nd_config& config,
                      ggml_backend_t backend = nullptr);
ggml_tensor* resize2d(ggml_context* ctx, ggml_tensor* input,
                      const ggml_ops_ext::ops_resize_nd_config& config,
                      ggml_backend_t backend = nullptr);
ggml_tensor* resize3d(ggml_context* ctx, ggml_tensor* input,
                      const ggml_ops_ext::ops_resize_nd_config& config,
                      ggml_backend_t backend = nullptr);

} // namespace nn::functional
