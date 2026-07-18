#pragma once

#include "ggml.h"
#include "nn/core/context.h"

#include <cstdint>

namespace nn::functional {

ggml_tensor* interpolate_nearest_2x(ggml_context* ctx, ggml_tensor* input);
ggml_tensor* interpolate_nearest(
    Context& context, ggml_tensor* input, int64_t target_length, double scale_factor = 0.0);
ggml_tensor* interpolate_linear(Context& context, ggml_tensor* input, int64_t target_length);

} // namespace nn::functional
