#pragma once

#include "ggml.h"
#include "nn/core/context.h"

#include <cstdint>

namespace nn::functional {

ggml_tensor* sinusoidal_position_embedding(
    Context& context,
    int64_t sequence_length,
    int64_t dimension,
    float theta = 10000.0f);

} // namespace nn::functional
