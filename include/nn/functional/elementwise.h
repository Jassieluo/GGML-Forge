#pragma once

#include "ggml.h"

namespace nn::functional {

ggml_tensor* add_scalar(ggml_context* ctx, ggml_tensor* input, float value);

} // namespace nn::functional
