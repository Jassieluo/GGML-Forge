#pragma once

#include "nn/core/module.h"

namespace nn {

class Linear : public Module<Linear> {
public:
    Parameter& weight = parameter("weight", Parameter::required(
        std::nullopt, Parameter::Usage::linear_weight));
    Parameter& bias = parameter("bias", Parameter::optional(
        std::nullopt, Parameter::Usage::bias));

    Linear() = default;
    Linear(ggml_tensor* weight_value, ggml_tensor* bias_value = nullptr) : Linear() {
        if (weight_value) weight.bind(weight_value);
        if (bias_value) bias.bind(bias_value);
    }

    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* input);
};

} // namespace nn
