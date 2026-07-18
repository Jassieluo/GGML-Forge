#pragma once

#include "nn/core/module.h"

namespace nn {

class Snake : public Module<Snake> {
public:
    float alpha = 1.0f;
    Snake() = default;
    explicit Snake(float value) : alpha(value) {}
    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* input, ggml_backend_t backend = nullptr);
};

class PReLU : public Module<PReLU> {
public:
    Parameter& weight = parameter("weight", Parameter::optional(
        std::nullopt, Parameter::Usage::scalar));
    PReLU() = default;
    explicit PReLU(ggml_tensor* value) : PReLU() {
        if (value) weight.bind(value);
    }
    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* input, ggml_backend_t backend = nullptr);
};

class GLU : public Module<GLU> {
public:
    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* input, ggml_backend_t backend = nullptr);
};

enum class ActivationType {
    GELU,
    GELU_ERF,
    RELU,
    LEAKY_RELU,
    MISH,
    DOUBLE_SWISH,
    SILU,
};

} // namespace nn
