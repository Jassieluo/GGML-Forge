#pragma once

#include "nn/layers/linear.h"

namespace nn {

class LayerNorm : public Module<LayerNorm> {
public:
    Parameter& gamma = parameter("weight", Parameter::optional(
        std::nullopt, Parameter::Usage::norm_affine));
    Parameter& beta = parameter("bias", Parameter::optional(
        std::nullopt, Parameter::Usage::norm_affine));
    float eps = 1e-5f;

    LayerNorm() = default;
    LayerNorm(ggml_tensor* gamma_value, ggml_tensor* beta_value, float epsilon = 1e-5f)
        : eps(epsilon) {
        if (gamma_value) gamma.bind(gamma_value);
        if (beta_value) beta.bind(beta_value);
    }

    ggml_tensor* forward(Context& context, ggml_tensor* input, ggml_backend_t backend = nullptr);

    // LayerNorm(input + residual) as one node where the backend supports the
    // fused kernel; composes add + layer_norm otherwise. Same semantics.
    ggml_tensor* forward_residual(
        Context& context, ggml_tensor* input, ggml_tensor* residual,
        ggml_backend_t backend = nullptr);
};

class InstanceNorm : public Module<InstanceNorm> {
public:
    Parameter& gamma = parameter("weight", Parameter::optional(
        std::nullopt, Parameter::Usage::norm_affine));
    Parameter& beta = parameter("bias", Parameter::optional(
        std::nullopt, Parameter::Usage::norm_affine));
    float eps = 1e-5f;

    InstanceNorm() = default;
    InstanceNorm(ggml_tensor* gamma_value, ggml_tensor* beta_value, float epsilon = 1e-5f)
        : eps(epsilon) {
        if (gamma_value) gamma.bind(gamma_value);
        if (beta_value) beta.bind(beta_value);
    }

    ggml_tensor* forward(Context& context, ggml_tensor* input, ggml_backend_t backend = nullptr);
};

class AdaLN : public Module<AdaLN> {
public:
    float eps = 1e-5f;

    AdaLN() = default;
    explicit AdaLN(float epsilon) : eps(epsilon) {}

    ggml_tensor* forward(
        Context& context, ggml_tensor* input, ggml_tensor* scale, ggml_tensor* shift,
        ggml_backend_t backend = nullptr);
};

class AdaLayerNormZero : public Module<AdaLayerNormZero> {
public:
    Linear& linear = submodule<Linear>("linear");
    float eps = 1e-6f;

    AdaLayerNormZero() = default;
    AdaLayerNormZero(ggml_tensor* weight, ggml_tensor* bias, float epsilon = 1e-6f)
        : eps(epsilon) {
        if (weight) linear.weight.bind(weight);
        if (bias) linear.bias.bind(bias);
    }

    struct Output {
        ggml_tensor* x_modulated = nullptr;
        ggml_tensor* gate_msa = nullptr;
        ggml_tensor* shift_mlp = nullptr;
        ggml_tensor* scale_mlp = nullptr;
        ggml_tensor* gate_mlp = nullptr;
    };

    Output forward(
        Context& context, ggml_tensor* input, ggml_tensor* embedding,
        ggml_backend_t backend = nullptr);
};

} // namespace nn
