#pragma once

#include "nn/core/module.h"

namespace nn {

class Conv1d : public Module<Conv1d> {
public:
    Parameter& weight = parameter("weight", Parameter::required(
        std::nullopt, Parameter::Usage::conv1d_weight));
    Parameter& bias = parameter("bias", Parameter::optional(
        std::nullopt, Parameter::Usage::bias));
    int stride = 1;
    int padding = 0;
    int dilation = 1;
    int groups = 1;

    Conv1d() = default;
    Conv1d(
        ggml_tensor* weight_value,
        ggml_tensor* bias_value = nullptr,
        int stride_value = 1,
        int padding_value = 0,
        int dilation_value = 1,
        int groups_value = 1)
        : stride(stride_value), padding(padding_value), dilation(dilation_value), groups(groups_value) {
        if (weight_value) weight.bind(weight_value);
        if (bias_value) bias.bind(bias_value);
    }

    ggml_tensor* forward(Context& context, ggml_tensor* input, ggml_backend_t backend = nullptr);
};

class ConvTranspose1d : public Module<ConvTranspose1d> {
public:
    Parameter& weight = parameter("weight", Parameter::required(
        std::nullopt, Parameter::Usage::conv_transpose1d_weight));
    Parameter& bias = parameter("bias", Parameter::optional(
        std::nullopt, Parameter::Usage::bias));
    int stride = 1;
    int padding = 0;
    int dilation = 1;
    int groups = 1;

    ConvTranspose1d() = default;
    ConvTranspose1d(
        ggml_tensor* weight_value,
        ggml_tensor* bias_value = nullptr,
        int stride_value = 1,
        int padding_value = 0,
        int dilation_value = 1,
        int groups_value = 1)
        : stride(stride_value), padding(padding_value), dilation(dilation_value), groups(groups_value) {
        if (weight_value) weight.bind(weight_value);
        if (bias_value) bias.bind(bias_value);
    }

    ggml_tensor* forward(Context& context, ggml_tensor* input, ggml_backend_t backend = nullptr);
};

class Conv2d : public Module<Conv2d> {
public:
    Parameter& weight = parameter("weight", Parameter::required(
        std::nullopt, Parameter::Usage::conv2d_weight));
    Parameter& bias = parameter("bias", Parameter::optional(
        std::nullopt, Parameter::Usage::bias));
    int stride_width = 1;
    int stride_height = 1;
    int padding_width = 0;
    int padding_height = 0;
    int dilation_width = 1;
    int dilation_height = 1;
    int groups = 1;

    Conv2d() = default;
    Conv2d(int stride, int padding)
        : stride_width(stride), stride_height(stride),
          padding_width(padding), padding_height(padding) {}

    ggml_tensor* forward(Context& context, ggml_tensor* input, ggml_backend_t backend = nullptr);
};

class ConvTranspose2d : public Module<ConvTranspose2d> {
public:
    Parameter& weight = parameter("weight", Parameter::required(
        std::nullopt, Parameter::Usage::conv_transpose2d_weight));
    Parameter& bias = parameter("bias", Parameter::optional(
        std::nullopt, Parameter::Usage::bias));
    int stride[2] = { 1, 1 };
    int padding[2] = { 0, 0 };
    int output_padding[2] = { 0, 0 };
    int dilation[2] = { 1, 1 };
    int groups = 1;

    ggml_tensor* forward(Context& context, ggml_tensor* input, ggml_backend_t backend = nullptr);
};

class Conv3d : public Module<Conv3d> {
public:
    Parameter& weight = parameter("weight", Parameter::required(
        std::nullopt, Parameter::Usage::conv3d_weight));
    Parameter& bias = parameter("bias", Parameter::optional(
        std::nullopt, Parameter::Usage::bias));
    int input_size[3] = {};
    int stride[3] = { 1, 1, 1 };
    int padding[3] = { 0, 0, 0 };
    int dilation[3] = { 1, 1, 1 };
    int groups = 1;

    ggml_tensor* forward(Context& context, ggml_tensor* input, ggml_backend_t backend = nullptr);
};

class ConvTranspose3d : public Module<ConvTranspose3d> {
public:
    Parameter& weight = parameter("weight", Parameter::required(
        std::nullopt, Parameter::Usage::conv_transpose3d_weight));
    Parameter& bias = parameter("bias", Parameter::optional(
        std::nullopt, Parameter::Usage::bias));
    int input_size[3] = {};
    int stride[3] = { 1, 1, 1 };
    int padding[3] = { 0, 0, 0 };
    int output_padding[3] = { 0, 0, 0 };
    int dilation[3] = { 1, 1, 1 };
    int groups = 1;

    ggml_tensor* forward(Context& context, ggml_tensor* input, ggml_backend_t backend = nullptr);
};

} // namespace nn
