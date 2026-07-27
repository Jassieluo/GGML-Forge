#pragma once

#include "nn/nn.h"

namespace visual_perception::depth::fastdepth {

enum class Activation { none, relu, relu6 };

class ConvAct final : public nn::Module<ConvAct> {
public:
    ConvAct(int kernel, int stride, int padding, int groups, Activation activation);
    ggml_tensor* forward(nn::Context& context, ggml_tensor* input, ggml_backend_t backend);
private:
    Activation activation_;
    nn::Conv2d& conv_;
};

class Separable final : public nn::Module<Separable> {
public:
    Separable(int channels, int kernel, int stride, int padding,
              Activation activation);
    ggml_tensor* forward(nn::Context& context, ggml_tensor* input, ggml_backend_t backend);
private:
    ConvAct& depthwise_;
    ConvAct& pointwise_;
};

class Model final : public nn::Module<Model> {
public:
    Model();
    ggml_tensor* forward(nn::Context& context, ggml_tensor* input, ggml_backend_t backend);
private:
    ConvAct& stem_;
    nn::ModuleList<Separable>& encoder_;
    nn::ModuleList<Separable>& decoder_;
    ConvAct& output_;
};

} // namespace visual_perception::depth::fastdepth
