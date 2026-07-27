#pragma once

#include "nn/nn.h"

namespace visual_perception::segmentation::lraspp {

enum class Activation { none, relu, hardswish };

class ConvAct final : public nn::Module<ConvAct> {
public:
    ConvAct(int kernel, int stride, int padding, int dilation, int groups,
            Activation activation);
    ggml_tensor* forward(nn::Context& context, ggml_tensor* input,
                         ggml_backend_t backend);
private:
    Activation activation_;
    nn::Conv2d& conv_;
};

class SqueezeExcitation final : public nn::Module<SqueezeExcitation> {
public:
    SqueezeExcitation();
    ggml_tensor* forward(nn::Context& context, ggml_tensor* input,
                         ggml_backend_t backend);
private:
    nn::Conv2d& fc1_;
    nn::Conv2d& fc2_;
};

class InvertedResidual final : public nn::Module<InvertedResidual> {
public:
    InvertedResidual(int input_channels, int expanded_channels, int output_channels,
                     int kernel, int stride, int dilation, bool squeeze_excitation,
                     Activation activation);
    ggml_tensor* forward(nn::Context& context, ggml_tensor* input,
                         ggml_backend_t backend);
private:
    bool residual_ = false;
    ConvAct* expand_ = nullptr;
    ConvAct& depthwise_;
    SqueezeExcitation* squeeze_excitation_ = nullptr;
    ConvAct& project_;
};

class Backbone final : public nn::Module<Backbone> {
public:
    Backbone();
    struct Output { ggml_tensor* low; ggml_tensor* high; };
    Output forward(nn::Context& context, ggml_tensor* input,
                   ggml_backend_t backend);
private:
    ConvAct& stem_;
    nn::ModuleList<InvertedResidual>& blocks_;
    ConvAct& final_;
};

class Head final : public nn::Module<Head> {
public:
    Head();
    ggml_tensor* forward(nn::Context& context, ggml_tensor* low,
                         ggml_tensor* high, ggml_backend_t backend);
private:
    ConvAct& cbr_;
    nn::Conv2d& scale_;
    nn::Conv2d& low_classifier_;
    nn::Conv2d& high_classifier_;
};

class Model final : public nn::Module<Model> {
public:
    explicit Model(int class_count = 21);
    ggml_tensor* forward(nn::Context& context, ggml_tensor* input,
                         ggml_backend_t backend);
private:
    int class_count_;
    Backbone& backbone_;
    Head& classifier_;
};

} // namespace visual_perception::segmentation::lraspp
