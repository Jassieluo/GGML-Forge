#pragma once

#include "nn/nn.h"

namespace gpt_sovits::vits {

class TemporalGLU final : public nn::Module<TemporalGLU> {
public:
    nn::Parameter& weight = parameter("weight", nn::Parameter::required(
        std::nullopt, nn::Parameter::Usage::conv1d_weight));
    nn::Parameter& bias = parameter("bias", nn::Parameter::required(
        std::nullopt, nn::Parameter::Usage::bias));

    TemporalGLU();
    ggml_tensor* forward(nn::Context& context, ggml_tensor* x, ggml_backend_t backend);
};

class ReferenceEncoder final : public nn::Module<ReferenceEncoder> {
public:
    nn::Linear& spectral0 = submodule<nn::Linear>("spectral0");
    nn::Linear& spectral3 = submodule<nn::Linear>("spectral3");
    TemporalGLU& temporal0 = submodule<TemporalGLU>("temporal0");
    TemporalGLU& temporal1 = submodule<TemporalGLU>("temporal1");
    nn::Linear& q_proj = submodule<nn::Linear>("q_proj");
    nn::Linear& k_proj = submodule<nn::Linear>("k_proj");
    nn::Linear& v_proj = submodule<nn::Linear>("v_proj");
    nn::Linear& attention_out = submodule<nn::Linear>("attention_out");
    nn::Linear& output = submodule<nn::Linear>("output");

    ReferenceEncoder();
    ggml_tensor* forward(nn::Context& context, ggml_tensor* mel, ggml_backend_t backend);
};

} // namespace gpt_sovits::vits
