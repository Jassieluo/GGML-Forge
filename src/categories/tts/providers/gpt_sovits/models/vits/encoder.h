#pragma once

#include "nn/nn.h"

#include <memory>
#include <vector>

namespace gpt_sovits::vits {

// Encoder modules shared by GPT-SoVITS VITS variants.

class LayerNorm final : public nn::Module<LayerNorm> {
public:
    nn::Parameter& gamma = parameter("gamma", nn::Parameter::required(
        std::nullopt, nn::Parameter::Usage::norm_affine));
    nn::Parameter& beta = parameter("beta", nn::Parameter::required(
        std::nullopt, nn::Parameter::Usage::norm_affine));

    LayerNorm();
    ggml_tensor* forward(nn::Context& context, ggml_tensor* x, ggml_backend_t backend);
};

class EncoderLayer final : public nn::Module<EncoderLayer> {
public:
    nn::Conv1d& q_proj = submodule<nn::Conv1d>("q_proj");
    nn::Conv1d& k_proj = submodule<nn::Conv1d>("k_proj");
    nn::Conv1d& v_proj = submodule<nn::Conv1d>("v_proj");
    nn::Conv1d& out_proj = submodule<nn::Conv1d>("out_proj");
    nn::Parameter& relative_key = parameter("relative_key", nn::Parameter::required(
        std::nullopt, nn::Parameter::Usage::relative_position));
    nn::Parameter& relative_value = parameter("relative_value", nn::Parameter::required(
        std::nullopt, nn::Parameter::Usage::relative_position));
    LayerNorm& norm1 = submodule<LayerNorm>("norm1");
    LayerNorm& norm2 = submodule<LayerNorm>("norm2");
    nn::Conv1d& ffn1 = submodule<nn::Conv1d>("ffn1");
    nn::Conv1d& ffn2 = submodule<nn::Conv1d>("ffn2");

    EncoderLayer(int heads, int head_dim);
    ggml_tensor* forward(nn::Context& context, ggml_tensor* x, ggml_backend_t backend);

private:
    int heads_ = 0;
    int head_dim_ = 0;
};

class Encoder final : public nn::Module<Encoder> {
public:
    Encoder(int layers, int heads, int head_dim);
    ggml_tensor* forward(nn::Context& context, ggml_tensor* x, ggml_backend_t backend);

private:
    nn::ModuleList<EncoderLayer>& layers_ = submodule<nn::ModuleList<EncoderLayer>>("layers");
};

class MRTE final : public nn::Module<MRTE> {
public:
    MRTE();
    ggml_tensor* forward(
        nn::Context& context,
        ggml_tensor* semantic,
        ggml_tensor* text,
        ggml_tensor* speaker,
        ggml_backend_t backend);

    nn::Conv1d& semantic_proj = submodule<nn::Conv1d>("semantic_proj");
    nn::Conv1d& text_proj = submodule<nn::Conv1d>("text_proj");
    nn::Conv1d& q_proj = submodule<nn::Conv1d>("q_proj");
    nn::Conv1d& k_proj = submodule<nn::Conv1d>("k_proj");
    nn::Conv1d& v_proj = submodule<nn::Conv1d>("v_proj");
    nn::Conv1d& out_proj = submodule<nn::Conv1d>("out_proj");
    nn::Conv1d& result_proj = submodule<nn::Conv1d>("result_proj");
};

} // namespace gpt_sovits::vits
