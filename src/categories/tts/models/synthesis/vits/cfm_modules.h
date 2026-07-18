#pragma once

#include "nn/nn.h"

namespace gpt_sovits::vits {

class TimestepEmbedding final : public nn::Module<TimestepEmbedding> {
public:
    nn::Linear& first = submodule<nn::Linear>("first");
    nn::Linear& second = submodule<nn::Linear>("second");

    TimestepEmbedding();
    ggml_tensor* forward(
        nn::Context& context, float value, int frequency_dim,
        ggml_backend_t backend);
};

class ConvNeXtV2Block final : public nn::Module<ConvNeXtV2Block> {
public:
    nn::Conv1d& depthwise = submodule<nn::Conv1d>("depthwise");
    nn::LayerNorm& norm = submodule<nn::LayerNorm>("norm");
    nn::Linear& pointwise_in = submodule<nn::Linear>("pointwise_in");
    nn::Linear& pointwise_out = submodule<nn::Linear>("pointwise_out");
    nn::Parameter& grn_beta = parameter("grn_beta", nn::Parameter::required(
        std::nullopt, nn::Parameter::Usage::norm_affine));
    nn::Parameter& grn_gamma = parameter("grn_gamma", nn::Parameter::required(
        std::nullopt, nn::Parameter::Usage::norm_affine));

    ConvNeXtV2Block();
    ggml_tensor* forward(
        ggml_context* ctx, ggml_tensor* x, int intermediate_dim,
        ggml_backend_t backend);
};

class PositionConvolution final : public nn::Module<PositionConvolution> {
public:
    nn::Conv1d& first = submodule<nn::Conv1d>("first");
    nn::Conv1d& second = submodule<nn::Conv1d>("second");

    PositionConvolution();
    ggml_tensor* forward(
        ggml_context* ctx, ggml_tensor* x, ggml_backend_t backend);
};

class AdaLNFinal final : public nn::Module<AdaLNFinal> {
public:
    nn::Linear& linear = submodule<nn::Linear>("linear");

    AdaLNFinal();
    ggml_tensor* forward(
        ggml_context* ctx, ggml_tensor* x, ggml_tensor* embedding,
        int dimension, ggml_backend_t backend);
};

class FlowMatchingEstimator final : public nn::Module<FlowMatchingEstimator> {
public:
    FlowMatchingEstimator();

    ggml_tensor* embed_delta(
        nn::Context& context, float value, ggml_backend_t backend);
    ggml_tensor* embed_time(
        nn::Context& context, float value, ggml_backend_t backend);
    ggml_tensor* encode_text(
        nn::Context& context, ggml_tensor* text, int64_t sequence_length,
        ggml_backend_t backend);
    ggml_tensor* velocity(
        nn::Context& context,
        ggml_tensor* state,
        ggml_tensor* prompt,
        ggml_tensor* condition,
        ggml_tensor* text,
        ggml_tensor* positions,
        ggml_backend_t backend);
    bool is_bound() const noexcept { return input_projection.weight.is_bound(); }

private:
    nn::Linear& input_projection = submodule<nn::Linear>("input_projection");
    nn::Linear& output_projection = submodule<nn::Linear>("output_projection");
    TimestepEmbedding& delta_embedding = submodule<TimestepEmbedding>("delta_embedding");
    TimestepEmbedding& time_embedding = submodule<TimestepEmbedding>("time_embedding");
    PositionConvolution& position_convolution =
        submodule<PositionConvolution>("position_convolution");
    nn::ModuleList<ConvNeXtV2Block>& text_blocks =
        submodule<nn::ModuleList<ConvNeXtV2Block>>("text_blocks");
    nn::ModuleList<nn::DiTBlock>& transformer_blocks =
        submodule<nn::ModuleList<nn::DiTBlock>>("transformer_blocks");
    AdaLNFinal& final_norm = submodule<AdaLNFinal>("final_norm");
};

} // namespace gpt_sovits::vits
