#include "cfm_modules.h"

#include <cmath>
#include <vector>

namespace gpt_sovits::vits {

TimestepEmbedding::TimestepEmbedding() = default;

ggml_tensor* TimestepEmbedding::forward(
    nn::Context& context,
    float value,
    int frequency_dim,
    ggml_backend_t backend
) {
    ggml_context* ctx = context.native_handle();
    std::vector<float> frequencies(static_cast<size_t>(frequency_dim));
    const int half = frequency_dim / 2;
    const double factor = std::log(10000.0) / (half - 1);
    for (int i = 0; i < half; ++i) {
        const float angle = 1000.0f * value * static_cast<float>(std::exp(-i * factor));
        frequencies[static_cast<size_t>(i)] = std::sin(angle);
        frequencies[static_cast<size_t>(i + half)] = std::cos(angle);
    }
    ggml_tensor* encoded = context.constant<float>(
        "vits.timestep", {frequency_dim}, nn::data::copy(frequencies));
    (void)backend;
    return second.forward(ctx, ggml_silu(ctx, first.forward(ctx, encoded)));
}

ConvNeXtV2Block::ConvNeXtV2Block() {
    depthwise.groups = 512;
    depthwise.padding = 3;
}

ggml_tensor* ConvNeXtV2Block::forward(
    ggml_context* ctx,
    ggml_tensor* x,
    int intermediate_dim,
    ggml_backend_t backend
) {
    ggml_tensor* residual = x;
    ggml_tensor* transposed = ggml_cont(ctx, ggml_transpose(ctx, x));
    ggml_tensor* value = nn::F::conv1d_no_transpose(
        ctx, transposed, depthwise.weight.tensor(), depthwise.bias.local_tensor(),
        1, 3, 1, depthwise.groups, backend);
    if (!value) return nullptr;
    value = ggml_cont(ctx, ggml_transpose(ctx, value));
    value = norm.forward(ctx, value, backend);
    value = pointwise_in.forward(ctx, value);
    value = ggml_gelu(ctx, value);

    ggml_tensor* squared = ggml_sqr(ctx, value);
    ggml_tensor* sums = ggml_sum_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, squared)));
    ggml_tensor* magnitude = ggml_cont(ctx, ggml_transpose(ctx, ggml_sqrt(ctx, sums)));
    ggml_tensor* mean = ggml_scale(ctx, ggml_sum_rows(ctx, magnitude), 1.0f / intermediate_dim);
    ggml_tensor* normalized = ggml_div(ctx, magnitude, nn::F::add_scalar(ctx, mean, 1e-6f));
    ggml_tensor* modulated = ggml_mul(ctx, value, ggml_repeat(ctx, normalized, value));
    modulated = ggml_mul(ctx, modulated, ggml_repeat(ctx, grn_gamma.tensor(), modulated));
    value = ggml_add(ctx, ggml_add(
        ctx, modulated, ggml_repeat(ctx, grn_beta.tensor(), modulated)), value);
    return ggml_add(ctx, residual, pointwise_out.forward(ctx, value));
}

PositionConvolution::PositionConvolution() {
    first.groups = 16;
    first.padding = 15;
    second.groups = 16;
    second.padding = 15;
}

ggml_tensor* PositionConvolution::forward(
    ggml_context* ctx,
    ggml_tensor* x,
    ggml_backend_t backend
) {
    auto apply = [&](nn::Conv1d& convolution, ggml_tensor* input) {
        ggml_tensor* transposed = ggml_cont(ctx, ggml_transpose(ctx, input));
        ggml_tensor* output = nn::F::conv1d_no_transpose(
            ctx, transposed, convolution.weight.tensor(), convolution.bias.local_tensor(),
            1, 15, 1, 16, backend);
        output = ggml_cont(ctx, ggml_transpose(ctx, output));
        return nn::F::mish(ctx, output, backend);
    };
    return apply(second, apply(first, x));
}

AdaLNFinal::AdaLNFinal() = default;

ggml_tensor* AdaLNFinal::forward(
    ggml_context* ctx,
    ggml_tensor* x,
    ggml_tensor* embedding,
    int dimension,
    ggml_backend_t backend
) {
    ggml_tensor* projected = nn::F::linear(
        ctx, ggml_silu(ctx, embedding), linear.weight.tensor(),
        linear.bias.local_tensor(), backend);
    ggml_tensor* scale = ggml_cont(ctx, ggml_view_2d(
        ctx, projected, dimension, 1, projected->nb[1], 0));
    ggml_tensor* shift = ggml_cont(ctx, ggml_view_2d(
        ctx, projected, dimension, 1, projected->nb[1], dimension * sizeof(float)));
    ggml_tensor* normalized = ggml_norm(ctx, x, 1e-6f);
    return ggml_add(ctx, ggml_mul(
        ctx, normalized, ggml_repeat(ctx, nn::F::add_scalar(ctx, scale, 1.0f), normalized)),
        ggml_repeat(ctx, shift, normalized));
}

FlowMatchingEstimator::FlowMatchingEstimator() {
    for (int i = 0; i < 4; ++i) text_blocks.emplace_back();
    for (int i = 0; i < 22; ++i) {
        nn::DiTBlock& block = transformer_blocks.emplace_back();
        block.attn.n_heads = 16;
        block.attn.head_dim = 64;
    }
}

ggml_tensor* FlowMatchingEstimator::embed_delta(
    nn::Context& context, float value, ggml_backend_t backend) {
    return delta_embedding.forward(context, value, 256, backend);
}

ggml_tensor* FlowMatchingEstimator::embed_time(
    nn::Context& context, float value, ggml_backend_t backend) {
    return time_embedding.forward(context, value, 256, backend);
}

ggml_tensor* FlowMatchingEstimator::encode_text(
    nn::Context& context,
    ggml_tensor* text,
    int64_t sequence_length,
    ggml_backend_t backend
) {
    ggml_context* ctx = context.native_handle();
    ggml_tensor* encoded = ggml_add(
        ctx, text, nn::F::sinusoidal_position_embedding(context, sequence_length, 512));
    for (ConvNeXtV2Block& block : text_blocks) {
        encoded = block.forward(ctx, encoded, 1024, backend);
    }
    return encoded;
}

ggml_tensor* FlowMatchingEstimator::velocity(
    nn::Context& context,
    ggml_tensor* state,
    ggml_tensor* prompt,
    ggml_tensor* condition,
    ggml_tensor* text,
    ggml_tensor* positions,
    ggml_backend_t backend
) {
    ggml_context* ctx = context.native_handle();
    ggml_tensor* input = ggml_concat(ctx, ggml_concat(ctx, state, prompt, 0), text, 0);
    ggml_tensor* hidden = input_projection.forward(ctx, input);
    hidden = ggml_add(ctx, hidden, position_convolution.forward(ctx, hidden, backend));
    for (nn::DiTBlock& block : transformer_blocks) {
        hidden = block.forward(ctx, hidden, condition, nullptr, backend, positions);
    }
    hidden = final_norm.forward(ctx, hidden, condition, 1024, backend);
    return output_projection.forward(ctx, hidden);
}

} // namespace gpt_sovits::vits
