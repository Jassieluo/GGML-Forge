#include "providers/gpt_sovits/models/vits/generator.h"

#include <string>

namespace gpt_sovits::vits {

ConvParameter::ConvParameter(bool weight_required, bool transpose)
    : weight(parameter("weight", weight_required
          ? nn::Parameter::required(
                std::nullopt, transpose ? nn::Parameter::Usage::conv_transpose1d_weight
                                        : nn::Parameter::Usage::conv1d_weight)
          : nn::Parameter::optional(
                std::nullopt, transpose ? nn::Parameter::Usage::conv_transpose1d_weight
                                        : nn::Parameter::Usage::conv1d_weight))),
      bias(parameter("bias", nn::Parameter::optional(
          std::nullopt, nn::Parameter::Usage::bias))) {}

AliasFreeActivation::AliasFreeActivation() = default;

ggml_tensor* AliasFreeActivation::forward(
    ggml_context* ctx, ggml_tensor* x, ggml_backend_t backend) {
    return nn::F::alias_free_activation1d(
        ctx, x, up_filter.tensor(), down_filter.tensor(), alpha.tensor(), beta.tensor(), backend);
}

GeneratorResidualBlock::GeneratorResidualBlock(int kernel, bool alias_free)
    : kernel_(kernel), alias_free_(alias_free) {
    for (int i = 0; i < 3; ++i) {
        conv1_.emplace_back();
        conv2_.emplace_back();
    }
    if (alias_free_) {
        for (int i = 0; i < 6; ++i) {
            activations_.emplace_back();
        }
    }
}

ggml_tensor* GeneratorResidualBlock::forward(
    ggml_context* ctx, ggml_tensor* x, ggml_backend_t backend) {
    static constexpr int dilations[3] = {1, 3, 5};
    ggml_tensor* current = x;
    for (int i = 0; i < 3; ++i) {
        ggml_tensor* value = alias_free_
            ? activations_[2 * i].forward(ctx, current, backend)
            : ggml_leaky_relu(ctx, current, 0.1f, false);
        value = nn::F::conv1d_no_transpose(
            ctx, value, conv1_[i].weight.tensor(), conv1_[i].bias.local_tensor(),
            1, (kernel_ - 1) * dilations[i] / 2, dilations[i], 1, backend);
        value = alias_free_
            ? activations_[2 * i + 1].forward(ctx, value, backend)
            : ggml_leaky_relu(ctx, value, 0.1f, false);
        value = nn::F::conv1d_no_transpose(
            ctx, value, conv2_[i].weight.tensor(), conv2_[i].bias.local_tensor(),
            1, (kernel_ - 1) / 2, 1, 1, backend);
        current = ggml_add(ctx, current, value);
    }
    return current;
}

Generator::Generator(const ModelProfile& profile) : profile_(profile) {
    const bool alias_free = profile_.vocoder_architecture == "bigvgan-v2";
    const int kernels[3] = {3, 7, 11};
    for (size_t stage = 0; stage < profile_.upsample_rates.size(); ++stage) {
        upsample_.emplace_back(true, true);
        for (int branch = 0; branch < 3; ++branch) {
            residuals_.emplace_back(kernels[branch], alias_free);
        }
    }
    if (alias_free) {
        post_activation_ = &submodule<AliasFreeActivation>("post_activation");
    }
}

ggml_tensor* Generator::forward(
    nn::Context& context, ggml_tensor* latent, ggml_tensor* speaker,
    ggml_backend_t backend) {
    ggml_context* ctx = context.native_handle();
    ggml_tensor* x = ggml_cont(ctx, ggml_transpose(ctx, latent));
    x = nn::F::conv1d_no_transpose(
        ctx, x, pre.weight.tensor(), pre.bias.local_tensor(), 1, 3, 1, 1, backend);
    if (speaker && condition.weight.is_bound()) {
        ggml_tensor* conditioning = ggml_cont(ctx, ggml_transpose(ctx, speaker));
        conditioning = nn::F::conv1d_no_transpose(
            ctx, conditioning, condition.weight.tensor(), condition.bias.local_tensor(),
            1, 0, 1, 1, backend);
        x = ggml_add(ctx, x, conditioning);
    }

    const bool classic = profile_.is_classic;
    const bool alias_free = profile_.vocoder_architecture == "bigvgan-v2";
    for (size_t stage = 0; stage < upsample_.size(); ++stage) {
        if (classic || !alias_free) x = ggml_leaky_relu(ctx, x, 0.1f, false);
        ggml_tensor* weight = upsample_[stage].weight.tensor();
        const int kernel = static_cast<int>(ggml_is_quantized(weight->type) ? weight->ne[1] : weight->ne[0]);
        const int stride = profile_.upsample_rates[stage];
        x = nn::F::conv_transpose1d_no_transpose(
            ctx, x, weight, upsample_[stage].bias.local_tensor(),
            stride, (kernel - stride) / 2, backend);
        ggml_tensor* sum = residuals_[stage * 3].forward(ctx, x, backend);
        sum = ggml_add(ctx, sum, residuals_[stage * 3 + 1].forward(ctx, x, backend));
        sum = ggml_add(ctx, sum, residuals_[stage * 3 + 2].forward(ctx, x, backend));
        x = ggml_scale(ctx, sum, 1.0f / 3.0f);
    }
    x = alias_free ? post_activation_->forward(ctx, x, backend)
                   : ggml_leaky_relu(ctx, x, classic ? 0.01f : 0.1f, false);
    x = nn::F::conv1d_no_transpose(
        ctx, x, post.weight.tensor(), post.bias.local_tensor(), 1, 3, 1, 1, backend);
    return ggml_tanh(ctx, ggml_cont(ctx, ggml_transpose(ctx, x)));
}

} // namespace gpt_sovits::vits
