#pragma once

#include "models/model_profile.h"
#include "nn/nn.h"

#include <array>
#include <memory>
#include <vector>

namespace gpt_sovits::vits {

class ConvParameter final : public nn::Module<ConvParameter> {
public:
    nn::Parameter& weight;
    nn::Parameter& bias;
    explicit ConvParameter(bool weight_required = true, bool transpose = false);
};

class AliasFreeActivation final : public nn::Module<AliasFreeActivation> {
public:
    nn::Parameter& alpha = parameter("alpha", nn::Parameter::required(
        std::nullopt, nn::Parameter::Usage::scalar));
    nn::Parameter& beta = parameter("beta", nn::Parameter::required(
        std::nullopt, nn::Parameter::Usage::scalar));
    nn::Parameter& up_filter = parameter("up_filter", nn::Parameter::required());
    nn::Parameter& down_filter = parameter("down_filter", nn::Parameter::required());

    AliasFreeActivation();
    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* x, ggml_backend_t backend);
};

class GeneratorResidualBlock final : public nn::Module<GeneratorResidualBlock> {
public:
    GeneratorResidualBlock(int kernel, bool alias_free);
    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* x, ggml_backend_t backend);

private:
    int kernel_ = 0;
    bool alias_free_ = false;
    nn::ModuleList<ConvParameter>& conv1_ = submodule<nn::ModuleList<ConvParameter>>("conv1");
    nn::ModuleList<ConvParameter>& conv2_ = submodule<nn::ModuleList<ConvParameter>>("conv2");
    nn::ModuleList<AliasFreeActivation>& activations_ =
        submodule<nn::ModuleList<AliasFreeActivation>>("activations");
};

class Generator final : public nn::Module<Generator> {
public:
    explicit Generator(const ModelProfile& profile);
    ggml_tensor* forward(
        nn::Context& context,
        ggml_tensor* latent,
        ggml_tensor* speaker,
        ggml_backend_t backend);
    const ConvParameter& upsample(size_t index) const { return upsample_[index]; }
    size_t upsample_count() const noexcept { return upsample_.size(); }

    ConvParameter& pre = submodule<ConvParameter>("pre");
    ConvParameter& condition = submodule<ConvParameter>("condition", false);
    ConvParameter& post = submodule<ConvParameter>("post");

private:
    ModelProfile profile_;
    nn::ModuleList<ConvParameter>& upsample_ = submodule<nn::ModuleList<ConvParameter>>("upsample");
    nn::ModuleList<GeneratorResidualBlock>& residuals_ =
        submodule<nn::ModuleList<GeneratorResidualBlock>>("residuals");
    AliasFreeActivation* post_activation_ = nullptr;
};

} // namespace gpt_sovits::vits
