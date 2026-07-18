#pragma once

#include "providers/gpt_sovits/models/debug.h"
#include "providers/gpt_sovits/models/model_profile.h"
#include "nn/nn.h"

struct gguf_context;

namespace gpt_sovits {

struct HubertFeatureExtractor : public nn::Module<HubertFeatureExtractor> {
    nn::InstanceNorm& first_norm = submodule<nn::InstanceNorm>("first_norm");
    nn::ModuleList<nn::Conv1d>& layers = submodule<nn::ModuleList<nn::Conv1d>>("layers");

    HubertFeatureExtractor() {
        first_norm.eps = 1e-5f;
        const int strides[7] = {5, 2, 2, 2, 2, 2, 2};
        for (int i = 0; i < 7; ++i) {
            layers.emplace_back().stride = strides[i];
        }
    }
    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* x, ggml_backend_t backend);
};

struct HubertFeatureProjection : public nn::Module<HubertFeatureProjection> {
    nn::LayerNorm& norm = submodule<nn::LayerNorm>("norm");
    nn::Linear& projection = submodule<nn::Linear>("projection");

    HubertFeatureProjection() { norm.eps = 1e-5f; }
    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* x, ggml_backend_t backend);
};

struct HubertPositionEncoder : public nn::Module<HubertPositionEncoder> {
    nn::Parameter& weight = parameter("weight", nn::Parameter::required(
        std::nullopt, nn::Parameter::Usage::conv1d_weight));
    nn::Parameter& bias = parameter("bias", nn::Parameter::required(
        std::nullopt, nn::Parameter::Usage::bias));
    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* x, ggml_backend_t backend);
};

// CNHuBERT Graph Builder
struct HubertModel : public nn::Module<HubertModel> {
    HubertFeatureExtractor& feature_extractor = submodule<HubertFeatureExtractor>("feature_extractor");
    HubertFeatureProjection& feature_projection = submodule<HubertFeatureProjection>("feature_projection");
    HubertPositionEncoder& position_encoder = submodule<HubertPositionEncoder>("position_encoder");
    nn::TransformerEncoder& encoder = submodule<nn::TransformerEncoder>(
        "encoder", 12, 12, 64, nn::ActivationType::GELU_ERF, 1e-5f, false);
    nn::LayerNorm& encoder_ln = submodule<nn::LayerNorm>("encoder_ln");

    HubertModel() { encoder_ln.eps = 1e-5f; }
    
    bool load(const std::string& path, ggml_backend_t backend);
    struct ggml_tensor* forward(
        nn::Context& context,
        struct ggml_tensor* audio,
        ggml_backend_t backend);
};

class HubertRunner {
public:
    HubertRunner(HubertModel& model, ggml_backend_t backend)
        : model_(model), backend_(backend) {}
    ggml_tensor* forward(ggml_context* output_context, const float* audio, int audio_length);

private:
    HubertModel& model_;
    ggml_backend_t backend_ = nullptr;
};

} // namespace gpt_sovits
