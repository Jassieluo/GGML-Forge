#pragma once

#include "providers/gpt_sovits/models/model_profile.h"
#include "providers/gpt_sovits/models/vits/cfm_modules.h"
#include "providers/gpt_sovits/models/debug.h"
#include "nn/nn.h"
#include "providers/gpt_sovits/models/vits/encoder.h"
#include "providers/gpt_sovits/models/vits/conditioning.h"
#include "providers/gpt_sovits/models/vits/generator.h"
#include "gguf.h"
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

namespace gpt_sovits {

struct VITSModel;

struct VITSRunState {
    std::vector<float> prompt_mel;
    std::vector<float> prompt_features;
};

class WN : public nn::Module<WN> {
public:
    int hidden_channels = 0;
    int kernel_size = 0;
    int dilation_rate = 0;
    int n_layers = 0;

    nn::ModuleList<nn::Conv1d>& in_layers = submodule<nn::ModuleList<nn::Conv1d>>("in_layers");
    nn::ModuleList<nn::Conv1d>& res_skip_layers =
        submodule<nn::ModuleList<nn::Conv1d>>("res_skip_layers");
    nn::Conv1d& cond_layer = submodule<nn::Conv1d>("cond_layer");

    WN() = default;
    WN(int hidden_channels, int kernel_size, int dilation_rate, int n_layers)
        : hidden_channels(hidden_channels), kernel_size(kernel_size), dilation_rate(dilation_rate), n_layers(n_layers) {
        for (int i = 0; i < n_layers; ++i) {
            int dilation = (int)std::pow((float)dilation_rate, i);
            int padding = (int)((kernel_size * dilation - dilation) / 2);

            auto& in_conv = in_layers.emplace_back();
            in_conv.padding = padding;
            in_conv.dilation = dilation;
            res_skip_layers.emplace_back();
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* x_mask, struct ggml_tensor* g, ggml_backend_t backend = nullptr);
};

class ResidualCouplingLayer : public nn::Module<ResidualCouplingLayer> {
public:
    WN& wn;
    nn::Conv1d& pre;
    nn::Conv1d& post;
    bool reverse = false;

    ResidualCouplingLayer()
        : wn(submodule<WN>("enc")), pre(submodule<nn::Conv1d>("pre")),
          post(submodule<nn::Conv1d>("post")) {}

    ResidualCouplingLayer(int channels, int hidden_channels, int kernel_size, int dilation_rate, int n_layers, bool reverse)
        : wn(submodule<WN>("enc", hidden_channels, kernel_size, dilation_rate, n_layers)),
          pre(submodule<nn::Conv1d>("pre")), post(submodule<nn::Conv1d>("post")),
          reverse(reverse) {}

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* x_mask, struct ggml_tensor* g, ggml_backend_t backend = nullptr);
};

class WNEncoder : public nn::Module<WNEncoder> {
public:
    WN& wn;
    nn::Conv1d& pre;
    nn::Conv1d& proj;

    WNEncoder()
        : wn(submodule<WN>("enc")), pre(submodule<nn::Conv1d>("pre")),
          proj(submodule<nn::Conv1d>("proj")) {}

    WNEncoder(int in_channels, int out_channels, int hidden_channels, int kernel_size, int dilation_rate, int n_layers)
        : wn(submodule<WN>("enc", hidden_channels, kernel_size, dilation_rate, n_layers)),
          pre(submodule<nn::Conv1d>("pre")), proj(submodule<nn::Conv1d>("proj")) {}

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* g, ggml_backend_t backend = nullptr);
};

struct SemanticStack : public nn::Module<SemanticStack> {
    nn::Conv1d& ssl_projection = submodule<nn::Conv1d>("ssl_projection");
    nn::Conv1d& output_projection = submodule<nn::Conv1d>("output_projection");
    nn::Embedding& text_embedding = submodule<nn::Embedding>("text_embedding");
    vits::Encoder& ssl_encoder = submodule<vits::Encoder>("ssl_encoder", 3, 2, 96);
    vits::Encoder& text_encoder = submodule<vits::Encoder>("text_encoder", 6, 2, 96);
    vits::MRTE& mrte = submodule<vits::MRTE>("mrte");
    vits::Encoder& output_encoder = submodule<vits::Encoder>("output_encoder", 3, 2, 96);
};

struct QuantizerStack : public nn::Module<QuantizerStack> {
    nn::Embedding& codebook = submodule<nn::Embedding>("codebook");
};

struct SpeakerStack : public nn::Module<SpeakerStack> {
    nn::Linear& projection = submodule<nn::Linear>("projection");
    nn::PReLU& activation = submodule<nn::PReLU>("activation");
    nn::Linear& output_projection = submodule<nn::Linear>("output_projection");
};

// SoVITS VITS Generator Graph Builder Base Class
struct VITSModel : public nn::Module<VITSModel> {
    SemanticStack& semantic = submodule<SemanticStack>("semantic");
    QuantizerStack& quantizer = submodule<QuantizerStack>("quantizer");
    nn::Conv1d& prompt_ssl_projection = submodule<nn::Conv1d>("prompt_ssl_projection");
    vits::ReferenceEncoder& reference_encoder = submodule<vits::ReferenceEncoder>("reference_encoder");
    vits::Generator* generator = nullptr;
    SpeakerStack* speaker = nullptr;

    int cfm_steps = 0;

    ModelProfile profile;
    std::string model_fingerprint;

    bool read_metadata(const struct gguf_context* ctx_gguf, const std::string& exact_version);
    struct EncodeResult {
        struct ggml_tensor* y2 = nullptr;
        struct ggml_tensor* ge = nullptr;
        struct ggml_tensor* ge_512 = nullptr;
        int T_y = 0;
    };

    EncodeResult encode_semantic_base(
        nn::Context& context,
        struct ggml_tensor* phone_ids,
        struct ggml_tensor* prompt_semantics,
        struct ggml_tensor* refer_audio,
        ggml_backend_t backend
    );

    bool load(const std::string& path, ggml_backend_t backend);
    bool configure_modules();

    static std::unique_ptr<VITSModel> create(const std::string& path);
    static std::unique_ptr<VITSModel> create_for_version(const std::string& version);

    virtual struct ggml_tensor* forward_from_latent(
        nn::Context& context,
        struct ggml_tensor* latent,
        struct ggml_tensor* speaker_embedding,
        VITSRunState& state,
        ggml_backend_t backend
    ) = 0;

    virtual struct ggml_tensor* forward(
        nn::Context& context,
        struct ggml_tensor* phone_ids,
        struct ggml_tensor* phone_lengths,
        struct ggml_tensor* word2ph,
        struct ggml_tensor* bert_features,
        struct ggml_tensor* prompt_semantics,
        struct ggml_tensor* refer_audio,
        VITSRunState& state,
        float speed,
        ggml_backend_t backend
    ) = 0;

    struct ggml_tensor* compute_speaker_embedding(
        nn::Context& context,
        struct ggml_tensor* mel_spec,
        struct ggml_tensor* sv_emb,
        ggml_backend_t backend
    );

    void refresh_profile();

    virtual ~VITSModel() = default;
};

// VITS Classic Generator (V1 / V2 / V2Pro)
struct ClassicFlowStack : public nn::Module<ClassicFlowStack> {
    nn::ModuleDict<ResidualCouplingLayer>& flows =
        submodule<nn::ModuleDict<ResidualCouplingLayer>>("flows");
};

struct VITSModelClassic : public VITSModel {
    ClassicFlowStack& flow = submodule<ClassicFlowStack>("flow");

    VITSModelClassic() {
        for (int i = 0; i < 4; ++i) {
            int flow_index = i * 2;
            flow.flows.emplace(std::to_string(flow_index), 192, 192, 5, 1, 4, true);
        }
    }

    struct ggml_tensor* forward_from_latent(
        nn::Context& context,
        struct ggml_tensor* latent,
        struct ggml_tensor* speaker_embedding,
        VITSRunState& state,
        ggml_backend_t backend
    ) override;

    struct ggml_tensor* forward(
        nn::Context& context,
        struct ggml_tensor* phone_ids,
        struct ggml_tensor* phone_lengths,
        struct ggml_tensor* word2ph,
        struct ggml_tensor* bert_features,
        struct ggml_tensor* prompt_semantics,
        struct ggml_tensor* refer_audio,
        VITSRunState& state,
        float speed,
        ggml_backend_t backend
    ) override;
};

struct FlowMatchingInputs {
    struct ggml_tensor* x = nullptr;
    struct ggml_tensor* prompt_x = nullptr;
    struct ggml_tensor* prompt_mask = nullptr;
    struct ggml_tensor* pos_tensor = nullptr;
};

// VITS Flow Matching / DiT Generator (V3 / V4)
struct VITSModelCFM : public VITSModel {
    nn::Conv1d& bridge_projection = submodule<nn::Conv1d>("bridge_projection");
    WNEncoder& wns1 = submodule<WNEncoder>("wns1", 512, 512, 512, 5, 1, 8);
    vits::FlowMatchingEstimator& estimator =
        submodule<vits::FlowMatchingEstimator>("estimator");

    struct ggml_tensor* forward_from_latent(
        nn::Context& context,
        struct ggml_tensor* latent,
        struct ggml_tensor* speaker_embedding,
        VITSRunState& state,
        ggml_backend_t backend
    ) override;

    struct ggml_tensor* condition_features(
        nn::Context& context,
        struct ggml_tensor* semantic_features,
        struct ggml_tensor* speaker_embedding,
        ggml_backend_t backend);

    struct ggml_tensor* forward(
        nn::Context& context,
        struct ggml_tensor* phone_ids,
        struct ggml_tensor* phone_lengths,
        struct ggml_tensor* word2ph,
        struct ggml_tensor* bert_features,
        struct ggml_tensor* prompt_semantics,
        struct ggml_tensor* refer_audio,
        VITSRunState& state,
        float speed,
        ggml_backend_t backend
    ) override;

private:
    FlowMatchingInputs prepare_inputs(nn::Context& context, int T_mel, int prompt_len, int prompt_start, VITSRunState& state);
    struct ggml_tensor* integrate(nn::Context& context, const FlowMatchingInputs& inputs, struct ggml_tensor* text_embed, ggml_backend_t backend);
};

} // namespace gpt_sovits
