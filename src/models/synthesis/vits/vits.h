#pragma once

#include "models/model_profile.h"
#include "models/debug.h"
#include "nn/nn.h"
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

class WN : public nn::Module {
public:
    int hidden_channels = 0;
    int kernel_size = 0;
    int dilation_rate = 0;
    int n_layers = 0;

    std::vector<std::unique_ptr<nn::Conv1d>> in_layers;
    std::vector<std::unique_ptr<nn::Conv1d>> res_skip_layers;

    WN() = default;
    WN(int hidden_channels, int kernel_size, int dilation_rate, int n_layers)
        : hidden_channels(hidden_channels), kernel_size(kernel_size), dilation_rate(dilation_rate), n_layers(n_layers) {
        for (int i = 0; i < n_layers; ++i) {
            int dilation = (int)std::pow((float)dilation_rate, i);
            int padding = (int)((kernel_size * dilation - dilation) / 2);

            auto in_conv = std::make_unique<nn::Conv1d>(nullptr, nullptr, 1, padding, dilation, 1);
            register_module("in_layers." + std::to_string(i), in_conv.get());
            in_layers.push_back(std::move(in_conv));

            auto res_skip_conv = std::make_unique<nn::Conv1d>(nullptr, nullptr, 1, 0, 1, 1);
            register_module("res_skip_layers." + std::to_string(i), res_skip_conv.get());
            res_skip_layers.push_back(std::move(res_skip_conv));
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* x_mask, struct ggml_tensor* g, ggml_backend_t backend = nullptr);
};

class ResidualCouplingLayer : public nn::Module {
public:
    WN wn;
    nn::Conv1d pre;
    nn::Conv1d post;
    nn::Conv1d cond_layer;
    bool reverse = false;

    ResidualCouplingLayer() {
        register_module("enc", &wn);
        register_module("pre", &pre);
        register_module("post", &post);
        register_module("enc.cond_layer", &cond_layer);
    }

    ResidualCouplingLayer(int channels, int hidden_channels, int kernel_size, int dilation_rate, int n_layers, bool reverse)
        : wn(hidden_channels, kernel_size, dilation_rate, n_layers),
          pre(nullptr, nullptr, 1, 0, 1, 1),
          post(nullptr, nullptr, 1, 0, 1, 1),
          cond_layer(nullptr, nullptr, 1, 0, 1, 1),
          reverse(reverse) {
        register_module("enc", &wn);
        register_module("pre", &pre);
        register_module("post", &post);
        register_module("enc.cond_layer", &cond_layer);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* x_mask, struct ggml_tensor* g, ggml_backend_t backend = nullptr);
};

class WNEncoder : public nn::Module {
public:
    WN wn;
    nn::Conv1d pre;
    nn::Conv1d proj;
    nn::Conv1d cond_layer;

    WNEncoder() {
        register_module("enc", &wn);
        register_module("pre", &pre);
        register_module("proj", &proj);
        register_module("enc.cond_layer", &cond_layer);
    }

    WNEncoder(int in_channels, int out_channels, int hidden_channels, int kernel_size, int dilation_rate, int n_layers)
        : wn(hidden_channels, kernel_size, dilation_rate, n_layers),
          pre(nullptr, nullptr, 1, 0, 1, 1),
          proj(nullptr, nullptr, 1, 0, 1, 1),
          cond_layer(nullptr, nullptr, 1, 0, 1, 1) {
        register_module("enc", &wn);
        register_module("pre", &pre);
        register_module("proj", &proj);
        register_module("enc.cond_layer", &cond_layer);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* g, ggml_backend_t backend = nullptr);
};

// SoVITS VITS Generator Graph Builder Base Class
struct VITSModel : public nn::Module {
    nn::ParameterDict artifact_parameters;

    VITSModel() {
        register_module("artifact", artifact_parameters);
    }

    struct ggml_tensor* get_tensor(const std::string& name) const {
        const nn::Parameter* parameter = artifact_parameters.find(name);
        return parameter && parameter->is_bound() ? parameter->tensor() : nullptr;
    }

    int64_t get_logical_tensor_dim(const std::string& name, int dim) const {
        const nn::Parameter* parameter = artifact_parameters.find(name);
        if (parameter && dim >= 0 && static_cast<size_t>(dim) < parameter->logical_shape().size()) {
            return parameter->logical_shape()[static_cast<size_t>(dim)];
        }
        return 0;
    }

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

    static std::unique_ptr<VITSModel> create(const std::string& path);

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
        struct ggml_context* ctx_graph,
        struct ggml_tensor* mel_spec,
        struct ggml_tensor* sv_emb,
        ggml_backend_t backend
    );

    void refresh_profile();

    struct GeneratorWeights {
        struct ggml_tensor* conv_pre_w = nullptr;
        struct ggml_tensor* conv_pre_b = nullptr;
        struct ggml_tensor* cond_w = nullptr;
        struct ggml_tensor* cond_b = nullptr;
        struct ggml_tensor* conv_post_w = nullptr;
        struct ggml_tensor* conv_post_b = nullptr;
        std::vector<struct ggml_tensor*> ups_w;
        std::vector<struct ggml_tensor*> ups_b;
        
        struct ResBlockWeights {
            struct ggml_tensor* c1_w = nullptr;
            struct ggml_tensor* c1_b = nullptr;
            struct ggml_tensor* c2_w = nullptr;
            struct ggml_tensor* c2_b = nullptr;
        };
        ResBlockWeights resblocks[18][3];
    } gen_weights;

    virtual ~VITSModel() = default;
};

// VITS Classic Generator (V1 / V2 / V2Pro)
struct VITSModelClassic : public VITSModel {
    std::vector<std::unique_ptr<ResidualCouplingLayer>> flows;

    VITSModelClassic() {
        for (int i = 0; i < 4; ++i) {
            int flow_index = i * 2;
            auto flow = std::make_unique<ResidualCouplingLayer>(192, 192, 5, 1, 4, true);
            register_module("flow.flows." + std::to_string(flow_index), flow.get());
            flows.push_back(std::move(flow));
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
    WNEncoder wns1;
    nn::Linear input_proj;
    nn::Linear proj_out;
    nn::DiTBlock transformer_blocks[22];

    VITSModelCFM() : wns1(512, 512, 512, 5, 1, 8),
                     input_proj(nullptr, nullptr),
                     proj_out(nullptr, nullptr) {
        register_module("wns1", &wns1);
        register_module("input_proj", &input_proj);
        register_module("proj_out", &proj_out);
        for (int i = 0; i < 22; ++i) {
            register_module("transformer_blocks." + std::to_string(i), &transformer_blocks[i]);
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

private:
    struct ggml_tensor* precompute_text_embeddings(nn::Context& context, struct ggml_tensor* cond_text, int T_mel, ggml_backend_t backend);
    FlowMatchingInputs build_flow_matching_inputs(nn::Context& context, int T_mel, int prompt_len, int prompt_start, VITSRunState& state);
    struct ggml_tensor* run_ode_loop(nn::Context& context, const FlowMatchingInputs& inputs, struct ggml_tensor* text_embed, ggml_backend_t backend);
};

// Shared Helper functions declared for use by classic and CFM subclasses
struct ggml_tensor* build_encoder(struct ggml_context* ctx, struct ggml_tensor* x, VITSModel& model, const std::string& base_prefix, int n_layers, int n_head, int d_k, int T, ggml_backend_t backend);
struct ggml_tensor* build_mrte(struct ggml_context* ctx, struct ggml_tensor* y, struct ggml_tensor* text, struct ggml_tensor* ge, VITSModel& model, ggml_backend_t backend);
struct ggml_tensor* build_vits_generator(struct ggml_context* ctx_graph, struct ggml_tensor* latent, struct ggml_tensor* speaker_embedding, VITSModel& model, ggml_backend_t backend);
struct ggml_tensor* build_vits_generator_cfm(struct ggml_context* ctx_graph, struct ggml_tensor* latent, struct ggml_tensor* speaker_embedding, VITSModel& model, ggml_backend_t backend);
struct ggml_tensor* interp_nearest_fractional(
    nn::Context& context,
    struct ggml_tensor* x,
    int64_t target_len,
    double scale_factor = 0.0);
struct ggml_tensor* interp_linear_fractional(nn::Context& context, struct ggml_tensor* x, int64_t target_len);

} // namespace gpt_sovits
