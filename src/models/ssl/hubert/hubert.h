#pragma once

#include "models/debug.h"
#include "models/model_profile.h"
#include "nn/nn.h"

struct gguf_context;

namespace gpt_sovits {

// CNHuBERT Graph Builder
struct HubertModel : public nn::Module {
    nn::Parameter pos_conv_weight = nn::Parameter::required();
    nn::Parameter pos_conv_bias = nn::Parameter::required();
    
    nn::TransformerEncoder encoder;
    nn::InstanceNorm ln0;
    nn::LayerNorm proj_ln;
    nn::Linear proj_dense;
    nn::LayerNorm encoder_ln;
    nn::Conv1d conv_layers[7];

    HubertModel() : encoder(12, 12, 64, nn::ActivationType::GELU_ERF, 1e-5f, false),
                    ln0(nullptr, nullptr, 1e-5f),
                    proj_ln(nullptr, nullptr, 1e-5f),
                    proj_dense(nullptr, nullptr),
                    encoder_ln(nullptr, nullptr, 1e-5f) {
        register_parameter("pos_conv_weight", pos_conv_weight);
        register_parameter("pos_conv_bias", pos_conv_bias);
        register_module("encoder", &encoder);
        register_module("ln0", &ln0);
        register_module("proj_ln", &proj_ln);
        register_module("proj_dense", &proj_dense);
        register_module("encoder_ln", &encoder_ln);

        conv_layers[0].stride = 5;
        conv_layers[1].stride = 2;
        conv_layers[2].stride = 2;
        conv_layers[3].stride = 2;
        conv_layers[4].stride = 2;
        conv_layers[5].stride = 2;
        conv_layers[6].stride = 2;

        for (int i = 0; i < 7; ++i) {
            register_module("conv_layers." + std::to_string(i), &conv_layers[i]);
        }
    }
    
    bool load(const std::string& path, ggml_backend_t backend);
    struct ggml_tensor* forward(struct ggml_context* ctx_graph, const float* audio_data, int audio_len, ggml_backend_t backend);

private:
    struct ggml_tensor* build_feature_extractor(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend);
    struct ggml_tensor* build_feature_projection(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend);
    struct ggml_tensor* build_position_embeddings(struct ggml_context* ctx, struct ggml_tensor* x_proj, ggml_backend_t backend);
};

} // namespace gpt_sovits
