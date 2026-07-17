#include "hubert.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ops/ops.h"
#include "nn/nn.h"
#include "nn/io/gguf.h"
#include "nn/io/load.h"
#include "gguf.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <exception>
#include <memory>
#include <optional>
#include <string_view>

namespace gpt_sovits {

bool HubertModel::load(const std::string& path, ggml_backend_t backend) {
    if (!backend) return false;
    std::unique_ptr<nn::io::GGUFSource> source;
    try {
        source = std::make_unique<nn::io::GGUFSource>(path);
    } catch (const std::exception& error) {
        std::cerr << "[HuBERT] " << error.what() << "\n";
        return false;
    }

    const gguf_context* metadata = source->metadata_context();
    const int64_t architecture_key = gguf_find_key(metadata, "general.architecture");
    const int64_t version_key = gguf_find_key(metadata, "gpt_sovits.version");
    if (architecture_key < 0 || gguf_get_kv_type(metadata, architecture_key) != GGUF_TYPE_STRING ||
        std::string(gguf_get_val_str(metadata, architecture_key)) != "gpt_sovits_hubert" ||
        version_key < 0 || gguf_get_kv_type(metadata, version_key) != GGUF_TYPE_STRING ||
        coarse_version_from_string(gguf_get_val_str(metadata, version_key)) == 0) {
        std::cerr << "[HuBERT] Invalid architecture or version metadata.\n";
        return false;
    }

    std::unordered_map<std::string, std::string> name_map;
    name_map["pos_conv_weight"] = "encoder.pos_conv_embed.conv.weight";
    name_map["pos_conv_bias"] = "encoder.pos_conv_embed.conv.bias";
    name_map["ln0.weight"] = "feature_extractor.conv_layers.0.layer_norm.weight";
    name_map["ln0.bias"]   = "feature_extractor.conv_layers.0.layer_norm.bias";
    
    name_map["proj_ln.weight"] = "feature_projection.layer_norm.weight";
    name_map["proj_ln.bias"]   = "feature_projection.layer_norm.bias";
    
    name_map["proj_dense.weight"] = "feature_projection.projection.weight";
    name_map["proj_dense.bias"]   = "feature_projection.projection.bias";
    
    name_map["encoder_ln.weight"] = "encoder.layer_norm.weight";
    name_map["encoder_ln.bias"]   = "encoder.layer_norm.bias";

    for (int i = 0; i < 7; ++i) {
        name_map["conv_layers." + std::to_string(i) + ".weight"] = "feature_extractor.conv_layers." + std::to_string(i) + ".conv.weight";
        name_map["conv_layers." + std::to_string(i) + ".bias"] = "";
    }

    for (int i = 0; i < 12; ++i) {
        std::string cpp_layer = "encoder.layers." + std::to_string(i) + ".";
        std::string gguf_layer = "encoder.layers." + std::to_string(i) + ".";

        name_map[cpp_layer + "self_attn.q_proj.weight"] = gguf_layer + "attention.q_proj.weight";
        name_map[cpp_layer + "self_attn.q_proj.bias"]   = gguf_layer + "attention.q_proj.bias";
        name_map[cpp_layer + "self_attn.k_proj.weight"] = gguf_layer + "attention.k_proj.weight";
        name_map[cpp_layer + "self_attn.k_proj.bias"]   = gguf_layer + "attention.k_proj.bias";
        name_map[cpp_layer + "self_attn.v_proj.weight"] = gguf_layer + "attention.v_proj.weight";
        name_map[cpp_layer + "self_attn.v_proj.bias"]   = gguf_layer + "attention.v_proj.bias";
        name_map[cpp_layer + "self_attn.out_proj.weight"] = gguf_layer + "attention.out_proj.weight";
        name_map[cpp_layer + "self_attn.out_proj.bias"]   = gguf_layer + "attention.out_proj.bias";

        name_map[cpp_layer + "norm1.weight"] = gguf_layer + "layer_norm.weight";
        name_map[cpp_layer + "norm1.bias"]   = gguf_layer + "layer_norm.bias";

        name_map[cpp_layer + "ffn.w1.weight"] = gguf_layer + "feed_forward.intermediate_dense.weight";
        name_map[cpp_layer + "ffn.w1.bias"]   = gguf_layer + "feed_forward.intermediate_dense.bias";
        name_map[cpp_layer + "ffn.w2.weight"] = gguf_layer + "feed_forward.output_dense.weight";
        name_map[cpp_layer + "ffn.w2.bias"]   = gguf_layer + "feed_forward.output_dense.bias";

        name_map[cpp_layer + "norm2.weight"] = gguf_layer + "final_layer_norm.weight";
        name_map[cpp_layer + "norm2.bias"]   = gguf_layer + "final_layer_norm.bias";
    }

    auto mapper = [&](std::string_view parameter_path, const nn::Parameter&) -> std::optional<std::string> {
        auto found = name_map.find(std::string(parameter_path));
        if (found == name_map.end()) return std::string(parameter_path);
        if (found->second.empty()) return std::nullopt;
        return found->second;
    };
    nn::io::LoadResult loaded = nn::io::load_into(*this, *source, backend, mapper);
    if (!loaded) {
        std::cerr << "[HuBERT] " << loaded.error << "\n";
        return false;
    }
    this->to(backend);
    return true;
}

struct ggml_tensor* HubertModel::build_feature_extractor(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend) {
    // Layer 0: Conv1D (kernel=10, stride=5, no-padding)
    x = conv_layers[0].forward(ctx, x, backend);
    
    // Layer 0 GroupNorm (groups=512, channels=512) -> represented as nn::InstanceNorm
    x = ggml_cont(ctx, ggml_transpose(ctx, x));
    x = ln0.forward(ctx, x, backend);
    x = ggml_gelu_erf(ctx, x);
    
    // Transpose back to [512, seq_len_0] for subsequent nn::Conv1d layers
    x = ggml_cont(ctx, ggml_transpose(ctx, x));
    
    // Layer 1 to 6: Conv1D + GELU
    for (int i = 1; i < 7; ++i) {
        x = conv_layers[i].forward(ctx, x, backend);
        x = ggml_gelu_erf(ctx, x);
    }
    return x;
}

struct ggml_tensor* HubertModel::build_feature_projection(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend) {
    struct ggml_tensor* x_proj = proj_ln(ctx, x, backend);
    x_proj = proj_dense(ctx, x_proj);
    return x_proj;
}

struct ggml_tensor* HubertModel::build_position_embeddings(struct ggml_context* ctx, struct ggml_tensor* x_proj, ggml_backend_t backend) {
    int seq_len = (int)x_proj->ne[1];
    struct ggml_tensor* x_pos_input_2d = ggml_cont(ctx, ggml_transpose(ctx, x_proj));
    struct ggml_tensor* x_pos_input = ggml_reshape_3d(ctx, x_pos_input_2d, seq_len, 768, 1);
    struct ggml_tensor* pos_conv_w_tensor = pos_conv_weight.tensor();
    
    struct ggml_tensor* pos_emb = nn::F::conv1d_no_transpose(
        ctx, x_pos_input, pos_conv_w_tensor, nullptr, 1, 64, 1, 16, backend);
    if (!pos_emb) return nullptr;
    pos_emb = ggml_view_3d(
        ctx, pos_emb, seq_len, 768, 1, pos_emb->nb[1], pos_emb->nb[2], 0);
    pos_emb = ggml_cont(ctx, pos_emb);

    pos_emb = ggml_reshape_2d(ctx, pos_emb, seq_len, 768);
    pos_emb = ggml_permute(ctx, pos_emb, 1, 0, 2, 3);
    pos_emb = ggml_cont(ctx, pos_emb);
    struct ggml_tensor* pos_conv_b_reshaped = ggml_reshape_2d(ctx, pos_conv_bias.tensor(), 768, 1);
    pos_emb = ggml_add(ctx, pos_emb, pos_conv_b_reshaped);
    pos_emb = ggml_cont(ctx, pos_emb);
    pos_emb = ggml_gelu_erf(ctx, pos_emb);
    return pos_emb;
}

struct ggml_tensor* HubertModel::forward(struct ggml_context* ctx_graph, const float* audio_data, int audio_len, ggml_backend_t backend) {
    nn::Context output_context = nn::Context::borrow(ctx_graph);
    if (audio_len == 0) {
        return output_context.empty<float>("hubert.output", {768, 0});
    }

    nn::Context execution_context(512 * 1024 * 1024, true);
    struct ggml_context* ctx_hubert = execution_context.native_handle();

    // Create input audio tensor in ctx_hubert as 2D: [in_channels, seq_len] = [1, audio_len]
    struct ggml_tensor* input_audio_tensor = execution_context.input<float>(
        "hubert.audio", {1, audio_len}, nn::data::borrow(audio_data, static_cast<size_t>(audio_len)));
    
    // 1. CNN Feature Extractor
    struct ggml_tensor* x = build_feature_extractor(ctx_hubert, input_audio_tensor, backend);
    if (!x) return nullptr;
    int seq_len = (int)x->ne[1];
    
    // 2. Feature Projection (512 -> 768)
    struct ggml_tensor* x_proj = build_feature_projection(ctx_hubert, x, backend);
    if (!x_proj) return nullptr;
    
    // 3. Positional Convolution Embedding
    struct ggml_tensor* pos_emb = build_position_embeddings(ctx_hubert, x_proj, backend);
    if (!pos_emb) {
        std::cerr << "[CNHuBERT] Positional convolution is unsupported by the selected backend.\n";
        return nullptr;
    }
    
    // 4. Sum inputs + Run Transformer Encoder layers
    struct ggml_tensor* hidden_states = ggml_add(ctx_hubert, x_proj, pos_emb);
    hidden_states = ggml_cont(ctx_hubert, hidden_states);
    hidden_states = encoder_ln(ctx_hubert, hidden_states, backend);
    
    for (int layer = 0; layer < 12; ++layer) {
        hidden_states = encoder.layers[layer]->forward(ctx_hubert, hidden_states, nullptr, backend);
    }
    
    // Allocate all tensors in ctx_hubert on the backend
    ggml_backend_buffer_t hubert_buffer = ggml_backend_alloc_ctx_tensors(ctx_hubert, backend);
    if (!hubert_buffer) {
        std::cerr << "[CNHuBERT] Error: Failed to allocate GPU backend buffer for CNHuBERT context!\n";
        return nullptr;
    }

    execution_context.materialize();
    
    // Build and compute the graph on the backend
    struct ggml_cgraph* gf = execution_context.build(hidden_states, 32768);
    const enum ggml_status status = ggml_ops_ext::ops_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::cerr << "[CNHuBERT] Graph execution failed with status " << status << "\n";
        ggml_backend_buffer_free(hubert_buffer);
        return nullptr;
    }
    
    // Retrieve computed output features back to CPU
    int out_elements = 768 * seq_len;
    std::vector<float> host_out(out_elements);
    ggml_backend_tensor_get(hidden_states, host_out.data(), 0, out_elements * sizeof(float));
    
    // Clean up private context
    ggml_backend_buffer_free(hubert_buffer);

    struct ggml_tensor* out_tensor = output_context.empty<float>("hubert.output", {768, seq_len});
    output_context.write(out_tensor, host_out.data(), host_out.size());
    
    return out_tensor;
}

} // namespace gpt_sovits
