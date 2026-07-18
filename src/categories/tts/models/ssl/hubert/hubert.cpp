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

    nn::io::LoadResult loaded = nn::io::load_into(*this, *source, backend);
    if (!loaded) {
        std::cerr << "[HuBERT] " << loaded.error << "\n";
        return false;
    }
    this->to(backend);
    return true;
}

struct ggml_tensor* HubertFeatureExtractor::forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend) {
    // Layer 0: Conv1D (kernel=10, stride=5, no-padding)
    x = layers[0].forward(ctx, x, backend);
    
    // Layer 0 GroupNorm (groups=512, channels=512) -> represented as nn::InstanceNorm
    x = ggml_cont(ctx, ggml_transpose(ctx, x));
    x = first_norm.forward(ctx, x, backend);
    x = ggml_gelu_erf(ctx, x);
    
    // Transpose back to [512, seq_len_0] for subsequent nn::Conv1d layers
    x = ggml_cont(ctx, ggml_transpose(ctx, x));
    
    // Layer 1 to 6: Conv1D + GELU
    for (int i = 1; i < 7; ++i) {
        x = layers[i].forward(ctx, x, backend);
        x = ggml_gelu_erf(ctx, x);
    }
    return x;
}

struct ggml_tensor* HubertFeatureProjection::forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend) {
    struct ggml_tensor* x_proj = norm(ctx, x, backend);
    x_proj = projection(ctx, x_proj);
    return x_proj;
}

struct ggml_tensor* HubertPositionEncoder::forward(struct ggml_context* ctx, struct ggml_tensor* x_proj, ggml_backend_t backend) {
    int seq_len = (int)x_proj->ne[1];
    struct ggml_tensor* x_pos_input_2d = ggml_cont(ctx, ggml_transpose(ctx, x_proj));
    struct ggml_tensor* x_pos_input = ggml_reshape_3d(ctx, x_pos_input_2d, seq_len, 768, 1);
    struct ggml_tensor* pos_conv_w_tensor = weight.tensor();
    
    struct ggml_tensor* pos_emb = nn::F::conv1d_no_transpose(
        ctx, x_pos_input, pos_conv_w_tensor, nullptr, 1, 64, 1, 16, backend);
    if (!pos_emb) return nullptr;
    pos_emb = ggml_view_3d(
        ctx, pos_emb, seq_len, 768, 1, pos_emb->nb[1], pos_emb->nb[2], 0);
    pos_emb = ggml_cont(ctx, pos_emb);

    pos_emb = ggml_reshape_2d(ctx, pos_emb, seq_len, 768);
    pos_emb = ggml_permute(ctx, pos_emb, 1, 0, 2, 3);
    pos_emb = ggml_cont(ctx, pos_emb);
    struct ggml_tensor* pos_conv_b_reshaped = ggml_reshape_2d(ctx, bias.tensor(), 768, 1);
    pos_emb = ggml_add(ctx, pos_emb, pos_conv_b_reshaped);
    pos_emb = ggml_cont(ctx, pos_emb);
    pos_emb = ggml_gelu_erf(ctx, pos_emb);
    return pos_emb;
}

struct ggml_tensor* HubertModel::forward(
    nn::Context& context, struct ggml_tensor* audio, ggml_backend_t backend) {
    struct ggml_context* ctx = context.native_handle();
    struct ggml_tensor* x = feature_extractor.forward(ctx, audio, backend);
    if (!x) return nullptr;
    struct ggml_tensor* projected = feature_projection.forward(ctx, x, backend);
    if (!projected) return nullptr;
    struct ggml_tensor* position = position_encoder.forward(ctx, projected, backend);
    if (!position) return nullptr;
    struct ggml_tensor* hidden = ggml_cont(ctx, ggml_add(ctx, projected, position));
    hidden = encoder_ln(ctx, hidden, backend);
    return encoder(ctx, hidden, nullptr, backend);
}

struct ggml_tensor* HubertRunner::forward(struct ggml_context* ctx_graph, const float* audio_data, int audio_len) {
    nn::Context output_context = nn::Context::borrow(ctx_graph);
    if (audio_len == 0) {
        return output_context.empty<float>("hubert.output", {768, 0});
    }

    nn::Context execution_context(512 * 1024 * 1024, true);
    struct ggml_context* ctx_hubert = execution_context.native_handle();

    // Create input audio tensor in ctx_hubert as 2D: [in_channels, seq_len] = [1, audio_len]
    struct ggml_tensor* input_audio_tensor = execution_context.input<float>(
        "hubert.audio", {1, audio_len}, nn::data::borrow(audio_data, static_cast<size_t>(audio_len)));
    
    struct ggml_tensor* hidden_states = model_.forward(execution_context, input_audio_tensor, backend_);
    if (!hidden_states) {
        std::cerr << "[CNHuBERT] Positional convolution is unsupported by the selected backend.\n";
        return nullptr;
    }
    int seq_len = (int)hidden_states->ne[1];
    
    // Allocate all tensors in ctx_hubert on the backend
    ggml_backend_buffer_t hubert_buffer = ggml_backend_alloc_ctx_tensors(ctx_hubert, backend_);
    if (!hubert_buffer) {
        std::cerr << "[CNHuBERT] Error: Failed to allocate GPU backend buffer for CNHuBERT context!\n";
        return nullptr;
    }

    execution_context.materialize();
    
    // Build and compute the graph on the backend
    struct ggml_cgraph* gf = execution_context.build(hidden_states, 32768);
    const enum ggml_status status = ggml_ops_ext::ops_backend_graph_compute(backend_, gf);
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
