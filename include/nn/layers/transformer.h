#pragma once

#include "nn/core/module_list.h"
#include "nn/layers/attention.h"
#include "nn/layers/feedforward.h"
#include "nn/layers/normalization.h"
#include "nn/runtime/kv_cache.h"

namespace nn {

class TransformerEncoderLayer : public Module<TransformerEncoderLayer> {
public:
    MultiHeadAttention& self_attn = submodule<MultiHeadAttention>("self_attn");
    FeedForward& ffn = submodule<FeedForward>("ffn");
    LayerNorm& norm1 = submodule<LayerNorm>("norm1");
    LayerNorm& norm2 = submodule<LayerNorm>("norm2");
    bool pre_ln = false;

    TransformerEncoderLayer() = default;
    TransformerEncoderLayer(
        int heads, int dimension, ActivationType activation,
        float epsilon = 1e-5f, bool use_pre_ln = false)
        : pre_ln(use_pre_ln) {
        self_attn.n_heads = heads;
        self_attn.head_dim = dimension;
        ffn.act_type = activation;
        norm1.eps = epsilon;
        norm2.eps = epsilon;
    }

    ggml_tensor* forward(
        ggml_context* ctx, ggml_tensor* input, ggml_tensor* mask,
        ggml_backend_t backend = nullptr);
};

class DiTBlock : public Module<DiTBlock> {
public:
    AdaLayerNormZero& attn_norm = submodule<AdaLayerNormZero>("attn_norm");
    MultiHeadAttention& attn = submodule<MultiHeadAttention>("attn");
    FeedForward& ff = submodule<FeedForward>("ff");
    float ff_norm_eps = 1e-5f;

    ggml_tensor* forward(
        ggml_context* ctx, ggml_tensor* input, ggml_tensor* timestep,
        ggml_tensor* mask = nullptr, ggml_backend_t backend = nullptr,
        ggml_tensor* position = nullptr);
};

class TransformerDecoderLayer : public Module<TransformerDecoderLayer> {
public:
    KVHeadAttention& self_attn = submodule<KVHeadAttention>("self_attn");
    LayerNorm& ln1 = submodule<LayerNorm>("ln1");
    LayerNorm& ln2 = submodule<LayerNorm>("ln2");
    FeedForward& ffn = submodule<FeedForward>("ffn");

    ggml_tensor* prefill(
        Context& context, ggml_tensor* input, KVCache& cache,
        ggml_tensor* mask = nullptr, ggml_cgraph* graph = nullptr,
        ggml_backend_t backend = nullptr);
    ggml_tensor* decode(
        Context& context, ggml_tensor* input, KVCache& cache,
        ggml_tensor* position, ggml_tensor* valid_length,
        ggml_backend_t backend = nullptr);
};

class TransformerEncoder : public Module<TransformerEncoder> {
public:
    ModuleList<TransformerEncoderLayer>& layers =
        submodule<ModuleList<TransformerEncoderLayer>>("layers");

    TransformerEncoder() = default;
    TransformerEncoder(
        int count, int heads, int dimension, ActivationType activation,
        float epsilon = 1e-5f, bool pre_ln = false) {
        for (int index = 0; index < count; ++index) {
            layers.emplace_back(heads, dimension, activation, epsilon, pre_ln);
        }
    }

    ggml_tensor* forward(
        ggml_context* ctx, ggml_tensor* input, ggml_tensor* mask,
        ggml_backend_t backend = nullptr);
};

class TransformerDecoder : public Module<TransformerDecoder> {
public:
    ModuleList<TransformerDecoderLayer>& layers =
        submodule<ModuleList<TransformerDecoderLayer>>("layers");

    TransformerDecoder() = default;
    TransformerDecoder(
        int count, int heads, int dimension, ActivationType activation,
        float epsilon = 1e-5f) {
        reset(count, heads, dimension, activation, epsilon);
    }

    void reset(
        int count, int heads, int dimension, ActivationType activation,
        float epsilon = 1e-5f) {
        layers.clear();
        for (int index = 0; index < count; ++index) {
            auto& layer = layers.emplace_back();
            layer.self_attn.n_heads = heads;
            layer.self_attn.head_dim = dimension;
            layer.self_attn.layer_idx = index;
            layer.ln1.eps = epsilon;
            layer.ln2.eps = epsilon;
            layer.ffn.act_type = activation;
        }
    }

    ggml_tensor* prefill(
        Context& context, ggml_tensor* input, KVCache& cache,
        ggml_tensor* mask = nullptr, ggml_cgraph* graph = nullptr,
        ggml_backend_t backend = nullptr);
    ggml_tensor* decode(
        Context& context, ggml_tensor* input, KVCache& cache,
        ggml_tensor* position, ggml_tensor* valid_length,
        ggml_backend_t backend = nullptr);
};

} // namespace nn
