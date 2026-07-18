#pragma once

#include "nn/core/context.h"
#include "nn/layers/linear.h"

namespace nn {

class KVCache;

class MultiHeadAttention : public Module<MultiHeadAttention> {
public:
    Linear& q_proj = submodule<Linear>("q_proj");
    Linear& k_proj = submodule<Linear>("k_proj");
    Linear& v_proj = submodule<Linear>("v_proj");
    Linear& out_proj = submodule<Linear>("out_proj");
    int n_heads = 1;
    int head_dim = 64;

    MultiHeadAttention() = default;
    MultiHeadAttention(int heads, int dimension) : n_heads(heads), head_dim(dimension) {}

    ggml_tensor* forward(
        ggml_context* ctx,
        ggml_tensor* input,
        ggml_tensor* mask,
        ggml_backend_t backend = nullptr,
        ggml_tensor* position = nullptr);
};

class KVHeadAttention : public Module<KVHeadAttention> {
public:
    Linear& q_proj = submodule<Linear>("q_proj");
    Linear& k_proj = submodule<Linear>("k_proj");
    Linear& v_proj = submodule<Linear>("v_proj");
    Linear& out_proj = submodule<Linear>("out_proj");
    int n_heads = 1;
    int head_dim = 64;
    int layer_idx = 0;

    ggml_tensor* prefill(
        Context& context,
        ggml_tensor* input,
        KVCache& cache,
        ggml_tensor* mask = nullptr,
        ggml_cgraph* graph = nullptr,
        ggml_backend_t backend = nullptr);

    ggml_tensor* decode(
        Context& context,
        ggml_tensor* input,
        KVCache& cache,
        ggml_tensor* position,
        ggml_tensor* valid_length,
        ggml_backend_t backend = nullptr);
};

} // namespace nn
