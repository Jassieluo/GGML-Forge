#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ops/ops.h"
#include <cmath>
#include <vector>

namespace nn {

// Base class for all neural network modules
class Module {
public:
    virtual ~Module() = default;
};

// Lightweight wrapper for graph-allocated input tensors (PyTorch-like data binding)
struct Input {
    struct ggml_tensor* tensor = nullptr;
    const void* data_ptr = nullptr;
    size_t size_bytes = 0;

    Input() = default;

    static Input tensor_1d(struct ggml_context* ctx, ggml_type type, int64_t ne0, const void* data, size_t bytes) {
        Input in;
        in.tensor = ggml_new_tensor_1d(ctx, type, ne0);
        in.data_ptr = data;
        in.size_bytes = bytes;
        return in;
    }

    static Input tensor_2d(struct ggml_context* ctx, ggml_type type, int64_t ne0, int64_t ne1, const void* data, size_t bytes) {
        Input in;
        in.tensor = ggml_new_tensor_2d(ctx, type, ne0, ne1);
        in.data_ptr = data;
        in.size_bytes = bytes;
        return in;
    }

    static Input tensor_3d(struct ggml_context* ctx, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, const void* data, size_t bytes) {
        Input in;
        in.tensor = ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
        in.data_ptr = data;
        in.size_bytes = bytes;
        return in;
    }

    void upload() const {
        if (tensor && data_ptr && size_bytes > 0) {
            ggml_backend_tensor_set(tensor, data_ptr, 0, size_bytes);
        }
    }
};

// Lightweight wrapper for pre-allocated static input placeholder buffers
struct Buffer {
    struct ggml_tensor* tensor = nullptr;

    Buffer() = default;
    Buffer(struct ggml_tensor* t) : tensor(t) {}

    // Upload data from CPU to backend
    void set(const void* data, size_t size_bytes) {
        if (tensor) {
            ggml_backend_tensor_set(tensor, data, 0, size_bytes);
        }
    }

    // Get 1D slice view
    struct ggml_tensor* view_1d(struct ggml_context* ctx, int64_t length, size_t offset_elements = 0) {
        if (!tensor) return nullptr;
        size_t element_size = ggml_element_size(tensor);
        return ggml_view_1d(ctx, tensor, length, offset_elements * element_size);
    }

    // Get 2D slice view
    struct ggml_tensor* view_2d(struct ggml_context* ctx, int64_t ne0, int64_t ne1, size_t offset_elements = 0) {
        if (!tensor) return nullptr;
        size_t element_size = ggml_element_size(tensor);
        return ggml_view_2d(ctx, tensor, ne0, ne1, tensor->nb[1], offset_elements * element_size);
    }
};

// 1. Embedding layer
class Embedding : public Module {
public:
    struct ggml_tensor* weight = nullptr; // [embedding_dim, num_embeddings]

    Embedding() = default;
    Embedding(struct ggml_tensor* w);

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* input_ids);
};

// 2. Linear (Dense) layer
class Linear : public Module {
public:
    struct ggml_tensor* weight = nullptr; // [in_features, out_features]
    struct ggml_tensor* bias = nullptr;   // [out_features] (optional)

    Linear() = default;
    Linear(struct ggml_tensor* w, struct ggml_tensor* b = nullptr);

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x);
};

// 3. 1D Convolution
class Conv1d : public Module {
public:
    struct ggml_tensor* weight = nullptr; // [kernel_size, in_channels, out_channels]
    struct ggml_tensor* bias = nullptr;   // [out_channels] (optional)
    int stride = 1;
    int padding = 0;
    int dilation = 1;
    int groups = 1;

    Conv1d() = default;
    Conv1d(struct ggml_tensor* w, struct ggml_tensor* b = nullptr, int stride = 1, int padding = 0, int dilation = 1, int groups = 1);

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend);
};

// 4. 1D Transposed Convolution
class ConvTranspose1d : public Module {
public:
    struct ggml_tensor* weight = nullptr; // [kernel_size, out_channels, in_channels]
    struct ggml_tensor* bias = nullptr;   // [out_channels] (optional)
    int stride = 1;
    int padding = 0;
    int dilation = 1;
    int groups = 1;

    ConvTranspose1d() = default;
    ConvTranspose1d(struct ggml_tensor* w, struct ggml_tensor* b = nullptr, int stride = 1, int padding = 0, int dilation = 1, int groups = 1);

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend);
};

// 5. Layer Normalization
class LayerNorm : public Module {
public:
    struct ggml_tensor* gamma = nullptr; // [channels]
    struct ggml_tensor* beta = nullptr;  // [channels]
    float eps = 1e-5f;

    LayerNorm() = default;
    LayerNorm(struct ggml_tensor* gamma, struct ggml_tensor* beta, float eps = 1e-5f);

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend);
};

// 6. Instance Normalization
class InstanceNorm : public Module {
public:
    struct ggml_tensor* gamma = nullptr; // [channels]
    struct ggml_tensor* beta = nullptr;  // [channels]
    float eps = 1e-5f;

    InstanceNorm() = default;
    InstanceNorm(struct ggml_tensor* gamma, struct ggml_tensor* beta, float eps = 1e-5f);

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend);
};

// 6b. Adaptive Layer Normalization (AdaLN)
class AdaLN : public Module {
public:
    float eps = 1e-5f;

    AdaLN() = default;
    AdaLN(float eps) : eps(eps) {}

    struct ggml_tensor* forward(
        struct ggml_context* ctx,
        struct ggml_tensor* x,
        struct ggml_tensor* scale,
        struct ggml_tensor* shift,
        ggml_backend_t backend = nullptr
    );
};

// 6c. AdaLayerNormZero (used in DiT / Flow Matching blocks)
class AdaLayerNormZero : public Module {
public:
    Linear linear;
    LayerNorm norm;
    float eps = 1e-6f;

    AdaLayerNormZero() = default;
    AdaLayerNormZero(struct ggml_tensor* linear_w, struct ggml_tensor* linear_b, float eps = 1e-6f);

    struct Output {
        struct ggml_tensor* x_modulated = nullptr;
        struct ggml_tensor* gate_msa = nullptr;
        struct ggml_tensor* shift_mlp = nullptr;
        struct ggml_tensor* scale_mlp = nullptr;
        struct ggml_tensor* gate_mlp = nullptr;
    };

    Output forward(
        struct ggml_context* ctx,
        struct ggml_tensor* x,
        struct ggml_tensor* emb,
        ggml_backend_t backend = nullptr
    );
};

// 6d. Snake Activation Layer
class Snake : public Module {
public:
    float alpha = 1.0f;

    Snake() = default;
    Snake(float alpha) : alpha(alpha) {}

    struct ggml_tensor* forward(
        struct ggml_context* ctx,
        struct ggml_tensor* x,
        ggml_backend_t backend = nullptr
    );
};

// 6e. Parametric ReLU (PReLU)
class PReLU : public Module {
public:
    struct ggml_tensor* weight = nullptr; // [num_parameters]

    PReLU() = default;
    PReLU(struct ggml_tensor* w) : weight(w) {}

    struct ggml_tensor* forward(
        struct ggml_context* ctx,
        struct ggml_tensor* x,
        ggml_backend_t backend = nullptr
    );
};

// 7. Gated Linear Unit (GLU)
class GLU : public Module {
public:
    GLU() = default;

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend);
};

// Activation Type enum for FeedForward and Transformer blocks
enum class ActivationType {
    GELU,
    GELU_ERF,
    RELU,
    LEAKY_RELU,
    MISH,
    DOUBLE_SWISH,
    SILU
};

// 8. FeedForward (FFN / MLP) Block
class FeedForward : public Module {
public:
    Linear w1;
    Linear w2;
    ActivationType act_type = ActivationType::GELU;

    FeedForward() = default;
    FeedForward(
        struct ggml_tensor* w1_w, struct ggml_tensor* w1_b,
        struct ggml_tensor* w2_w, struct ggml_tensor* w2_b,
        ActivationType act = ActivationType::GELU
    );

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend);
};

// 9. Multi-Head Self Attention (MHA)
class MultiHeadAttention : public Module {
public:
    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear out_proj;
    int n_heads = 1;
    int head_dim = 64;

    MultiHeadAttention() = default;
    MultiHeadAttention(
        struct ggml_tensor* qw, struct ggml_tensor* qb,
        struct ggml_tensor* kw, struct ggml_tensor* kb,
        struct ggml_tensor* vw, struct ggml_tensor* vb,
        struct ggml_tensor* ow, struct ggml_tensor* ob,
        int n_heads, int head_dim
    );

    struct ggml_tensor* forward(
        struct ggml_context* ctx,
        struct ggml_tensor* x,
        struct ggml_tensor* mask,
        ggml_backend_t backend,
        struct ggml_tensor* pos_tensor = nullptr
    );
};

// 10. Transformer Encoder Layer (Post-LN or Pre-LN)
class TransformerEncoderLayer : public Module {
public:
    MultiHeadAttention self_attn;
    FeedForward ffn;
    LayerNorm norm1;
    LayerNorm norm2;
    bool pre_ln = false;

    TransformerEncoderLayer() = default;
    TransformerEncoderLayer(
        // MHA
        struct ggml_tensor* qw, struct ggml_tensor* qb,
        struct ggml_tensor* kw, struct ggml_tensor* kb,
        struct ggml_tensor* vw, struct ggml_tensor* vb,
        struct ggml_tensor* ow, struct ggml_tensor* ob,
        int n_heads, int head_dim,
        // FFN
        struct ggml_tensor* ffn_w1, struct ggml_tensor* ffn_b1,
        struct ggml_tensor* ffn_w2, struct ggml_tensor* ffn_b2,
        ActivationType act,
        // Norm
        struct ggml_tensor* ln1_w, struct ggml_tensor* ln1_b,
        struct ggml_tensor* ln2_w, struct ggml_tensor* ln2_b,
        float eps = 1e-5f,
        bool pre_ln = false
    );

    struct ggml_tensor* forward(
        struct ggml_context* ctx,
        struct ggml_tensor* x,
        struct ggml_tensor* mask,
        ggml_backend_t backend
    );
};

// 10b. Diffusion Transformer Block (DiTBlock)
class DiTBlock : public Module {
public:
    AdaLayerNormZero attn_norm;
    MultiHeadAttention attn;
    LayerNorm ff_norm;
    FeedForward ff;

    DiTBlock() = default;
    DiTBlock(
        // attn_norm
        struct ggml_tensor* attn_ln_w, struct ggml_tensor* attn_ln_b,
        float attn_ln_eps,
        // attn
        struct ggml_tensor* qw, struct ggml_tensor* qb,
        struct ggml_tensor* kw, struct ggml_tensor* kb,
        struct ggml_tensor* vw, struct ggml_tensor* vb,
        struct ggml_tensor* ow, struct ggml_tensor* ob,
        int n_heads, int head_dim,
        // ff_norm
        struct ggml_tensor* ff_ln_gamma, struct ggml_tensor* ff_ln_beta,
        float ff_ln_eps,
        // ff (FeedForward)
        struct ggml_tensor* ffn_w1, struct ggml_tensor* ffn_b1,
        struct ggml_tensor* ffn_w2, struct ggml_tensor* ffn_b2,
        ActivationType act = ActivationType::GELU
    );

    struct ggml_tensor* forward(
        struct ggml_context* ctx,
        struct ggml_tensor* x,
        struct ggml_tensor* t,
        struct ggml_tensor* mask = nullptr,
        ggml_backend_t backend = nullptr,
        struct ggml_tensor* pos_tensor = nullptr
    );
};

// 11. Multi-Receptive Field (MRF) Residual Block for VITS/BigVGAN
class ResBlock1d : public Module {
public:
    Conv1d convs1[3];
    Conv1d convs2[3];

    ResBlock1d() = default;
    ResBlock1d(
        struct ggml_tensor* convs1_w[3], struct ggml_tensor* convs1_b[3],
        struct ggml_tensor* convs2_w[3], struct ggml_tensor* convs2_b[3],
        const std::vector<int>& dilations,
        int kernel_size
    );

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend);
};

// 12. Direct Activation wrappers
struct Mish {
    static struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend) {
        return ggml_ops_mish(ctx, x, backend);
    }
};

struct DoubleSwish {
    static struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend) {
        return ggml_ops_double_swish(ctx, x, backend);
    }
};

struct GatedTanhSigmoid {
    static struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, int hidden_channels, ggml_backend_t backend) {
        return ggml_ops_gated_tanh_sigmoid(ctx, x, hidden_channels, backend);
    }
};

// 13. Multi-Head Self Attention with KV Cache (KVHeadAttention)
class KVHeadAttention : public Module {
public:
    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear out_proj;
    int n_heads = 1;
    int head_dim = 64;
    int layer_idx = 0;

    KVHeadAttention() = default;
    KVHeadAttention(
        struct ggml_tensor* qw, struct ggml_tensor* qb,
        struct ggml_tensor* kw, struct ggml_tensor* kb,
        struct ggml_tensor* vw, struct ggml_tensor* vb,
        struct ggml_tensor* ow, struct ggml_tensor* ob,
        int n_heads, int head_dim, int layer_idx
    );

    struct ggml_tensor* forward(
        struct ggml_context* ctx,
        struct ggml_tensor* x,
        struct ggml_tensor* kv_k,
        struct ggml_tensor* kv_v,
        int q_len,
        int total_len,
        struct ggml_tensor* mask = nullptr,
        struct ggml_cgraph* cgraph = nullptr,
        ggml_backend_t backend = nullptr
    );
};

} // namespace nn
