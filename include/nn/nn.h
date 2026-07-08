#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ops/ops.h"
#include <cmath>
#include <vector>
#include <string>
#include <unordered_map>

namespace nn {

class Module;

// Universal bind template function
template <typename TModel>
void bind(
    TModel& model,
    nn::Module& root,
    const std::unordered_map<std::string, std::string>& name_map,
    const std::string& current_path = ""
);

// Simple header-only flat JSON parser for flat string-to-string maps
inline std::unordered_map<std::string, std::string> parse_flat_json(const std::string& json) {
    std::unordered_map<std::string, std::string> res;
    size_t i = 0;
    while (i < json.size()) {
        size_t key_start = json.find('"', i);
        if (key_start == std::string::npos) break;
        size_t key_end = json.find('"', key_start + 1);
        if (key_end == std::string::npos) break;
        std::string key = json.substr(key_start + 1, key_end - key_start - 1);

        size_t colon = json.find(':', key_end + 1);
        if (colon == std::string::npos) break;

        size_t val_start = json.find('"', colon + 1);
        if (val_start == std::string::npos) break;
        size_t val_end = json.find('"', val_start + 1);
        if (val_end == std::string::npos) break;
        std::string val = json.substr(val_start + 1, val_end - val_start - 1);

        res[key] = val;
        i = val_end + 1;
    }
    return res;
}

// Base class for all neural network modules with parameter binding support
class Module {
protected:
    std::vector<std::pair<std::string, Module*>> children_;
    std::vector<std::pair<std::string, struct ggml_tensor**>> parameters_;

public:
    ggml_backend_t backend = nullptr;

    virtual ~Module() = default;

    void register_module(const std::string& name, Module* child) {
        children_.push_back({name, child});
    }

    void register_parameter(const std::string& name, struct ggml_tensor** param) {
        parameters_.push_back({name, param});
    }

    void to(ggml_backend_t b) {
        backend = b;
        for (auto& child : children_) {
            child.second->to(b);
        }
    }

    template <typename TModel>
    friend void bind(TModel&, nn::Module&, const std::unordered_map<std::string, std::string>&, const std::string&);
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

    Embedding() {
        register_parameter("weight", &weight);
    }
    Embedding(struct ggml_tensor* w) : weight(w) {
        register_parameter("weight", &weight);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* input_ids);
    struct ggml_tensor* operator()(struct ggml_context* ctx, struct ggml_tensor* input_ids) {
        return forward(ctx, input_ids);
    }
};

// 2. Linear (Dense) layer
class Linear : public Module {
public:
    struct ggml_tensor* weight = nullptr; // [in_features, out_features]
    struct ggml_tensor* bias = nullptr;   // [out_features] (optional)

    Linear() {
        register_parameter("weight", &weight);
        register_parameter("bias", &bias);
    }
    Linear(struct ggml_tensor* w, struct ggml_tensor* b = nullptr) : weight(w), bias(b) {
        register_parameter("weight", &weight);
        register_parameter("bias", &bias);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x);
    struct ggml_tensor* operator()(struct ggml_context* ctx, struct ggml_tensor* x) {
        return forward(ctx, x);
    }
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

    Conv1d() {
        register_parameter("weight", &weight);
        register_parameter("bias", &bias);
    }
    Conv1d(struct ggml_tensor* w, struct ggml_tensor* b = nullptr, int stride = 1, int padding = 0, int dilation = 1, int groups = 1)
        : weight(w), bias(b), stride(stride), padding(padding), dilation(dilation), groups(groups) {
        register_parameter("weight", &weight);
        register_parameter("bias", &bias);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend = nullptr);
    struct ggml_tensor* operator()(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend = nullptr) {
        return forward(ctx, x, backend);
    }
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

    ConvTranspose1d() {
        register_parameter("weight", &weight);
        register_parameter("bias", &bias);
    }
    ConvTranspose1d(struct ggml_tensor* w, struct ggml_tensor* b = nullptr, int stride = 1, int padding = 0, int dilation = 1, int groups = 1)
        : weight(w), bias(b), stride(stride), padding(padding), dilation(dilation), groups(groups) {
        register_parameter("weight", &weight);
        register_parameter("bias", &bias);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend = nullptr);
    struct ggml_tensor* operator()(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend = nullptr) {
        return forward(ctx, x, backend);
    }
};

// 5. Layer Normalization
class LayerNorm : public Module {
public:
    struct ggml_tensor* gamma = nullptr; // [channels]
    struct ggml_tensor* beta = nullptr;  // [channels]
    float eps = 1e-5f;

    LayerNorm() {
        register_parameter("weight", &gamma);
        register_parameter("bias", &beta);
    }
    LayerNorm(struct ggml_tensor* gamma, struct ggml_tensor* beta, float eps = 1e-5f)
        : gamma(gamma), beta(beta), eps(eps) {
        register_parameter("weight", &this->gamma);
        register_parameter("bias", &this->beta);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend = nullptr);
    struct ggml_tensor* operator()(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend = nullptr) {
        return forward(ctx, x, backend);
    }
};

// 6. Instance Normalization
class InstanceNorm : public Module {
public:
    struct ggml_tensor* gamma = nullptr; // [channels]
    struct ggml_tensor* beta = nullptr;  // [channels]
    float eps = 1e-5f;

    InstanceNorm() {
        register_parameter("weight", &gamma);
        register_parameter("bias", &beta);
    }
    InstanceNorm(struct ggml_tensor* gamma, struct ggml_tensor* beta, float eps = 1e-5f)
        : gamma(gamma), beta(beta), eps(eps) {
        register_parameter("weight", &this->gamma);
        register_parameter("bias", &this->beta);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend = nullptr);
    struct ggml_tensor* operator()(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend = nullptr) {
        return forward(ctx, x, backend);
    }
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
    struct ggml_tensor* operator()(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* scale, struct ggml_tensor* shift, ggml_backend_t backend = nullptr) {
        return forward(ctx, x, scale, shift, backend);
    }
};

// 6c. AdaLayerNormZero (used in DiT / Flow Matching blocks)
class AdaLayerNormZero : public Module {
public:
    Linear linear;
    LayerNorm norm;
    float eps = 1e-6f;

    AdaLayerNormZero() {
        register_module("linear", &linear);
        register_module("norm", &norm);
    }
    AdaLayerNormZero(struct ggml_tensor* linear_w, struct ggml_tensor* linear_b, float eps = 1e-6f)
        : linear(linear_w, linear_b), norm(nullptr, nullptr, eps), eps(eps) {
        register_module("linear", &linear);
        register_module("norm", &norm);
    }

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
    Output operator()(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* emb, ggml_backend_t backend = nullptr) {
        return forward(ctx, x, emb, backend);
    }
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
    struct ggml_tensor* operator()(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend = nullptr) {
        return forward(ctx, x, backend);
    }
};

// 6e. Parametric ReLU (PReLU)
class PReLU : public Module {
public:
    struct ggml_tensor* weight = nullptr; // [num_parameters]

    PReLU() {
        register_parameter("weight", &weight);
    }
    PReLU(struct ggml_tensor* w) : weight(w) {
        register_parameter("weight", &weight);
    }

    struct ggml_tensor* forward(
        struct ggml_context* ctx,
        struct ggml_tensor* x,
        ggml_backend_t backend = nullptr
    );
    struct ggml_tensor* operator()(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend = nullptr) {
        return forward(ctx, x, backend);
    }
};

// 7. Gated Linear Unit (GLU)
class GLU : public Module {
public:
    GLU() = default;

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend = nullptr);
    struct ggml_tensor* operator()(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend = nullptr) {
        return forward(ctx, x, backend);
    }
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

    FeedForward() {
        register_module("w1", &w1);
        register_module("w2", &w2);
    }
    FeedForward(
        struct ggml_tensor* w1_w, struct ggml_tensor* w1_b,
        struct ggml_tensor* w2_w, struct ggml_tensor* w2_b,
        ActivationType act = ActivationType::GELU
    ) : w1(w1_w, w1_b), w2(w2_w, w2_b), act_type(act) {
        register_module("w1", &w1);
        register_module("w2", &w2);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend = nullptr);
    struct ggml_tensor* operator()(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend = nullptr) {
        return forward(ctx, x, backend);
    }
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

    MultiHeadAttention() {
        register_module("q_proj", &q_proj);
        register_module("k_proj", &k_proj);
        register_module("v_proj", &v_proj);
        register_module("out_proj", &out_proj);
    }
    MultiHeadAttention(
        struct ggml_tensor* qw, struct ggml_tensor* qb,
        struct ggml_tensor* kw, struct ggml_tensor* kb,
        struct ggml_tensor* vw, struct ggml_tensor* vb,
        struct ggml_tensor* ow, struct ggml_tensor* ob,
        int n_heads, int head_dim
    ) : q_proj(qw, qb), k_proj(kw, kb), v_proj(vw, vb), out_proj(ow, ob), n_heads(n_heads), head_dim(head_dim) {
        register_module("q_proj", &q_proj);
        register_module("k_proj", &k_proj);
        register_module("v_proj", &v_proj);
        register_module("out_proj", &out_proj);
    }

    struct ggml_tensor* forward(
        struct ggml_context* ctx,
        struct ggml_tensor* x,
        struct ggml_tensor* mask,
        ggml_backend_t backend = nullptr,
        struct ggml_tensor* pos_tensor = nullptr
    );
    struct ggml_tensor* operator()(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* mask, ggml_backend_t backend = nullptr, struct ggml_tensor* pos_tensor = nullptr) {
        return forward(ctx, x, mask, backend, pos_tensor);
    }
};

// 10. Transformer Encoder Layer (Post-LN or Pre-LN)
class TransformerEncoderLayer : public Module {
public:
    MultiHeadAttention self_attn;
    FeedForward ffn;
    LayerNorm norm1;
    LayerNorm norm2;
    bool pre_ln = false;

    TransformerEncoderLayer() {
        register_module("self_attn", &self_attn);
        register_module("ffn", &ffn);
        register_module("norm1", &norm1);
        register_module("norm2", &norm2);
    }
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
    ) : self_attn(qw, qb, kw, kb, vw, vb, ow, ob, n_heads, head_dim),
        ffn(ffn_w1, ffn_b1, ffn_w2, ffn_b2, act),
        norm1(ln1_w, ln1_b, eps),
        norm2(ln2_w, ln2_b, eps),
        pre_ln(pre_ln) {
        register_module("self_attn", &self_attn);
        register_module("ffn", &ffn);
        register_module("norm1", &norm1);
        register_module("norm2", &norm2);
    }

    struct ggml_tensor* forward(
        struct ggml_context* ctx,
        struct ggml_tensor* x,
        struct ggml_tensor* mask,
        ggml_backend_t backend = nullptr
    );
    struct ggml_tensor* operator()(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* mask, ggml_backend_t backend = nullptr) {
        return forward(ctx, x, mask, backend);
    }
};

// 10b. Diffusion Transformer Block (DiTBlock)
class DiTBlock : public Module {
public:
    AdaLayerNormZero attn_norm;
    MultiHeadAttention attn;
    LayerNorm ff_norm;
    FeedForward ff;

    DiTBlock() {
        register_module("attn_norm", &attn_norm);
        register_module("attn", &attn);
        register_module("ff_norm", &ff_norm);
        register_module("ff", &ff);
    }
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
    ) : attn_norm(attn_ln_w, attn_ln_b, attn_ln_eps),
        attn(qw, qb, kw, kb, vw, vb, ow, ob, n_heads, head_dim),
        ff_norm(ff_ln_gamma, ff_ln_beta, ff_ln_eps),
        ff(ffn_w1, ffn_b1, ffn_w2, ffn_b2, act) {
        register_module("attn_norm", &attn_norm);
        register_module("attn", &attn);
        register_module("ff_norm", &ff_norm);
        register_module("ff", &ff);
    }

    struct ggml_tensor* forward(
        struct ggml_context* ctx,
        struct ggml_tensor* x,
        struct ggml_tensor* t,
        struct ggml_tensor* mask = nullptr,
        ggml_backend_t backend = nullptr,
        struct ggml_tensor* pos_tensor = nullptr
    );
    struct ggml_tensor* operator()(
        struct ggml_context* ctx,
        struct ggml_tensor* x,
        struct ggml_tensor* t,
        struct ggml_tensor* mask = nullptr,
        ggml_backend_t backend = nullptr,
        struct ggml_tensor* pos_tensor = nullptr
    ) {
        return forward(ctx, x, t, mask, backend, pos_tensor);
    }
};

// 11. Multi-Receptive Field (MRF) Residual Block for VITS/BigVGAN
class ResBlock1d : public Module {
public:
    Conv1d convs1[3];
    Conv1d convs2[3];

    ResBlock1d() {
        for (int i = 0; i < 3; ++i) {
            register_module("convs1." + std::to_string(i), &convs1[i]);
            register_module("convs2." + std::to_string(i), &convs2[i]);
        }
    }
    ResBlock1d(
        struct ggml_tensor* convs1_w[3], struct ggml_tensor* convs1_b[3],
        struct ggml_tensor* convs2_w[3], struct ggml_tensor* convs2_b[3],
        const std::vector<int>& dilations,
        int kernel_size
    ) : convs1{
            Conv1d(convs1_w[0], convs1_b[0], 1, (kernel_size - 1) * dilations[0] / 2, dilations[0]),
            Conv1d(convs1_w[1], convs1_b[1], 1, (kernel_size - 1) * dilations[1] / 2, dilations[1]),
            Conv1d(convs1_w[2], convs1_b[2], 1, (kernel_size - 1) * dilations[2] / 2, dilations[2])
        },
        convs2{
            Conv1d(convs2_w[0], convs2_b[0], 1, (kernel_size - 1) / 2, 1),
            Conv1d(convs2_w[1], convs2_b[1], 1, (kernel_size - 1) / 2, 1),
            Conv1d(convs2_w[2], convs2_b[2], 1, (kernel_size - 1) / 2, 1)
        } {
        for (int i = 0; i < 3; ++i) {
            register_module("convs1." + std::to_string(i), &convs1[i]);
            register_module("convs2." + std::to_string(i), &convs2[i]);
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend = nullptr);
    struct ggml_tensor* operator()(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend = nullptr) {
        return forward(ctx, x, backend);
    }
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

    KVHeadAttention() {
        register_module("q_proj", &q_proj);
        register_module("k_proj", &k_proj);
        register_module("v_proj", &v_proj);
        register_module("out_proj", &out_proj);
    }
    KVHeadAttention(
        struct ggml_tensor* qw, struct ggml_tensor* qb,
        struct ggml_tensor* kw, struct ggml_tensor* kb,
        struct ggml_tensor* vw, struct ggml_tensor* vb,
        struct ggml_tensor* ow, struct ggml_tensor* ob,
        int n_heads, int head_dim, int layer_idx
    ) : q_proj(qw, qb), k_proj(kw, kb), v_proj(vw, vb), out_proj(ow, ob), n_heads(n_heads), head_dim(head_dim), layer_idx(layer_idx) {
        register_module("q_proj", &q_proj);
        register_module("k_proj", &k_proj);
        register_module("v_proj", &v_proj);
        register_module("out_proj", &out_proj);
    }

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
    struct ggml_tensor* operator()(
        struct ggml_context* ctx,
        struct ggml_tensor* x,
        struct ggml_tensor* kv_k,
        struct ggml_tensor* kv_v,
        int q_len,
        int total_len,
        struct ggml_tensor* mask = nullptr,
        struct ggml_cgraph* cgraph = nullptr,
        ggml_backend_t backend = nullptr
    ) {
        return forward(ctx, x, kv_k, kv_v, q_len, total_len, mask, cgraph, backend);
    }
};

// Universal bind template function definition
template <typename TModel>
inline void bind(
    TModel& model,
    nn::Module& root,
    const std::unordered_map<std::string, std::string>& name_map,
    const std::string& current_path
) {
    for (auto& param : root.parameters_) {
        std::string cpp_path = current_path.empty() ? param.first : (current_path + "." + param.first);
        auto it = name_map.find(cpp_path);
        if (it != name_map.end()) {
            struct ggml_tensor* t = model.get_tensor(it->second);
            if (t) {
                *(param.second) = t;
            }
        }
    }
    for (auto& child : root.children_) {
        std::string child_path = current_path.empty() ? child.first : (current_path + "." + child.first);
        bind(model, *(child.second), name_map, child_path);
    }
}

} // namespace nn
