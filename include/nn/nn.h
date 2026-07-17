#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "nn/context.h"
#include "nn/executor.h"
#include "nn/module.h"
#include "nn/parameter_dict.h"
#include "ops/ops.h"
#include <cmath>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <utility>

namespace nn {

// Functional operations
namespace functional {
    struct ggml_tensor* linear(
        struct ggml_context* ctx,
        struct ggml_tensor* x,
        struct ggml_tensor* w,
        struct ggml_tensor* b = nullptr,
        ggml_backend_t backend = nullptr
    );
    struct ggml_tensor* layer_norm(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* gamma, struct ggml_tensor* beta, float eps = 1e-5f, ggml_backend_t backend = nullptr);

    // Convolution functional ops
    struct ggml_tensor* conv1d(
        struct ggml_context* ctx,
        struct ggml_tensor* x,      // [in_channels, seq_len]
        struct ggml_tensor* w,      // [kernel_size, in_channels, out_channels]
        struct ggml_tensor* b = nullptr,
        int stride = 1,
        int padding = 0,
        int dilation = 1,
        int groups = 1,
        ggml_backend_t backend = nullptr
    );

    struct ggml_tensor* conv1d_no_transpose(
        struct ggml_context* ctx,
        struct ggml_tensor* x,      // [seq_len, in_channels]
        struct ggml_tensor* w,      // [kernel_size, in_channels, out_channels]
        struct ggml_tensor* b = nullptr,
        int stride = 1,
        int padding = 0,
        int dilation = 1,
        int groups = 1,
        ggml_backend_t backend = nullptr
    );

    struct ggml_tensor* conv_transpose1d(
        struct ggml_context* ctx,
        struct ggml_tensor* x,      // [in_channels, seq_len]
        struct ggml_tensor* w,      // [kernel_size, out_channels, in_channels]
        struct ggml_tensor* b = nullptr,
        int stride = 1,
        int padding = 0,
        int groups = 1,
        ggml_backend_t backend = nullptr
    );

    struct ggml_tensor* conv_transpose1d_no_transpose(
        struct ggml_context* ctx,
        struct ggml_tensor* x,      // [seq_len, in_channels]
        struct ggml_tensor* w,      // [kernel_size, out_channels, in_channels]
        struct ggml_tensor* b = nullptr,
        int stride = 1,
        int padding = 0,
        ggml_backend_t backend = nullptr
    );

    // Interpolation functional ops
    struct ggml_tensor* interpolate_nearest_2x(
        struct ggml_context* ctx,
        struct ggml_tensor* x
    );
}
namespace F = functional;

// Universal bind template function
template <typename TModel>
void bind(
    TModel& model,
    nn::Module& root,
    const std::unordered_map<std::string, std::string>& name_map,
    const std::string& current_path = ""
);

// 1. Embedding layer
class Embedding : public Module {
public:
    Parameter weight = Parameter::required(std::nullopt, Parameter::StoragePolicy::floating); // [embedding_dim, num_embeddings]

    Embedding() {
        register_parameter("weight", weight);
    }
    explicit Embedding(struct ggml_tensor* w) : Embedding() {
        if (w) weight.bind(w);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* input_ids);
    struct ggml_tensor* operator()(struct ggml_context* ctx, struct ggml_tensor* input_ids) {
        return forward(ctx, input_ids);
    }
};

// 2. Linear (Dense) layer
class Linear : public Module {
public:
    Parameter weight = Parameter::required(); // [in_features, out_features]
    Parameter bias = Parameter::optional();   // [out_features]

    Linear() {
        register_parameter("weight", weight);
        register_parameter("bias", bias);
    }
    Linear(struct ggml_tensor* w, struct ggml_tensor* b = nullptr) : Linear() {
        if (w) weight.bind(w);
        if (b) bias.bind(b);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x);
    struct ggml_tensor* operator()(struct ggml_context* ctx, struct ggml_tensor* x) {
        return forward(ctx, x);
    }
};

// 3. 1D Convolution
class Conv1d : public Module {
public:
    Parameter weight = Parameter::required(); // [kernel_size, in_channels, out_channels]
    Parameter bias = Parameter::optional();   // [out_channels]
    int stride = 1;
    int padding = 0;
    int dilation = 1;
    int groups = 1;

    Conv1d() {
        register_parameter("weight", weight);
        register_parameter("bias", bias);
    }
    Conv1d(struct ggml_tensor* w, struct ggml_tensor* b = nullptr, int stride = 1, int padding = 0, int dilation = 1, int groups = 1)
        : Conv1d() {
        this->stride = stride;
        this->padding = padding;
        this->dilation = dilation;
        this->groups = groups;
        if (w) weight.bind(w);
        if (b) bias.bind(b);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend = nullptr);
    struct ggml_tensor* operator()(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend = nullptr) {
        return forward(ctx, x, backend);
    }
};

// 4. 1D Transposed Convolution
class ConvTranspose1d : public Module {
public:
    Parameter weight = Parameter::required(); // [kernel_size, out_channels, in_channels]
    Parameter bias = Parameter::optional();   // [out_channels]
    int stride = 1;
    int padding = 0;
    int dilation = 1;
    int groups = 1;

    ConvTranspose1d() {
        register_parameter("weight", weight);
        register_parameter("bias", bias);
    }
    ConvTranspose1d(struct ggml_tensor* w, struct ggml_tensor* b = nullptr, int stride = 1, int padding = 0, int dilation = 1, int groups = 1)
        : ConvTranspose1d() {
        this->stride = stride;
        this->padding = padding;
        this->dilation = dilation;
        this->groups = groups;
        if (w) weight.bind(w);
        if (b) bias.bind(b);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend = nullptr);
    struct ggml_tensor* operator()(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend = nullptr) {
        return forward(ctx, x, backend);
    }
};

// 5. Layer Normalization
class LayerNorm : public Module {
public:
    Parameter gamma = Parameter::optional(); // [channels]
    Parameter beta = Parameter::optional();  // [channels]
    float eps = 1e-5f;

    LayerNorm() {
        register_parameter("weight", gamma);
        register_parameter("bias", beta);
    }
    LayerNorm(struct ggml_tensor* gamma, struct ggml_tensor* beta, float eps = 1e-5f) : LayerNorm() {
        this->eps = eps;
        if (gamma) this->gamma.bind(gamma);
        if (beta) this->beta.bind(beta);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend = nullptr);
    struct ggml_tensor* operator()(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend = nullptr) {
        return forward(ctx, x, backend);
    }
};

// 6. Instance Normalization
class InstanceNorm : public Module {
public:
    Parameter gamma = Parameter::optional(); // [channels]
    Parameter beta = Parameter::optional();  // [channels]
    float eps = 1e-5f;

    InstanceNorm() {
        register_parameter("weight", gamma);
        register_parameter("bias", beta);
    }
    InstanceNorm(struct ggml_tensor* gamma, struct ggml_tensor* beta, float eps = 1e-5f) : InstanceNorm() {
        this->eps = eps;
        if (gamma) this->gamma.bind(gamma);
        if (beta) this->beta.bind(beta);
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
    }
    AdaLayerNormZero(struct ggml_tensor* linear_w, struct ggml_tensor* linear_b, float eps = 1e-6f)
        : linear(linear_w, linear_b), norm(nullptr, nullptr, eps), eps(eps) {
        register_module("linear", &linear);
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
    Parameter weight = Parameter::optional(); // [num_parameters]

    PReLU() {
        register_parameter("weight", weight);
    }
    explicit PReLU(struct ggml_tensor* w) : PReLU() {
        if (w) weight.bind(w);
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

class TransformerDecoderLayer : public Module {
public:
    KVHeadAttention self_attn;
    LayerNorm ln1;
    LayerNorm ln2;
    FeedForward ffn;

    TransformerDecoderLayer() : ln1(nullptr, nullptr, 1e-5f), ln2(nullptr, nullptr, 1e-5f) {
        register_module("self_attn", &self_attn);
        register_module("ln1", &ln1);
        register_module("ln2", &ln2);
        register_module("ffn", &ffn);
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

// 12. Transformer Encoder Stack
class TransformerEncoder : public Module {
public:
    std::vector<std::unique_ptr<TransformerEncoderLayer>> layers;

    TransformerEncoder() = default;
    TransformerEncoder(int num_layers, int n_heads, int head_dim, ActivationType act, float eps = 1e-5f, bool pre_ln = false) {
        for (int i = 0; i < num_layers; ++i) {
            auto layer = std::make_unique<TransformerEncoderLayer>(
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, n_heads, head_dim,
                nullptr, nullptr, nullptr, nullptr, act,
                nullptr, nullptr, nullptr, nullptr, eps, pre_ln
            );
            register_module("layers." + std::to_string(i), layer.get());
            layers.push_back(std::move(layer));
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* mask, ggml_backend_t backend = nullptr);
    struct ggml_tensor* operator()(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* mask, ggml_backend_t backend = nullptr) {
        return forward(ctx, x, mask, backend);
    }
};

class TransformerDecoder : public Module {
public:
    std::vector<std::unique_ptr<TransformerDecoderLayer>> layers;

    TransformerDecoder() = default;
    TransformerDecoder(int num_layers, int n_heads, int head_dim, ActivationType act, float eps = 1e-5f) {
        reset(num_layers, n_heads, head_dim, act, eps);
    }

    void reset(int num_layers, int n_heads, int head_dim, ActivationType act, float eps = 1e-5f) {
        clear_registered_modules();
        layers.clear();
        for (int i = 0; i < num_layers; ++i) {
            auto layer = std::make_unique<TransformerDecoderLayer>();
            layer->self_attn.n_heads = n_heads;
            layer->self_attn.head_dim = head_dim;
            layer->self_attn.layer_idx = i;
            layer->ln1.eps = eps;
            layer->ln2.eps = eps;
            layer->ffn.act_type = act;
            register_module("layers." + std::to_string(i), layer.get());
            layers.push_back(std::move(layer));
        }
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

struct AttentionCacheConfig {
    ggml_type key_type = GGML_TYPE_F16;
    ggml_type value_type = GGML_TYPE_F16;
};

class AttentionCache {
public:
    struct ggml_tensor* k = nullptr;
    struct ggml_tensor* v = nullptr;

    int head_dim = 0;
    int max_len = 512;
    int n_heads = 0;
    int n_layers = 0;
    AttentionCacheConfig config;

    AttentionCache() = default;

    AttentionCache(const AttentionCache&) = delete;
    AttentionCache& operator=(const AttentionCache&) = delete;

    AttentionCache(AttentionCache&& other) noexcept {
        context_ = std::move(other.context_);
        buffer_ = std::exchange(other.buffer_, nullptr);
        k = other.k;
        v = other.v;
        head_dim = other.head_dim;
        max_len = other.max_len;
        n_heads = other.n_heads;
        n_layers = other.n_layers;
        config = other.config;

        other.k = nullptr;
        other.v = nullptr;
    }

    AttentionCache& operator=(AttentionCache&& other) noexcept {
        if (this != &other) {
            release();
            context_ = std::move(other.context_);
            buffer_ = std::exchange(other.buffer_, nullptr);
            k = other.k;
            v = other.v;
            head_dim = other.head_dim;
            max_len = other.max_len;
            n_heads = other.n_heads;
            n_layers = other.n_layers;
            config = other.config;

            other.k = nullptr;
            other.v = nullptr;
        }
        return *this;
    }

    ~AttentionCache() { release(); }

    bool allocate(
        ggml_backend_t backend,
        int head_dim,
        int max_len,
        int n_heads,
        int n_layers,
        AttentionCacheConfig cache_config = {}) {
        release();
        this->head_dim = head_dim;
        this->max_len = max_len;
        this->n_heads = n_heads;
        this->n_layers = n_layers;
        config = cache_config;

        const auto valid_type = [head_dim](ggml_type type) {
            return (type == GGML_TYPE_F32 || type == GGML_TYPE_F16 ||
                    type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0) &&
                   head_dim % ggml_blck_size(type) == 0;
        };
        if (!valid_type(config.key_type) || !valid_type(config.value_type)) return false;

        try {
            context_ = std::make_unique<Context>(64 * 1024);
            k = context_->empty("attention_cache.key", {head_dim, max_len, n_heads, n_layers}, config.key_type);
            v = context_->empty("attention_cache.value", {head_dim, max_len, n_heads, n_layers}, config.value_type);
        } catch (const std::exception&) {
            release();
            return false;
        }

        buffer_ = ggml_backend_alloc_ctx_tensors(context_->native_handle(), backend);
        if (!buffer_) {
            release();
            return false;
        }

        return true;
    }

private:
    std::unique_ptr<Context> context_;
    ggml_backend_buffer_t buffer_ = nullptr;

    void release() noexcept {
        if (buffer_) ggml_backend_buffer_free(buffer_);
        buffer_ = nullptr;
        context_.reset();
        k = nullptr;
        v = nullptr;
    }
};

} // namespace nn
