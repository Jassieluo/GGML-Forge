#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "nn/context.h"
#include "nn/executor.h"
#include "nn/module.h"
#include "nn/model_schema.h"
#include "nn/module_list.h"
#include "nn/module_dict.h"
#include "nn/parameter_list.h"
#include "ops/ops.h"
#include <cmath>
#include <vector>
#include <string>
#include <memory>
#include <utility>

namespace nn {

class KVCache;

// Functional operations
namespace functional {
    struct ggml_tensor* attention(
        struct ggml_context* ctx,
        struct ggml_tensor* q,
        struct ggml_tensor* k,
        struct ggml_tensor* v,
        struct ggml_tensor* mask = nullptr,
        struct ggml_tensor* weights = nullptr,
        float scale = 1.0f,
        int sliding_window = -1,
        ggml_backend_t backend = nullptr
    );

    struct ggml_tensor* relative_position_keys(
        struct ggml_context* ctx,
        struct ggml_tensor* q,
        struct ggml_tensor* embedding,
        float scale,
        int window,
        ggml_backend_t backend = nullptr
    );

    struct ggml_tensor* relative_position_values(
        struct ggml_context* ctx,
        struct ggml_tensor* weights,
        struct ggml_tensor* embedding,
        struct ggml_tensor* attention_output,
        int window,
        ggml_backend_t backend = nullptr
    );

    struct ggml_tensor* mish(
        struct ggml_context* ctx,
        struct ggml_tensor* x,
        ggml_backend_t backend = nullptr
    );

    struct ggml_tensor* gated_tanh_sigmoid(
        struct ggml_context* ctx,
        struct ggml_tensor* x,
        int channels,
        ggml_backend_t backend = nullptr
    );

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

    struct ggml_tensor* alias_free_activation1d(
        struct ggml_context* ctx,
        struct ggml_tensor* x,
        struct ggml_tensor* up_filter,
        struct ggml_tensor* down_filter,
        struct ggml_tensor* alpha,
        struct ggml_tensor* beta,
        ggml_backend_t backend = nullptr
    );

    // Interpolation functional ops
    struct ggml_tensor* interpolate_nearest_2x(
        struct ggml_context* ctx,
        struct ggml_tensor* x
    );

    struct ggml_tensor* sinusoidal_position_embedding(
        Context& context,
        int64_t sequence_length,
        int64_t dimension,
        float theta = 10000.0f
    );

    struct ggml_tensor* add_scalar(
        struct ggml_context* ctx,
        struct ggml_tensor* input,
        float value
    );

    struct ggml_tensor* interpolate_nearest(
        Context& context,
        struct ggml_tensor* input,
        int64_t target_length,
        double scale_factor = 0.0
    );

    struct ggml_tensor* interpolate_linear(
        Context& context,
        struct ggml_tensor* input,
        int64_t target_length
    );
}
namespace F = functional;

// 1. Embedding layer
class Embedding : public Module<Embedding> {
public:
    Parameter& weight = parameter("weight", Parameter::required(
        std::nullopt, Parameter::Usage::embedding_weight));

    Embedding() = default;
    explicit Embedding(struct ggml_tensor* w) : Embedding() {
        if (w) weight.bind(w);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* input_ids);
};

// 2. Linear (Dense) layer
class Linear : public Module<Linear> {
public:
    Parameter& weight = parameter("weight", Parameter::required(
        std::nullopt, Parameter::Usage::linear_weight));
    Parameter& bias = parameter("bias", Parameter::optional(
        std::nullopt, Parameter::Usage::bias));

    Linear() = default;
    Linear(struct ggml_tensor* w, struct ggml_tensor* b = nullptr) : Linear() {
        if (w) weight.bind(w);
        if (b) bias.bind(b);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x);
};

// 3. 1D Convolution
class Conv1d : public Module<Conv1d> {
public:
    Parameter& weight = parameter("weight", Parameter::required(
        std::nullopt, Parameter::Usage::conv1d_weight));
    Parameter& bias = parameter("bias", Parameter::optional(
        std::nullopt, Parameter::Usage::bias));
    int stride = 1;
    int padding = 0;
    int dilation = 1;
    int groups = 1;

    Conv1d() = default;
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
};

// 4. 1D Transposed Convolution
class ConvTranspose1d : public Module<ConvTranspose1d> {
public:
    Parameter& weight = parameter("weight", Parameter::required(
        std::nullopt, Parameter::Usage::conv_transpose1d_weight));
    Parameter& bias = parameter("bias", Parameter::optional(
        std::nullopt, Parameter::Usage::bias));
    int stride = 1;
    int padding = 0;
    int dilation = 1;
    int groups = 1;

    ConvTranspose1d() = default;
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
};

// 5. Layer Normalization
class LayerNorm : public Module<LayerNorm> {
public:
    Parameter& gamma = parameter("weight", Parameter::optional(
        std::nullopt, Parameter::Usage::norm_affine));
    Parameter& beta = parameter("bias", Parameter::optional(
        std::nullopt, Parameter::Usage::norm_affine));
    float eps = 1e-5f;

    LayerNorm() = default;
    LayerNorm(struct ggml_tensor* gamma, struct ggml_tensor* beta, float eps = 1e-5f) : LayerNorm() {
        this->eps = eps;
        if (gamma) this->gamma.bind(gamma);
        if (beta) this->beta.bind(beta);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend = nullptr);
};

// 6. Instance Normalization
class InstanceNorm : public Module<InstanceNorm> {
public:
    Parameter& gamma = parameter("weight", Parameter::optional(
        std::nullopt, Parameter::Usage::norm_affine));
    Parameter& beta = parameter("bias", Parameter::optional(
        std::nullopt, Parameter::Usage::norm_affine));
    float eps = 1e-5f;

    InstanceNorm() = default;
    InstanceNorm(struct ggml_tensor* gamma, struct ggml_tensor* beta, float eps = 1e-5f) : InstanceNorm() {
        this->eps = eps;
        if (gamma) this->gamma.bind(gamma);
        if (beta) this->beta.bind(beta);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend = nullptr);
};

// 6b. Adaptive Layer Normalization (AdaLN)
class AdaLN : public Module<AdaLN> {
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
class AdaLayerNormZero : public Module<AdaLayerNormZero> {
public:
    Linear& linear = submodule<Linear>("linear");
    LayerNorm& norm = submodule<LayerNorm>("norm");
    float eps = 1e-6f;

    AdaLayerNormZero() = default;
    AdaLayerNormZero(struct ggml_tensor* linear_w, struct ggml_tensor* linear_b, float eps = 1e-6f)
        : eps(eps) {
        if (linear_w) linear.weight.bind(linear_w);
        if (linear_b) linear.bias.bind(linear_b);
        norm.eps = eps;
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
};

// 6d. Snake Activation Layer
class Snake : public Module<Snake> {
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
class PReLU : public Module<PReLU> {
public:
    Parameter& weight = parameter("weight", Parameter::optional(
        std::nullopt, Parameter::Usage::scalar));

    PReLU() = default;
    explicit PReLU(struct ggml_tensor* w) : PReLU() {
        if (w) weight.bind(w);
    }

    struct ggml_tensor* forward(
        struct ggml_context* ctx,
        struct ggml_tensor* x,
        ggml_backend_t backend = nullptr
    );
};

// 7. Gated Linear Unit (GLU)
class GLU : public Module<GLU> {
public:
    GLU() = default;

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend = nullptr);
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
class FeedForward : public Module<FeedForward> {
public:
    Linear& w1 = submodule<Linear>("w1");
    Linear& w2 = submodule<Linear>("w2");
    ActivationType act_type = ActivationType::GELU;

    FeedForward() = default;
    FeedForward(
        struct ggml_tensor* w1_w, struct ggml_tensor* w1_b,
        struct ggml_tensor* w2_w, struct ggml_tensor* w2_b,
        ActivationType act = ActivationType::GELU
    ) : act_type(act) {
        if (w1_w) w1.weight.bind(w1_w);
        if (w1_b) w1.bias.bind(w1_b);
        if (w2_w) w2.weight.bind(w2_w);
        if (w2_b) w2.bias.bind(w2_b);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend = nullptr);
};

// 9. Multi-Head Self Attention (MHA)
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

    struct ggml_tensor* forward(
        struct ggml_context* ctx,
        struct ggml_tensor* x,
        struct ggml_tensor* mask,
        ggml_backend_t backend = nullptr,
        struct ggml_tensor* pos_tensor = nullptr
    );
};

// 10. Transformer Encoder Layer (Post-LN or Pre-LN)
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
        float epsilon = 1e-5f, bool use_pre_ln = false
    ) : pre_ln(use_pre_ln) {
        self_attn.n_heads = heads;
        self_attn.head_dim = dimension;
        ffn.act_type = activation;
        norm1.eps = epsilon;
        norm2.eps = epsilon;
    }

    struct ggml_tensor* forward(
        struct ggml_context* ctx,
        struct ggml_tensor* x,
        struct ggml_tensor* mask,
        ggml_backend_t backend = nullptr
    );
};

// 10b. Diffusion Transformer Block (DiTBlock)
class DiTBlock : public Module<DiTBlock> {
public:
    AdaLayerNormZero& attn_norm = submodule<AdaLayerNormZero>("attn_norm");
    MultiHeadAttention& attn = submodule<MultiHeadAttention>("attn");
    FeedForward& ff = submodule<FeedForward>("ff");
    float ff_norm_eps = 1e-5f;

    DiTBlock() = default;

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
class ResBlock1d : public Module<ResBlock1d> {
public:
    ModuleList<Conv1d>& convs1 = submodule<ModuleList<Conv1d>>("convs1");
    ModuleList<Conv1d>& convs2 = submodule<ModuleList<Conv1d>>("convs2");

    ResBlock1d() {
        for (int i = 0; i < 3; ++i) {
            convs1.emplace_back();
            convs2.emplace_back();
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend = nullptr);
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
class KVHeadAttention : public Module<KVHeadAttention> {
public:
    Linear& q_proj = submodule<Linear>("q_proj");
    Linear& k_proj = submodule<Linear>("k_proj");
    Linear& v_proj = submodule<Linear>("v_proj");
    Linear& out_proj = submodule<Linear>("out_proj");
    int n_heads = 1;
    int head_dim = 64;
    int layer_idx = 0;


    KVHeadAttention() = default;

    struct ggml_tensor* prefill(
        Context& context,
        struct ggml_tensor* x,
        KVCache& cache,
        struct ggml_tensor* mask = nullptr,
        struct ggml_cgraph* cgraph = nullptr,
        ggml_backend_t backend = nullptr
    );
    struct ggml_tensor* decode(
        Context& context,
        struct ggml_tensor* x,
        KVCache& cache,
        struct ggml_tensor* position,
        struct ggml_tensor* valid_length,
        ggml_backend_t backend = nullptr
    );
};

class TransformerDecoderLayer : public Module<TransformerDecoderLayer> {
public:
    KVHeadAttention& self_attn = submodule<KVHeadAttention>("self_attn");
    LayerNorm& ln1 = submodule<LayerNorm>("ln1");
    LayerNorm& ln2 = submodule<LayerNorm>("ln2");
    FeedForward& ffn = submodule<FeedForward>("ffn");

    TransformerDecoderLayer() = default;

    struct ggml_tensor* prefill(
        Context& context,
        struct ggml_tensor* x,
        KVCache& cache,
        struct ggml_tensor* mask = nullptr,
        struct ggml_cgraph* cgraph = nullptr,
        ggml_backend_t backend = nullptr
    );

    struct ggml_tensor* decode(
        Context& context,
        struct ggml_tensor* x,
        KVCache& cache,
        struct ggml_tensor* position,
        struct ggml_tensor* valid_length,
        ggml_backend_t backend = nullptr
    );

};

// 12. Transformer Encoder Stack
class TransformerEncoder : public Module<TransformerEncoder> {
public:
    ModuleList<TransformerEncoderLayer>& layers =
        submodule<ModuleList<TransformerEncoderLayer>>("layers");

    TransformerEncoder() = default;
    TransformerEncoder(int num_layers, int n_heads, int head_dim, ActivationType act, float eps = 1e-5f, bool pre_ln = false) {
        for (int i = 0; i < num_layers; ++i) {
            layers.emplace_back(n_heads, head_dim, act, eps, pre_ln);
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* mask, ggml_backend_t backend = nullptr);
};

class TransformerDecoder : public Module<TransformerDecoder> {
public:
    ModuleList<TransformerDecoderLayer>& layers =
        submodule<ModuleList<TransformerDecoderLayer>>("layers");

    TransformerDecoder() = default;
    TransformerDecoder(int num_layers, int n_heads, int head_dim, ActivationType act, float eps = 1e-5f) {
        reset(num_layers, n_heads, head_dim, act, eps);
    }

    void reset(int num_layers, int n_heads, int head_dim, ActivationType act, float eps = 1e-5f) {
        layers.clear();
        for (int i = 0; i < num_layers; ++i) {
            auto& layer = layers.emplace_back();
            layer.self_attn.n_heads = n_heads;
            layer.self_attn.head_dim = head_dim;
            layer.self_attn.layer_idx = i;
            layer.ln1.eps = eps;
            layer.ln2.eps = eps;
            layer.ffn.act_type = act;
        }
    }

    struct ggml_tensor* prefill(
        Context& context,
        struct ggml_tensor* x,
        KVCache& cache,
        struct ggml_tensor* mask = nullptr,
        struct ggml_cgraph* cgraph = nullptr,
        ggml_backend_t backend = nullptr
    );

    struct ggml_tensor* decode(
        Context& context,
        struct ggml_tensor* x,
        KVCache& cache,
        struct ggml_tensor* position,
        struct ggml_tensor* valid_length,
        ggml_backend_t backend = nullptr
    );

};

struct KVCacheConfig {
    ggml_type key_type = GGML_TYPE_F16;
    ggml_type value_type = GGML_TYPE_F16;
};

class KVCache {
public:
    int head_dim = 0;
    int max_len = 512;
    int n_heads = 0;
    int n_layers = 0;
    KVCacheConfig config;

    KVCache() = default;

    KVCache(const KVCache&) = delete;
    KVCache& operator=(const KVCache&) = delete;

    KVCache(KVCache&& other) noexcept {
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

    KVCache& operator=(KVCache&& other) noexcept {
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

    ~KVCache() { release(); }

    bool allocate(
        ggml_backend_t backend,
        int head_dim,
        int max_len,
        int n_heads,
        int n_layers,
        KVCacheConfig cache_config = {}) {
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
            k = context_->empty("kv_cache.key", {head_dim, max_len, n_heads, n_layers}, config.key_type);
            v = context_->empty("kv_cache.value", {head_dim, max_len, n_heads, n_layers}, config.value_type);
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

    struct ggml_tensor* decode_attention(
        Context& context,
        int layer,
        struct ggml_tensor* q,
        struct ggml_tensor* new_k,
        struct ggml_tensor* new_v,
        struct ggml_tensor* position,
        struct ggml_tensor* valid_length,
        float scale,
        ggml_backend_t backend,
        struct ggml_tensor* mask = nullptr
    );

    struct ggml_tensor* prefill_attention(
        Context& context,
        int layer,
        struct ggml_tensor* q,
        struct ggml_tensor* new_k,
        struct ggml_tensor* new_v,
        float scale,
        ggml_backend_t backend,
        struct ggml_tensor* mask = nullptr,
        struct ggml_cgraph* graph = nullptr
    );

private:
    struct ggml_tensor* k = nullptr;
    struct ggml_tensor* v = nullptr;
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
