#include "nn/nn.h"

namespace nn {

// 0. Embedding
Embedding::Embedding(struct ggml_tensor* w)
    : weight(w) {}

struct ggml_tensor* Embedding::forward(struct ggml_context* ctx, struct ggml_tensor* input_ids) {
    return ggml_get_rows(ctx, weight, input_ids);
}

// 1. Linear
Linear::Linear(struct ggml_tensor* w, struct ggml_tensor* b)
    : weight(w), bias(b) {}

struct ggml_tensor* Linear::forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    struct ggml_tensor* out = ggml_mul_mat(ctx, weight, x);
    if (bias) {
        struct ggml_tensor* b_reshaped = ggml_reshape_2d(ctx, bias, bias->ne[0], 1);
        out = ggml_add(ctx, out, b_reshaped);
    }
    return out;
}

// 2. Conv1d
Conv1d::Conv1d(struct ggml_tensor* w, struct ggml_tensor* b, int stride, int padding, int dilation)
    : weight(w), bias(b), stride(stride), padding(padding), dilation(dilation) {}

struct ggml_tensor* Conv1d::forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend) {
    // x shape: [in_channels, seq_len] -> transpose to [seq_len, in_channels]
    struct ggml_tensor* x_trans = ggml_cont(ctx, ggml_transpose(ctx, x));
    struct ggml_tensor* conv = ggml_ops_conv_1d(ctx, weight, x_trans, stride, padding, dilation, backend);
    struct ggml_tensor* out = ggml_cont(ctx, ggml_transpose(ctx, conv));
    if (bias) {
        struct ggml_tensor* b_reshaped = ggml_reshape_2d(ctx, bias, bias->ne[0], 1);
        out = ggml_add(ctx, out, b_reshaped);
    }
    return out;
}

// 3. ConvTranspose1d
ConvTranspose1d::ConvTranspose1d(struct ggml_tensor* w, struct ggml_tensor* b, int stride, int padding, int dilation)
    : weight(w), bias(b), stride(stride), padding(padding), dilation(dilation) {}

struct ggml_tensor* ConvTranspose1d::forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend) {
    // x shape: [in_channels, seq_len] -> transpose to [seq_len, in_channels]
    struct ggml_tensor* x_trans = ggml_cont(ctx, ggml_transpose(ctx, x));
    struct ggml_tensor* conv = ggml_ops_conv_transpose_1d(ctx, weight, x_trans, stride, padding, dilation, backend);
    struct ggml_tensor* out = ggml_cont(ctx, ggml_transpose(ctx, conv));
    if (bias) {
        struct ggml_tensor* b_reshaped = ggml_reshape_2d(ctx, bias, bias->ne[0], 1);
        out = ggml_add(ctx, out, b_reshaped);
    }
    return out;
}

// 4. LayerNorm
LayerNorm::LayerNorm(struct ggml_tensor* gamma, struct ggml_tensor* beta, float eps)
    : gamma(gamma), beta(beta), eps(eps) {}

struct ggml_tensor* LayerNorm::forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend) {
    return ggml_ops_layer_norm(ctx, x, gamma, beta, eps, backend);
}

// 5. InstanceNorm
InstanceNorm::InstanceNorm(struct ggml_tensor* gamma, struct ggml_tensor* beta, float eps)
    : gamma(gamma), beta(beta), eps(eps) {}

struct ggml_tensor* InstanceNorm::forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend) {
    return ggml_ops_instance_norm(ctx, x, gamma, beta, eps, backend);
}

// 6. GLU
struct ggml_tensor* GLU::forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend) {
    return ggml_ops_glu(ctx, x, backend);
}

// 7. FeedForward
FeedForward::FeedForward(
    struct ggml_tensor* w1_w, struct ggml_tensor* w1_b,
    struct ggml_tensor* w2_w, struct ggml_tensor* w2_b,
    ActivationType act
) : w1(w1_w, w1_b), w2(w2_w, w2_b), act_type(act) {}

struct ggml_tensor* FeedForward::forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend) {
    struct ggml_tensor* h = w1.forward(ctx, x);
    switch (act_type) {
        case ActivationType::GELU:
            h = ggml_gelu(ctx, h);
            break;
        case ActivationType::GELU_ERF:
            h = ggml_gelu_erf(ctx, h);
            break;
        case ActivationType::RELU:
            h = ggml_relu(ctx, h);
            break;
        case ActivationType::LEAKY_RELU:
            h = ggml_leaky_relu(ctx, h, 0.1f, false);
            break;
        case ActivationType::MISH:
            h = ggml_ops_mish(ctx, h, backend);
            break;
        case ActivationType::DOUBLE_SWISH:
            h = ggml_ops_double_swish(ctx, h, backend);
            break;
        case ActivationType::SILU:
            h = ggml_silu(ctx, h);
            break;
    }
    return w2.forward(ctx, h);
}

// 8. MultiHeadAttention
MultiHeadAttention::MultiHeadAttention(
    struct ggml_tensor* qw, struct ggml_tensor* qb,
    struct ggml_tensor* kw, struct ggml_tensor* kb,
    struct ggml_tensor* vw, struct ggml_tensor* vb,
    struct ggml_tensor* ow, struct ggml_tensor* ob,
    int n_heads, int head_dim
) : q_proj(qw, qb), k_proj(kw, kb), v_proj(vw, vb), out_proj(ow, ob),
    n_heads(n_heads), head_dim(head_dim) {}

struct ggml_tensor* MultiHeadAttention::forward(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* mask,
    ggml_backend_t backend
) {
    struct ggml_tensor* q = q_proj.forward(ctx, x);
    struct ggml_tensor* k = k_proj.forward(ctx, x);
    struct ggml_tensor* v = v_proj.forward(ctx, x);

    q = ggml_cont(ctx, ggml_reshape_3d(ctx, q, head_dim, n_heads, q->ne[1]));
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    q = ggml_reshape_4d(ctx, q, head_dim, q->ne[1], n_heads, 1);

    k = ggml_cont(ctx, ggml_reshape_3d(ctx, k, head_dim, n_heads, k->ne[1]));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    k = ggml_reshape_4d(ctx, k, head_dim, k->ne[1], n_heads, 1);

    v = ggml_cont(ctx, ggml_reshape_3d(ctx, v, head_dim, n_heads, v->ne[1]));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
    v = ggml_reshape_4d(ctx, v, head_dim, v->ne[1], n_heads, 1);

    float scale = 1.0f / std::sqrt((float)head_dim);
    struct ggml_tensor* attn_out = ggml_ops_attention(ctx, q, k, v, mask, nullptr, scale, -1, backend);

    attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));
    attn_out = ggml_reshape_2d(ctx, attn_out, n_heads * head_dim, attn_out->ne[2]);

    return out_proj.forward(ctx, attn_out);
}

// 9. TransformerEncoderLayer
TransformerEncoderLayer::TransformerEncoderLayer(
    struct ggml_tensor* qw, struct ggml_tensor* qb,
    struct ggml_tensor* kw, struct ggml_tensor* kb,
    struct ggml_tensor* vw, struct ggml_tensor* vb,
    struct ggml_tensor* ow, struct ggml_tensor* ob,
    int n_heads, int head_dim,
    struct ggml_tensor* ffn_w1, struct ggml_tensor* ffn_b1,
    struct ggml_tensor* ffn_w2, struct ggml_tensor* ffn_b2,
    ActivationType act,
    struct ggml_tensor* ln1_w, struct ggml_tensor* ln1_b,
    struct ggml_tensor* ln2_w, struct ggml_tensor* ln2_b,
    float eps,
    bool pre_ln
) : self_attn(qw, qb, kw, kb, vw, vb, ow, ob, n_heads, head_dim),
    ffn(ffn_w1, ffn_b1, ffn_w2, ffn_b2, act),
    norm1(ln1_w, ln1_b, eps),
    norm2(ln2_w, ln2_b, eps),
    pre_ln(pre_ln) {}

struct ggml_tensor* TransformerEncoderLayer::forward(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* mask,
    ggml_backend_t backend
) {
    if (pre_ln) {
        // Pre-LN: x = x + Attention(LN1(x))
        struct ggml_tensor* norm_x = norm1.forward(ctx, x, backend);
        struct ggml_tensor* attn_out = self_attn.forward(ctx, norm_x, mask, backend);
        x = ggml_add(ctx, x, attn_out);

        // x = x + FFN(LN2(x))
        struct ggml_tensor* norm_x2 = norm2.forward(ctx, x, backend);
        struct ggml_tensor* ffn_out = ffn.forward(ctx, norm_x2, backend);
        x = ggml_add(ctx, x, ffn_out);
    } else {
        // Post-LN: x = LN1(x + Attention(x))
        struct ggml_tensor* attn_out = self_attn.forward(ctx, x, mask, backend);
        x = ggml_add(ctx, x, attn_out);
        x = norm1.forward(ctx, x, backend);

        // x = LN2(x + FFN(x))
        struct ggml_tensor* ffn_out = ffn.forward(ctx, x, backend);
        x = ggml_add(ctx, x, ffn_out);
        x = norm2.forward(ctx, x, backend);
    }
    return x;
}

// 10. ResBlock1d
ResBlock1d::ResBlock1d(
    struct ggml_tensor* convs1_w[3], struct ggml_tensor* convs1_b[3],
    struct ggml_tensor* convs2_w[3], struct ggml_tensor* convs2_b[3],
    const std::vector<int>& dilations,
    int kernel_size
) {
    for (int i = 0; i < 3; ++i) {
        int dilation = dilations[i];
        int padding = (kernel_size - 1) * dilation / 2;
        convs1[i] = Conv1d(convs1_w[i], convs1_b[i], 1, padding, dilation);
        convs2[i] = Conv1d(convs2_w[i], convs2_b[i], 1, (kernel_size - 1) / 2, 1);
    }
}

struct ggml_tensor* ResBlock1d::forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend) {
    struct ggml_tensor* current_x = x;
    for (int i = 0; i < 3; ++i) {
        struct ggml_tensor* xt = ggml_leaky_relu(ctx, current_x, 0.1f, false);
        xt = convs1[i].forward(ctx, xt, backend);
        xt = ggml_leaky_relu(ctx, xt, 0.1f, false);
        xt = convs2[i].forward(ctx, xt, backend);
        current_x = ggml_add(ctx, xt, current_x);
    }
    return current_x;
}

} // namespace nn
