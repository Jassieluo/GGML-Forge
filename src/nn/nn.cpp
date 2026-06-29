#include "nn/nn.h"

namespace nn {

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

// 7. MultiHeadAttention
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
    // x: [hidden_dim, seq_len]
    struct ggml_tensor* q = q_proj.forward(ctx, x);
    struct ggml_tensor* k = k_proj.forward(ctx, x);
    struct ggml_tensor* v = v_proj.forward(ctx, x);

    // Project Q, K, V to [head_dim, seq_len, n_heads, 1]
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

    // Permute and reshape back to [hidden_dim, seq_len]
    attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));
    attn_out = ggml_reshape_2d(ctx, attn_out, n_heads * head_dim, attn_out->ne[2]);

    return out_proj.forward(ctx, attn_out);
}

} // namespace nn
