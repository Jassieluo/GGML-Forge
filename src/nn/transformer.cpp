#include "nn/nn.h"

namespace nn {

// TransformerEncoderLayer
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

// DiTBlock
DiTBlock::DiTBlock(
    struct ggml_tensor* attn_ln_w, struct ggml_tensor* attn_ln_b,
    float attn_ln_eps,
    struct ggml_tensor* qw, struct ggml_tensor* qb,
    struct ggml_tensor* kw, struct ggml_tensor* kb,
    struct ggml_tensor* vw, struct ggml_tensor* vb,
    struct ggml_tensor* ow, struct ggml_tensor* ob,
    int n_heads, int head_dim,
    struct ggml_tensor* ff_ln_gamma, struct ggml_tensor* ff_ln_beta,
    float ff_ln_eps,
    struct ggml_tensor* ffn_w1, struct ggml_tensor* ffn_b1,
    struct ggml_tensor* ffn_w2, struct ggml_tensor* ffn_b2,
    ActivationType act
) : attn_norm(attn_ln_w, attn_ln_b, attn_ln_eps),
    attn(qw, qb, kw, kb, vw, vb, ow, ob, n_heads, head_dim),
    ff_norm(ff_ln_gamma, ff_ln_beta, ff_ln_eps),
    ff(ffn_w1, ffn_b1, ffn_w2, ffn_b2, act) {}

struct ggml_tensor* DiTBlock::forward(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* t,
    struct ggml_tensor* mask,
    ggml_backend_t backend,
    struct ggml_tensor* pos_tensor
) {
    // 1. Attention Norm & Modulation
    AdaLayerNormZero::Output norm_out = attn_norm.forward(ctx, x, t, backend);

    // 2. Attention
    struct ggml_tensor* attn_out = attn.forward(ctx, norm_out.x_modulated, mask, backend, pos_tensor);

    // 3. Process attention output for input x: x = x + gate_msa.unsqueeze(1) * attn_output
    int64_t C = attn_out->ne[0];
    int64_t B = attn_out->ne[2];
    struct ggml_tensor* gate_msa_reshaped = ggml_reshape_3d(ctx, norm_out.gate_msa, C, 1, B);
    struct ggml_tensor* scaled_attn = ggml_mul(ctx, attn_out, gate_msa_reshaped);
    x = ggml_add(ctx, x, scaled_attn);

    // 4. MLP Norm & Modulation: norm = LayerNorm(x) * (1 + scale_mlp) + shift_mlp
    struct ggml_tensor* norm_x = ggml_ops_ada_ln(ctx, x, norm_out.scale_mlp, norm_out.shift_mlp, ff_norm.eps, backend);

    // 5. FeedForward
    struct ggml_tensor* ff_out = ff.forward(ctx, norm_x, backend);

    // 6. Process FeedForward output: x = x + gate_mlp.unsqueeze(1) * ff_output
    struct ggml_tensor* gate_mlp_reshaped = ggml_reshape_3d(ctx, norm_out.gate_mlp, C, 1, B);
    struct ggml_tensor* scaled_ff = ggml_mul(ctx, ff_out, gate_mlp_reshaped);
    x = ggml_add(ctx, x, scaled_ff);

    return x;
}

} // namespace nn
