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

} // namespace nn
