#include "nn/layers/transformer.h"

#include "ops/ops.h"

namespace nn {

struct ggml_tensor* TransformerEncoderLayer::forward(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* mask,
    ggml_backend_t backend
) {
    ggml_backend_t b = backend ? backend : this->backend;
    if (pre_ln) {
        // Pre-LN: x = x + Attention(LN1(x))
        struct ggml_tensor* norm_x = norm1.forward(ctx, x, b);
        struct ggml_tensor* attn_out = self_attn.forward(ctx, norm_x, mask, b);
        x = ggml_add(ctx, x, attn_out);

        // x = x + FFN(LN2(x))
        struct ggml_tensor* norm_x2 = norm2.forward(ctx, x, b);
        struct ggml_tensor* ffn_out = ffn.forward(ctx, norm_x2, b);
        x = ggml_add(ctx, x, ffn_out);
    } else {
        // Post-LN: x = LN1(x + Attention(x)), residual add fused into the norm
        struct ggml_tensor* attn_out = self_attn.forward(ctx, x, mask, b);
        x = norm1.forward_residual(ctx, x, attn_out, b);

        // x = LN2(x + FFN(x))
        struct ggml_tensor* ffn_out = ffn.forward(ctx, x, b);
        x = norm2.forward_residual(ctx, x, ffn_out, b);
    }
    return x;
}

// DiTBlock

struct ggml_tensor* DiTBlock::forward(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* t,
    struct ggml_tensor* mask,
    ggml_backend_t backend,
    struct ggml_tensor* pos_tensor
) {
    ggml_backend_t b = backend ? backend : this->backend;
    // 1. Attention Norm & Modulation
    AdaLayerNormZero::Output norm_out = attn_norm.forward(ctx, x, t, b);

    // 2. Attention
    struct ggml_tensor* attn_out = attn.forward(ctx, norm_out.x_modulated, mask, b, pos_tensor);

    // 3. Process attention output for input x: x = x + gate_msa.unsqueeze(1) * attn_output
    int64_t C = attn_out->ne[0];
    int64_t B = attn_out->ne[2];
    struct ggml_tensor* gate_msa_reshaped = ggml_reshape_3d(ctx, norm_out.gate_msa, C, 1, B);
    struct ggml_tensor* scaled_attn = ggml_mul(ctx, attn_out, gate_msa_reshaped);
    x = ggml_add(ctx, x, scaled_attn);

    // 4. MLP Norm & Modulation: norm = LayerNorm(x) * (1 + scale_mlp) + shift_mlp
    struct ggml_tensor* norm_x = ggml_ops_ada_ln(
        ctx, x, norm_out.scale_mlp, norm_out.shift_mlp, ff_norm_eps, b);

    // 5. FeedForward
    struct ggml_tensor* ff_out = ff.forward(ctx, norm_x, b);

    // 6. Process FeedForward output: x = x + gate_mlp.unsqueeze(1) * ff_output
    struct ggml_tensor* gate_mlp_reshaped = ggml_reshape_3d(ctx, norm_out.gate_mlp, C, 1, B);
    struct ggml_tensor* scaled_ff = ggml_mul(ctx, ff_out, gate_mlp_reshaped);
    x = ggml_add(ctx, x, scaled_ff);

    return x;
}

struct ggml_tensor* TransformerEncoder::forward(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* mask,
    ggml_backend_t backend
) {
    ggml_backend_t b = backend ? backend : this->backend;
    for (auto& layer : layers) {
        x = layer.forward(ctx, x, mask, b);
    }
    return x;
}

struct ggml_tensor* TransformerDecoderLayer::prefill(
    Context& context,
    struct ggml_tensor* x,
    KVCache& cache,
    struct ggml_tensor* mask,
    ggml_backend_t backend
) {
    ggml_context* ctx = context.native_handle();
    ggml_backend_t b = backend ? backend : this->backend;

    struct ggml_tensor* attn_out = self_attn.prefill(context, x, cache, mask, b);
    x = ln1.forward_residual(ctx, x, attn_out, b);

    struct ggml_tensor* mlp_out = ffn(ctx, x, b);
    x = ln2.forward_residual(ctx, x, mlp_out, b);

    return x;
}

struct ggml_tensor* TransformerDecoder::prefill(
    Context& context,
    struct ggml_tensor* x,
    KVCache& cache,
    struct ggml_tensor* mask,
    ggml_backend_t backend
) {
    ggml_backend_t b = backend ? backend : this->backend;
    for (auto& layer : layers) {
        x = layer.prefill(context, x, cache, mask, b);
    }
    return x;
}

struct ggml_tensor* TransformerDecoderLayer::decode(
    Context& context,
    ggml_tensor* x,
    KVCache& cache,
    ggml_tensor* position,
    ggml_tensor* valid_length,
    ggml_backend_t backend
) {
    ggml_context* ctx = context.native_handle();
    ggml_backend_t b = backend ? backend : this->backend;
    ggml_tensor* attn_out = self_attn.decode(context, x, cache, position, valid_length, b);
    x = ln1.forward_residual(ctx, x, attn_out, b);
    x = ln2.forward_residual(ctx, x, ffn(ctx, x, b), b);
    return x;
}

struct ggml_tensor* TransformerDecoder::decode(
    Context& context,
    ggml_tensor* x,
    KVCache& cache,
    ggml_tensor* position,
    ggml_tensor* valid_length,
    ggml_backend_t backend
) {
    ggml_backend_t b = backend ? backend : this->backend;
    for (auto& layer : layers) {
        x = layer.decode(context, x, cache, position, valid_length, b);
    }
    return x;
}

} // namespace nn
