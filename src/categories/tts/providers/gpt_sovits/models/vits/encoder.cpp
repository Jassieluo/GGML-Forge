#include "providers/gpt_sovits/models/vits/encoder.h"

#include <cmath>
#include <string>

namespace gpt_sovits::vits {

LayerNorm::LayerNorm() = default;

ggml_tensor* LayerNorm::forward(
    nn::Context& context, ggml_tensor* x, ggml_backend_t backend) {
    return nn::F::layer_norm(
        context.native_handle(), x, gamma.tensor(), beta.tensor(), 1e-5f, backend);
}

EncoderLayer::EncoderLayer(int heads, int head_dim)
    : heads_(heads), head_dim_(head_dim) {
    ffn1.padding = 1;
    ffn2.padding = 1;
}

ggml_tensor* EncoderLayer::forward(
    nn::Context& context, ggml_tensor* x, ggml_backend_t backend) {
    ggml_context* ctx = context.native_handle();
    const int64_t time = x->ne[1];

    ggml_tensor* q = q_proj.forward(context, x, backend);
    ggml_tensor* k = k_proj.forward(context, x, backend);
    ggml_tensor* v = v_proj.forward(context, x, backend);
    q = ggml_cont(ctx, ggml_reshape_4d(ctx, q, head_dim_, heads_, time, 1));
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_reshape_4d(ctx, k, head_dim_, heads_, time, 1));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_reshape_4d(ctx, v, head_dim_, heads_, time, 1));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    if (q->type != GGML_TYPE_F32) q = ggml_cast(ctx, q, GGML_TYPE_F32);
    if (k->type != GGML_TYPE_F32) k = ggml_cast(ctx, k, GGML_TYPE_F32);
    if (v->type != GGML_TYPE_F32) v = ggml_cast(ctx, v, GGML_TYPE_F32);
    ggml_tensor* rel_k = relative_key.tensor();
    ggml_tensor* rel_v = relative_value.tensor();
    if (rel_k->type != GGML_TYPE_F32) rel_k = ggml_cast(ctx, rel_k, GGML_TYPE_F32);
    if (rel_v->type != GGML_TYPE_F32) rel_v = ggml_cast(ctx, rel_v, GGML_TYPE_F32);

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    ggml_tensor* bias = nn::F::relative_position_keys(ctx, q, rel_k, scale, 4, backend);
    ggml_tensor* weights = context.empty<float>("vits.attn.weights", {time, time, heads_, 1});
    ggml_tensor* out = nn::F::attention(ctx, q, k, v, bias, weights, scale, 4, backend);
    out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
    ggml_tensor* attention_output = out;
    out = ggml_reshape_2d(ctx, attention_output, head_dim_ * heads_, time);
    out = ggml_add(
        ctx, out,
        nn::F::relative_position_values(ctx, weights, rel_v, attention_output, 4, backend));

    x = norm1.forward(context, ggml_add(ctx, x, out_proj.forward(context, out, backend)), backend);
    ggml_tensor* ffn = ffn1.forward(context, x, backend);
    ffn = ffn2.forward(context, ggml_relu(ctx, ffn), backend);
    return norm2.forward(context, ggml_add(ctx, x, ffn), backend);
}

Encoder::Encoder(int layers, int heads, int head_dim) {
    for (int i = 0; i < layers; ++i) {
        layers_.emplace_back(heads, head_dim);
    }
}

ggml_tensor* Encoder::forward(
    nn::Context& context, ggml_tensor* x, ggml_backend_t backend) {
    for (auto& layer : layers_) x = layer.forward(context, x, backend);
    return x;
}

MRTE::MRTE() = default;

ggml_tensor* MRTE::forward(
    nn::Context& context, ggml_tensor* semantic, ggml_tensor* text,
    ggml_tensor* speaker, ggml_backend_t backend) {
    ggml_context* ctx = context.native_handle();
    constexpr int heads = 4;
    constexpr int head_dim = 128;
    const int64_t semantic_time = semantic->ne[1];
    const int64_t text_time = text->ne[1];
    ggml_tensor* semantic_hidden = semantic_proj.forward(context, semantic, backend);
    ggml_tensor* text_hidden = text_proj.forward(context, text, backend);

    auto split_heads = [&](ggml_tensor* value, int64_t time) {
        value = ggml_cont(ctx, ggml_reshape_3d(ctx, value, head_dim, heads, time));
        return ggml_cont(ctx, ggml_permute(ctx, value, 0, 2, 1, 3));
    };
    ggml_tensor* q = split_heads(q_proj.forward(context, semantic_hidden, backend), semantic_time);
    ggml_tensor* k = split_heads(k_proj.forward(context, text_hidden, backend), text_time);
    ggml_tensor* v = split_heads(v_proj.forward(context, text_hidden, backend), text_time);
    if (q->type != GGML_TYPE_F32) q = ggml_cast(ctx, q, GGML_TYPE_F32);
    if (k->type != GGML_TYPE_F32) k = ggml_cast(ctx, k, GGML_TYPE_F32);
    if (v->type != GGML_TYPE_F32) v = ggml_cast(ctx, v, GGML_TYPE_F32);

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    ggml_tensor* out = nn::F::attention(ctx, q, k, v, nullptr, nullptr, scale, -1, backend);
    out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
    out = ggml_reshape_2d(ctx, out, head_dim * heads, semantic_time);
    out = out_proj.forward(context, out, backend);
    out = ggml_add(ctx, semantic_hidden, out);
    out = ggml_add(ctx, out, ggml_reshape_2d(ctx, speaker, speaker->ne[0], 1));
    return result_proj.forward(context, out, backend);
}

} // namespace gpt_sovits::vits
