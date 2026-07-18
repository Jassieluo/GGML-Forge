#include "conditioning.h"

#include <cmath>

namespace gpt_sovits::vits {

TemporalGLU::TemporalGLU() = default;

ggml_tensor* TemporalGLU::forward(
    nn::Context& context, ggml_tensor* x, ggml_backend_t backend) {
    ggml_context* ctx = context.native_handle();
    const int64_t channels = x->ne[0];
    ggml_tensor* transposed = ggml_cont(ctx, ggml_transpose(ctx, x));
    ggml_tensor* w = weight.tensor();
    const int padding = static_cast<int>((ggml_is_quantized(w->type) ? w->ne[1] : w->ne[0]) - 1) / 2;
    ggml_tensor* conv = nn::F::conv1d_no_transpose(
        ctx, transposed, w, nullptr, 1, padding, 1, 1, backend);
    conv = ggml_cont(ctx, ggml_transpose(ctx, conv));
    conv = ggml_add(ctx, conv, ggml_reshape_2d(ctx, bias.tensor(), bias.tensor()->ne[0], 1));
    ggml_tensor* lhs = ggml_view_2d(ctx, conv, channels, conv->ne[1], conv->nb[1], 0);
    ggml_tensor* rhs = ggml_view_2d(
        ctx, conv, channels, conv->ne[1], conv->nb[1], channels * sizeof(float));
    return ggml_add(ctx, x, ggml_mul(ctx, lhs, ggml_sigmoid(ctx, rhs)));
}

ReferenceEncoder::ReferenceEncoder() = default;

ggml_tensor* ReferenceEncoder::forward(
    nn::Context& context, ggml_tensor* mel, ggml_backend_t backend) {
    ggml_context* ctx = context.native_handle();
    const int64_t time = mel->ne[1];
    ggml_tensor* x = nn::F::mish(ctx, spectral0.forward(ctx, mel), backend);
    x = nn::F::mish(ctx, spectral3.forward(ctx, x), backend);
    x = temporal0.forward(context, x, backend);
    x = temporal1.forward(context, x, backend);

    constexpr int heads = 2;
    constexpr int head_dim = 64;
    ggml_tensor* residual = x;
    auto project = [&](nn::Linear& projection) {
        ggml_tensor* value = projection.forward(ctx, x);
        value = ggml_cont(ctx, ggml_reshape_3d(ctx, value, head_dim, heads, time));
        value = ggml_cont(ctx, ggml_permute(ctx, value, 0, 2, 1, 3));
        return ggml_reshape_4d(ctx, value, head_dim, time, heads, 1);
    };
    ggml_tensor* q = project(q_proj);
    ggml_tensor* k = project(k_proj);
    ggml_tensor* v = project(v_proj);
    if (q->type != GGML_TYPE_F32) q = ggml_cast(ctx, q, GGML_TYPE_F32);
    if (k->type != GGML_TYPE_F32) k = ggml_cast(ctx, k, GGML_TYPE_F32);
    if (v->type != GGML_TYPE_F32) v = ggml_cast(ctx, v, GGML_TYPE_F32);

    const float scale = 1.0f / std::sqrt(128.0f);
    x = nn::F::attention(ctx, q, k, v, nullptr, nullptr, scale, -1, backend);
    x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
    x = ggml_cont(ctx, ggml_reshape_2d(ctx, x, head_dim * heads, time));
    x = ggml_add(ctx, attention_out.forward(ctx, x), residual);
    x = output.forward(ctx, x);
    x = ggml_cont(ctx, ggml_transpose(ctx, x));
    x = ggml_scale(ctx, ggml_sum_rows(ctx, x), 1.0f / static_cast<float>(time));
    return ggml_cont(ctx, x);
}

} // namespace gpt_sovits::vits
