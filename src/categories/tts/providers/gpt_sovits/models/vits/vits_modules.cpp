#include "providers/gpt_sovits/models/vits/vits.h"

#include "nn/nn.h"

namespace gpt_sovits {

ggml_tensor* VITSModel::compute_speaker_embedding(
    nn::Context& context,
    ggml_tensor* mel_spec,
    ggml_tensor* sv_emb,
    ggml_backend_t backend
) {
    ggml_context* ctx = context.native_handle();
    ggml_tensor* ge = reference_encoder.forward(context, mel_spec, backend);

    if (speaker && speaker->projection.bias.is_bound()) {
        ggml_tensor* sv_emb_b = speaker->projection.bias.tensor();
        const int64_t dim = sv_emb_b->ne[0];

        ggml_tensor* sv_proj = nullptr;
        if (sv_emb && speaker->projection.weight.is_bound()) {
            sv_proj = speaker->projection.forward(ctx, sv_emb);
            sv_proj = ggml_reshape_2d(ctx, sv_proj, 1, dim);
        } else {
            sv_proj = ggml_reshape_2d(ctx, sv_emb_b, 1, dim);
        }

        ge = ggml_add(ctx, ge, sv_proj);
        ge = speaker->activation.forward(ctx, ge, backend);
    }
    return ge;
}

ggml_tensor* WN::forward(
    ggml_context* ctx,
    ggml_tensor* x,
    ggml_tensor* x_mask,
    ggml_tensor* g,
    ggml_backend_t backend
) {
    ggml_backend_t selected_backend = backend ? backend : this->backend;
    nn::Context graph_context = nn::Context::borrow(ctx);
    ggml_tensor* output = graph_context.empty<float>(
        "vits.wn.output", {hidden_channels, x->ne[1]});
    output = ggml_fill(ctx, output, 0.0f);

    for (int i = 0; i < n_layers; ++i) {
        ggml_tensor* x_in = in_layers[i].forward(ctx, x, selected_backend);
        if (g) {
            const int cond_offset = i * 2 * hidden_channels;
            ggml_tensor* g_l = ggml_view_2d(
                ctx, g, 2 * hidden_channels, g->ne[1], g->nb[1],
                cond_offset * sizeof(float));
            x_in = ggml_add(ctx, x_in, ggml_cont(ctx, g_l));
        }

        ggml_tensor* acts = nn::F::gated_tanh_sigmoid(
            ctx, x_in, hidden_channels, selected_backend);
        ggml_tensor* res_skip = res_skip_layers[i].forward(
            ctx, acts, selected_backend);

        if (i < n_layers - 1) {
            const size_t stride = res_skip->ne[0] * sizeof(float);
            ggml_tensor* residual = ggml_cont(ctx, ggml_view_2d(
                ctx, res_skip, hidden_channels, res_skip->ne[1], stride, 0));
            x = ggml_add(ctx, x, residual);
            if (x_mask) x = ggml_mul(ctx, x, ggml_repeat(ctx, x_mask, x));
            ggml_tensor* skip = ggml_cont(ctx, ggml_view_2d(
                ctx, res_skip, hidden_channels, res_skip->ne[1], stride,
                hidden_channels * sizeof(float)));
            output = ggml_add(ctx, output, skip);
        } else {
            output = ggml_add(ctx, output, res_skip);
        }
    }
    if (x_mask) output = ggml_mul(ctx, output, ggml_repeat(ctx, x_mask, output));
    return output;
}

ggml_tensor* ResidualCouplingLayer::forward(
    ggml_context* ctx,
    ggml_tensor* x,
    ggml_tensor* x_mask,
    ggml_tensor* g,
    ggml_backend_t backend
) {
    ggml_backend_t selected_backend = backend ? backend : this->backend;
    const int half_channels = static_cast<int>(x->ne[0] / 2);
    const size_t stride = x->ne[0] * sizeof(float);
    ggml_tensor* x0 = ggml_view_2d(ctx, x, half_channels, x->ne[1], stride, 0);
    ggml_tensor* x1 = ggml_view_2d(
        ctx, x, half_channels, x->ne[1], stride, half_channels * sizeof(float));

    ggml_tensor* h = pre.forward(ctx, x0, selected_backend);
    if (x_mask) h = ggml_mul(ctx, h, ggml_repeat(ctx, x_mask, h));

    ggml_tensor* projected_condition = nullptr;
    if (wn.cond_layer.weight.is_bound() && g) {
        projected_condition = wn.cond_layer.forward(ctx, g, selected_backend);
    }
    h = wn.forward(ctx, h, x_mask, projected_condition, selected_backend);

    ggml_tensor* mean = post.forward(ctx, h, selected_backend);
    if (x_mask) mean = ggml_mul(ctx, mean, ggml_repeat(ctx, x_mask, mean));
    ggml_tensor* new_x1 = ggml_sub(ctx, ggml_cont(ctx, x1), mean);
    return ggml_cont(ctx, ggml_concat(ctx, x0, new_x1, 0));
}

ggml_tensor* WNEncoder::forward(
    ggml_context* ctx,
    ggml_tensor* x,
    ggml_tensor* g,
    ggml_backend_t backend
) {
    ggml_backend_t selected_backend = backend ? backend : this->backend;
    ggml_tensor* h = pre.forward(ctx, x, selected_backend);
    ggml_tensor* projected_condition = g
        ? wn.cond_layer.forward(ctx, g, selected_backend)
        : nullptr;
    ggml_tensor* encoded = wn.forward(
        ctx, h, nullptr, projected_condition, selected_backend);
    return proj.forward(ctx, encoded, selected_backend);
}

} // namespace gpt_sovits
