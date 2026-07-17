#include "vits.h"
#include "ops/ops.h"
#include <iostream>
#include <random>
#include <cstring>
#include <vector>
#include <cmath>
#include <cstdlib>

namespace gpt_sovits {

struct ggml_tensor* ggml_snake_beta(
    struct ggml_context* ctx,
    struct ggml_tensor* x,          // [seq_len, channels]
    struct ggml_tensor* alpha,      // [channels]
    struct ggml_tensor* beta,       // [channels]
    ggml_backend_t backend
) {
    return ggml_ops_snake_beta(ctx, x, alpha, beta, backend);
}

static struct ggml_tensor* alias_free_activation_with_prefix(
    struct ggml_context* ctx,
    struct ggml_tensor* x,          // [seq_len, channels]
    VITSModel& model,
    const std::string& act_prefix,
    int channels,
    struct ggml_tensor* alpha,
    struct ggml_tensor* beta,
    ggml_backend_t backend
) {
    struct ggml_tensor* up_filter_rep = model.get_tensor(act_prefix + ".upsample.filter_repeated");
    struct ggml_tensor* down_filter_rep = model.get_tensor(act_prefix + ".downsample.filter_repeated");
    if (struct ggml_tensor* fused = ggml_ops_alias_free_activation(
            ctx, x, up_filter_rep, down_filter_rep, alpha, beta, backend)) {
        return fused;
    }
    struct ggml_tensor* upsampled = ggml_ops_conv_transpose_1d(ctx, up_filter_rep, x, 2, 5, 1, channels, backend);
    upsampled = ggml_scale(ctx, upsampled, 2.0f);

    struct ggml_tensor* act_out = ggml_snake_beta(ctx, upsampled, alpha, beta, backend);

    struct ggml_tensor* downsampled = ggml_ops_conv_1d(ctx, down_filter_rep, act_out, 2, 5, 1, channels, backend);

    return downsampled;
}

static struct ggml_tensor* alias_free_activation(
    struct ggml_context* ctx,
    struct ggml_tensor* x,          // [seq_len, channels]
    VITSModel& model,
    int block_idx,
    int act_idx,
    int channels,
    struct ggml_tensor* alpha,
    struct ggml_tensor* beta,
    ggml_backend_t backend
) {
    std::string act_prefix = "dec.resblocks." + std::to_string(block_idx) + ".activations." + std::to_string(act_idx);
    return alias_free_activation_with_prefix(ctx, x, model, act_prefix, channels, alpha, beta, backend);
}

static struct ggml_tensor* cfm_resblock_no_transpose(
    struct ggml_context* ctx,
    struct ggml_tensor* x,      // [seq_len, channels]
    VITSModel& model,
    int block_idx,
    int channels,
    int kernel_size,
    const std::vector<int>& dilations,
    ggml_backend_t backend
) {
    struct ggml_tensor* current_x = x;

    for (int l = 0; l < 3; ++l) {
        int dilation = dilations[l];
        int padding = (kernel_size - 1) * dilation / 2;
        int padding2 = (kernel_size - 1) / 2;

        std::string prefix1 = "dec.resblocks." + std::to_string(block_idx) + ".convs1." + std::to_string(l);
        std::string prefix2 = "dec.resblocks." + std::to_string(block_idx) + ".convs2." + std::to_string(l);

        struct ggml_tensor* c1_w = model.get_tensor(prefix1 + ".weight");
        struct ggml_tensor* c1_b = model.get_tensor(prefix1 + ".bias");
        struct ggml_tensor* c2_w = model.get_tensor(prefix2 + ".weight");
        struct ggml_tensor* c2_b = model.get_tensor(prefix2 + ".bias");

        // Activations
        std::string act_p1 = "dec.resblocks." + std::to_string(block_idx) + ".activations." + std::to_string(2 * l) + ".act.";
        std::string act_p2 = "dec.resblocks." + std::to_string(block_idx) + ".activations." + std::to_string(2 * l + 1) + ".act.";

        struct ggml_tensor* alpha1 = model.get_tensor(act_p1 + "alpha");
        struct ggml_tensor* beta1 = model.get_tensor(act_p1 + "beta");
        struct ggml_tensor* alpha2 = model.get_tensor(act_p2 + "alpha");
        struct ggml_tensor* beta2 = model.get_tensor(act_p2 + "beta");

        // xt = a1(x)
        struct ggml_tensor* xt = alias_free_activation(ctx, current_x, model, block_idx, 2 * l, channels, alpha1, beta1, backend);
        // xt = c1(xt)
        xt = nn::F::conv1d_no_transpose(ctx, xt, c1_w, c1_b, 1, padding, dilation, 1, backend);
        // xt = a2(xt)
        xt = alias_free_activation(ctx, xt, model, block_idx, 2 * l + 1, channels, alpha2, beta2, backend);
        // xt = c2(xt)
        xt = nn::F::conv1d_no_transpose(ctx, xt, c2_w, c2_b, 1, padding2, 1, 1, backend);

        // x = xt + x
        current_x = ggml_add(ctx, xt, current_x);
    }

    return current_x;
}

static struct ggml_tensor* mrf_resblock_no_transpose_cfm(
    struct ggml_context* ctx,
    struct ggml_tensor* x,      // [seq_len, channels]
    VITSModel& model,
    int block_idx,
    int channels,
    int kernel_size,
    const std::vector<int>& dilations,
    ggml_backend_t backend
) {
    struct ggml_tensor* current_x = x;

    for (int l = 0; l < 3; ++l) {
        int dilation = dilations[l];
        int padding = (kernel_size - 1) * dilation / 2;

        std::string prefix1 = "dec.resblocks." + std::to_string(block_idx) + ".convs1." + std::to_string(l);
        std::string prefix2 = "dec.resblocks." + std::to_string(block_idx) + ".convs2." + std::to_string(l);

        struct ggml_tensor* c1_w = model.get_tensor(prefix1 + ".weight");
        int dilation_effective = dilation;
        
        struct ggml_tensor* c1_b = model.get_tensor(prefix1 + ".bias");
        struct ggml_tensor* c2_w = model.get_tensor(prefix2 + ".weight");
        struct ggml_tensor* c2_b = model.get_tensor(prefix2 + ".bias");

        if (!c1_w || !c1_b || !c2_w || !c2_b) {
            continue;
        }

        struct ggml_tensor* xt = ggml_leaky_relu(ctx, current_x, 0.1f, false);
        xt = nn::F::conv1d_no_transpose(ctx, xt, c1_w, c1_b, 1, padding, dilation_effective, 1, backend);
        xt = ggml_leaky_relu(ctx, xt, 0.1f, false);
        xt = nn::F::conv1d_no_transpose(ctx, xt, c2_w, c2_b, 1, (kernel_size - 1) / 2, 1, 1, backend);

        current_x = ggml_add(ctx, xt, current_x);
    }

    return current_x;
}

struct ggml_tensor* build_vits_generator_cfm(
    struct ggml_context* ctx_graph,
    struct ggml_tensor* latent,
    struct ggml_tensor* speaker_embedding,
    VITSModel& model,
    ggml_backend_t backend
) {
    struct ggml_tensor* dec_conv_pre_w = model.get_tensor("dec.conv_pre.weight");
    struct ggml_tensor* dec_conv_pre_b = model.get_tensor("dec.conv_pre.bias");

    struct ggml_tensor* latent_transposed = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, latent));

    struct ggml_tensor* h = nn::F::conv1d_no_transpose(ctx_graph, latent_transposed, dec_conv_pre_w, dec_conv_pre_b, 1, 3, 1, 1, backend);

    const std::vector<int>& upsample_rates = model.profile.upsample_rates;

    const std::vector<int> resblock_kernel_sizes = {3, 7, 11};
    const std::vector<int> dilation_sizes = {1, 3, 5};

    for (size_t i = 0; i < upsample_rates.size(); ++i) {
        int stride = upsample_rates[i];
        std::string ups_prefix = "dec.ups." + std::to_string(i) + ".0.";
        struct ggml_tensor* ups_w = model.get_tensor(ups_prefix + "weight");
        struct ggml_tensor* ups_b = model.get_tensor(ups_prefix + "bias");
        if (!ups_w) {
            ups_prefix = "dec.ups." + std::to_string(i) + ".";
            ups_w = model.get_tensor(ups_prefix + "weight");
            ups_b = model.get_tensor(ups_prefix + "bias");
        }

        int kernel_size = (int)(ggml_is_quantized(ups_w->type) ? ups_w->ne[1] : ups_w->ne[0]);
        int padding = (kernel_size - stride) / 2;
        int out_channels = (int)(ggml_is_quantized(ups_w->type) ? ups_w->ne[0] : ups_w->ne[1]);

        if (model.profile.vocoder_architecture == "hifigan") {
            h = ggml_leaky_relu(ctx_graph, h, 0.1f, false);
        }

        h = nn::F::conv_transpose1d_no_transpose(ctx_graph, h, ups_w, ups_b, stride, padding, backend);

        struct ggml_tensor* res_x = nullptr;
        for (int j = 0; j < 3; ++j) {
            int block_idx = i * 3 + j;
            int r_kernel_size = resblock_kernel_sizes[j];
            
            struct ggml_tensor* block_out = nullptr;
            if (model.profile.vocoder_architecture == "bigvgan-v2") {
                block_out = cfm_resblock_no_transpose(ctx_graph, h, model, block_idx, out_channels, r_kernel_size, dilation_sizes, backend);
            } else { // V4
                block_out = mrf_resblock_no_transpose_cfm(ctx_graph, h, model, block_idx, out_channels, r_kernel_size, dilation_sizes, backend);
            }

            if (res_x == nullptr) {
                res_x = block_out;
            } else {
                res_x = ggml_add(ctx_graph, res_x, block_out);
            }
        }
        h = ggml_scale(ctx_graph, res_x, 1.0f / 3.0f);
    }

    if (model.profile.vocoder_architecture == "bigvgan-v2") {
        struct ggml_tensor* alpha_post = model.get_tensor("dec.activation_post.act.alpha");
        struct ggml_tensor* beta_post = model.get_tensor("dec.activation_post.act.beta");
        h = alias_free_activation_with_prefix(ctx_graph, h, model, "dec.activation_post", 24, alpha_post, beta_post, backend);
    } else { // V4
        h = ggml_leaky_relu(ctx_graph, h, 0.1f, false);
    }

    struct ggml_tensor* conv_post_w = model.get_tensor("dec.conv_post.weight");
    h = nn::F::conv1d_no_transpose(ctx_graph, h, conv_post_w, nullptr, 1, 3, 1, 1, backend); // [out_seq_len, 1]

    struct ggml_tensor* audio = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, h));
    return ggml_tanh(ctx_graph, audio);
}

struct ggml_tensor* VITSModelCFM::forward_from_latent(
    nn::Context& context,
    struct ggml_tensor* latent,
    struct ggml_tensor* speaker_embedding,
    VITSRunState& state,
    ggml_backend_t backend
) {
    (void)state;
    struct ggml_context* ctx_graph = context.native_handle();
    return build_vits_generator_cfm(ctx_graph, latent, speaker_embedding, *this, backend);
}

static struct ggml_tensor* ggml_add_constant(struct ggml_context* ctx, struct ggml_tensor* a, float value) {
    nn::Context graph_context = nn::Context::borrow(ctx);
    struct ggml_tensor* c = graph_context.empty<float>("vits.scalar", {1});
    c = ggml_fill(ctx, c, value);
    return ggml_add(ctx, a, ggml_repeat(ctx, c, a));
}

struct ggml_tensor* interp_nearest_fractional(
    nn::Context& context,
    struct ggml_tensor* x,
    int64_t target_len,
    double scale_factor
) {
    struct ggml_context* ctx = context.native_handle();
    int64_t T = x->ne[1];
    std::vector<int32_t> indices_host(target_len);
    const double scale = scale_factor > 0.0 ? scale_factor : (double)target_len / T;
    for (int64_t i = 0; i < target_len; ++i) {
        int64_t src_idx = (int64_t)(i / scale);
        if (src_idx >= T) src_idx = T - 1;
        indices_host[i] = (int32_t)src_idx;
    }
    struct ggml_tensor* indices_tensor = context.constant<int32_t>(
        "vits.interp.indices", {target_len}, nn::data::copy(indices_host));
    return ggml_get_rows(ctx, x, indices_tensor);
}

struct ggml_tensor* interp_linear_fractional(
    nn::Context& context,
    struct ggml_tensor* x,
    int64_t target_len
) {
    struct ggml_context* ctx = context.native_handle();
    const int64_t source_len = x->ne[1];
    if (target_len <= 0 || source_len <= 0) {
        return nullptr;
    }
    if (target_len == source_len) {
        return x;
    }

    // Matches torch interpolate(mode="linear", align_corners=False).
    std::vector<int32_t> left_indices(target_len);
    std::vector<int32_t> right_indices(target_len);
    std::vector<float> left_weights(target_len);
    std::vector<float> right_weights(target_len);
    const double scale = static_cast<double>(source_len) / static_cast<double>(target_len);
    for (int64_t i = 0; i < target_len; ++i) {
        const double source_pos = std::max(0.0, (static_cast<double>(i) + 0.5) * scale - 0.5);
        const int64_t left = std::min<int64_t>(source_len - 1, static_cast<int64_t>(std::floor(source_pos)));
        const int64_t right = std::min<int64_t>(source_len - 1, left + 1);
        const float right_weight = static_cast<float>(std::max(0.0, source_pos - std::floor(source_pos)));
        left_indices[i] = static_cast<int32_t>(left);
        right_indices[i] = static_cast<int32_t>(right);
        right_weights[i] = right_weight;
        left_weights[i] = 1.0f - right_weight;
    }

    struct ggml_tensor* left_index_tensor = context.constant<int32_t>(
        "vits.interp.left_index", {target_len}, nn::data::copy(left_indices));
    struct ggml_tensor* right_index_tensor = context.constant<int32_t>(
        "vits.interp.right_index", {target_len}, nn::data::copy(right_indices));
    struct ggml_tensor* left_weight_tensor = context.constant<float>(
        "vits.interp.left_weight", {1, target_len}, nn::data::copy(left_weights));
    struct ggml_tensor* right_weight_tensor = context.constant<float>(
        "vits.interp.right_weight", {1, target_len}, nn::data::copy(right_weights));

    struct ggml_tensor* left_rows = ggml_get_rows(ctx, x, left_index_tensor);
    struct ggml_tensor* right_rows = ggml_get_rows(ctx, x, right_index_tensor);
    left_rows = ggml_mul(ctx, left_rows, ggml_repeat(ctx, left_weight_tensor, left_rows));
    right_rows = ggml_mul(ctx, right_rows, ggml_repeat(ctx, right_weight_tensor, right_rows));
    return ggml_add(ctx, left_rows, right_rows);
}



static struct ggml_tensor* build_timestep_embedding(
    nn::Context& context,
    float time_val,
    VITSModel& model,
    const std::string& prefix,
    int dim,
    int freq_embed_dim,
    ggml_backend_t backend
) {
    struct ggml_context* ctx = context.native_handle();
    (void)dim;
    std::vector<float> time_hidden_host(freq_embed_dim);
    int half_dim = freq_embed_dim / 2;
    float scale = 1000.0f;
    double emb_factor = std::log(10000.0) / (half_dim - 1);
    for (int i = 0; i < half_dim; ++i) {
        float emb = (float)std::exp(i * -emb_factor);
        float val = scale * time_val * emb;
        time_hidden_host[i] = std::sin(val);
        time_hidden_host[i + half_dim] = std::cos(val);
    }

    struct ggml_tensor* time_hidden = context.constant<float>(
        "vits.timestep", {freq_embed_dim}, nn::data::copy(time_hidden_host));

    struct ggml_tensor* w1 = model.get_tensor(prefix + "time_mlp.0.weight");
    struct ggml_tensor* b1 = model.get_tensor(prefix + "time_mlp.0.bias");
    struct ggml_tensor* h1 = nn::F::linear(ctx, time_hidden, w1, b1, backend);
    struct ggml_tensor* h1_act = ggml_silu(ctx, h1);

    struct ggml_tensor* w2 = model.get_tensor(prefix + "time_mlp.2.weight");
    struct ggml_tensor* b2 = model.get_tensor(prefix + "time_mlp.2.bias");
    struct ggml_tensor* out = nn::F::linear(ctx, h1_act, w2, b2, backend);

    return out;
}

static struct ggml_tensor* build_text_pos_embed(
    nn::Context& context,
    int64_t seq_len,
    int64_t text_dim
) {
    std::vector<float> freqs_cis_host(seq_len * text_dim);
    int64_t half_dim = text_dim / 2;
    double theta = 10000.0;
    std::vector<float> inv_freq(half_dim);
    for (int i = 0; i < half_dim; ++i) {
        inv_freq[i] = (float)(1.0 / std::pow(theta, (double)(2 * i) / text_dim));
    }
    for (int64_t t = 0; t < seq_len; ++t) {
        for (int64_t d = 0; d < half_dim; ++d) {
            float val = t * inv_freq[d];
            freqs_cis_host[t * text_dim + d] = std::cos(val);
            freqs_cis_host[t * text_dim + half_dim + d] = std::sin(val);
        }
    }
    return context.constant<float>(
        "vits.text_position", {text_dim, seq_len}, nn::data::copy(freqs_cis_host));
}

static struct ggml_tensor* build_convnextv2_block(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    VITSModel& model,
    const std::string& prefix,
    int dim,
    int intermediate_dim,
    ggml_backend_t backend
) {
    struct ggml_tensor* residual = x;
    struct ggml_tensor* dw_w = model.get_tensor(prefix + "dwconv.weight");
    struct ggml_tensor* dw_b = model.get_tensor(prefix + "dwconv.bias");
    struct ggml_tensor* x_transposed = ggml_cont(ctx, ggml_transpose(ctx, x));
    struct ggml_tensor* x_dw = ggml_ops_conv_1d(
        ctx, dw_w, x_transposed, 1, 3, 1, dim, backend, dw_b);
    if (!x_dw) return nullptr;
    x_dw = ggml_cont(ctx, ggml_transpose(ctx, x_dw));

    struct ggml_tensor* norm_w = model.get_tensor(prefix + "norm.weight");
    struct ggml_tensor* norm_b = model.get_tensor(prefix + "norm.bias");
    struct ggml_tensor* x_norm = ggml_norm(ctx, x_dw, 1e-6f);
    if (norm_w && norm_b) {
        x_norm = ggml_add(ctx, ggml_mul(ctx, x_norm, ggml_repeat(ctx, norm_w, x_norm)), ggml_repeat(ctx, norm_b, x_norm));
    }

    struct ggml_tensor* pw1_w = model.get_tensor(prefix + "pwconv1.weight");
    struct ggml_tensor* pw1_b = model.get_tensor(prefix + "pwconv1.bias");
    struct ggml_tensor* x_pw1 = nn::F::linear(ctx, x_norm, pw1_w, pw1_b, backend);

    struct ggml_tensor* x_act = ggml_gelu(ctx, x_pw1);

    struct ggml_tensor* grn_beta = model.get_tensor(prefix + "grn.beta");
    struct ggml_tensor* grn_gamma = model.get_tensor(prefix + "grn.gamma");
    if (grn_beta) {
        grn_beta = ggml_cast(ctx, grn_beta, GGML_TYPE_F32);
    }
    if (grn_gamma) {
        grn_gamma = ggml_cast(ctx, grn_gamma, GGML_TYPE_F32);
    }
    struct ggml_tensor* x_sq = ggml_sqr(ctx, x_act);
    struct ggml_tensor* x_sq_T = ggml_cont(ctx, ggml_transpose(ctx, x_sq));
    struct ggml_tensor* sum_T = ggml_sum_rows(ctx, x_sq_T);
    struct ggml_tensor* Gx = ggml_cont(ctx, ggml_transpose(ctx, ggml_sqrt(ctx, sum_T)));
    struct ggml_tensor* sum_Gx = ggml_sum_rows(ctx, Gx);
    struct ggml_tensor* mean_Gx = ggml_scale(ctx, sum_Gx, 1.0f / intermediate_dim);
    struct ggml_tensor* Nx = ggml_div(ctx, Gx, ggml_add_constant(ctx, mean_Gx, 1e-6f));
    struct ggml_tensor* x_Nx = ggml_mul(ctx, x_act, ggml_repeat(ctx, Nx, x_act));
    struct ggml_tensor* gamma_x_Nx = ggml_mul(ctx, x_Nx, ggml_repeat(ctx, grn_gamma, x_Nx));
    struct ggml_tensor* out = ggml_add(ctx, ggml_add(ctx, gamma_x_Nx, ggml_repeat(ctx, grn_beta, gamma_x_Nx)), x_act);

    struct ggml_tensor* pw2_w = model.get_tensor(prefix + "pwconv2.weight");
    struct ggml_tensor* pw2_b = model.get_tensor(prefix + "pwconv2.bias");
    struct ggml_tensor* x_pw2 = nn::F::linear(ctx, out, pw2_w, pw2_b, backend);

    return ggml_add(ctx, residual, x_pw2);
}

static struct ggml_tensor* build_group_conv_1d(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* weight,
    struct ggml_tensor* bias,
    int stride,
    int padding,
    int dilation,
    int groups,
    ggml_backend_t backend
) {
    struct ggml_tensor* x_transposed = ggml_cont(ctx, ggml_transpose(ctx, x));
    struct ggml_tensor* conv = ggml_ops_conv_1d(ctx, weight, x_transposed, stride, padding, dilation, groups, backend);
    struct ggml_tensor* conv_transposed = ggml_cont(ctx, ggml_transpose(ctx, conv));
    struct ggml_tensor* b_reshaped = ggml_reshape_2d(ctx, bias, bias->ne[0], 1);
    return ggml_add(ctx, conv_transposed, b_reshaped);
}

static struct ggml_tensor* build_adaln_zero_final(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* emb,
    VITSModel& model,
    const std::string& prefix,
    int dim,
    ggml_backend_t backend
) {
    struct ggml_tensor* emb_silu = ggml_silu(ctx, emb);
    struct ggml_tensor* linear_w = model.get_tensor(prefix + "linear.weight");
    struct ggml_tensor* linear_b = model.get_tensor(prefix + "linear.bias");
    struct ggml_tensor* emb_proj = nn::F::linear(ctx, emb_silu, linear_w, linear_b, backend);

    // AdaLayerNormZero_Final returns scale first, then shift (unlike AdaLayerNormZero).
    struct ggml_tensor* scale = ggml_view_2d(ctx, emb_proj, dim, 1, emb_proj->nb[1], 0);
    struct ggml_tensor* shift = ggml_view_2d(ctx, emb_proj, dim, 1, emb_proj->nb[1], dim * sizeof(float));

    struct ggml_tensor* scale_cont = ggml_cont(ctx, scale);
    struct ggml_tensor* shift_cont = ggml_cont(ctx, shift);

    struct ggml_tensor* x_norm = ggml_norm(ctx, x, 1e-6f);
    struct ggml_tensor* scale_plus_one = ggml_add_constant(ctx, scale_cont, 1.0f);
    struct ggml_tensor* scale_plus_one_cont = ggml_cont(ctx, scale_plus_one);
    struct ggml_tensor* out = ggml_add(ctx, ggml_mul(ctx, x_norm, ggml_repeat(ctx, scale_plus_one_cont, x_norm)), ggml_repeat(ctx, shift_cont, x_norm));

    return out;
}

static struct ggml_tensor* build_dit_estimator(
    struct ggml_context* ctx,
    struct ggml_tensor* x,          // [100, T_mel]
    struct ggml_tensor* prompt_x,   // [100, T_mel]
    struct ggml_tensor* t_cond,     // [1024, 1]
    struct ggml_tensor* text_embed,  // [512, T_mel]
    VITSModelCFM& model,
    ggml_backend_t backend,
    struct ggml_tensor* pos_tensor
) {
    struct ggml_tensor* input_cat = ggml_concat(ctx, ggml_concat(ctx, x, prompt_x, 0), text_embed, 0);
    struct ggml_tensor* h = model.input_proj(ctx, input_cat); // [1024, T_mel]

    struct ggml_tensor* conv1_w = model.get_tensor("cfm.estimator.input_embed.conv_pos_embed.conv1d.0.weight");
    struct ggml_tensor* conv1_b = model.get_tensor("cfm.estimator.input_embed.conv_pos_embed.conv1d.0.bias");
    struct ggml_tensor* conv2_w = model.get_tensor("cfm.estimator.input_embed.conv_pos_embed.conv1d.2.weight");
    struct ggml_tensor* conv2_b = model.get_tensor("cfm.estimator.input_embed.conv_pos_embed.conv1d.2.bias");
    
    struct ggml_tensor* h_conv = build_group_conv_1d(ctx, h, conv1_w, conv1_b, 1, 15, 1, 16, backend);
    h_conv = ggml_mul(ctx, h_conv, ggml_tanh(ctx, ggml_softplus(ctx, h_conv))); // Mish
    h_conv = build_group_conv_1d(ctx, h_conv, conv2_w, conv2_b, 1, 15, 1, 16, backend);
    h_conv = ggml_mul(ctx, h_conv, ggml_tanh(ctx, ggml_softplus(ctx, h_conv))); // Mish

    h = ggml_add(ctx, h, h_conv);

    for (int l = 0; l < 22; ++l) {
        h = model.transformer_blocks[l](ctx, h, t_cond, nullptr, nullptr, pos_tensor);
    }

    h = build_adaln_zero_final(ctx, h, t_cond, model, "cfm.estimator.norm_out.", 1024, backend);
    struct ggml_tensor* v_pred = model.proj_out(ctx, h); // [100, T]

    return v_pred;
}

struct ggml_tensor* VITSModelCFM::precompute_text_embeddings(
    nn::Context& context,
    struct ggml_tensor* cond_text,
    int T_mel,
    ggml_backend_t backend
) {
    struct ggml_context* ctx = context.native_handle();
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-CFM] Precomputing static text embeddings with ConvNeXtV2..." << std::endl;
    struct ggml_tensor* text_pos = build_text_pos_embed(context, T_mel, 512);
    struct ggml_tensor* text_embed = ggml_add(ctx, cond_text, text_pos);
    for (int l = 0; l < 4; ++l) {
        std::string block_p = "cfm.estimator.text_embed.text_blocks." + std::to_string(l) + ".";
        text_embed = build_convnextv2_block(ctx, text_embed, *this, block_p, 512, 1024, backend);
    }
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-CFM Debug] Completed ConvNeXtV2 static text embeddings" << std::endl;
    return text_embed;
}

FlowMatchingInputs VITSModelCFM::build_flow_matching_inputs(
    nn::Context& context,
    int T_mel,
    int prompt_len,
    int prompt_start,
    VITSRunState& state
) {
    // Generate initial noise x_init on host
    std::vector<float> x_host(100 * T_mel);
    std::random_device rd;
    uint32_t seed = rd();
    if (const char* seed_env = std::getenv("CFM_RANDOM_SEED")) {
        seed = static_cast<uint32_t>(std::strtoul(seed_env, nullptr, 10));
    }
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    float temperature = 1.0f;
    for (int i = 0; i < 100 * T_mel; ++i) {
        x_host[i] = dist(gen) * temperature;
    }
    for (int t = 0; t < prompt_len; ++t) {
        for (int c = 0; c < 100; ++c) {
            x_host[t * 100 + c] = 0.0f;
        }
    }
    struct ggml_tensor* x = context.constant<float>(
        "vits.cfm.noise", {100, T_mel}, nn::data::copy(x_host));

    // Construct prompt_x
    std::vector<float> prompt_x_host(100 * T_mel, 0.0f);
    if (prompt_len > 0 && !state.prompt_mel.empty()) {
        const float* sliced_mel_ptr = state.prompt_mel.data() + prompt_start * 100;
        int copy_len = prompt_len * 100;
        std::memcpy(prompt_x_host.data(), sliced_mel_ptr, copy_len * sizeof(float));
    }
    struct ggml_tensor* prompt_x = context.constant<float>(
        "vits.cfm.prompt", {100, T_mel}, nn::data::copy(prompt_x_host));

    // Construct prompt_mask
    std::vector<float> mask_host(T_mel, 1.0f);
    for (int t = 0; t < prompt_len; ++t) {
        mask_host[t] = 0.0f;
    }
    struct ggml_tensor* prompt_mask = context.constant<float>(
        "vits.cfm.prompt_mask", {1, T_mel}, nn::data::copy(mask_host));

    // Construct pos_tensor for RoPE
    std::vector<int32_t> pos_host(T_mel);
    for (int i = 0; i < T_mel; ++i) pos_host[i] = i;
    struct ggml_tensor* pos_tensor = context.constant<int32_t>(
        "vits.cfm.position", {T_mel}, nn::data::copy(pos_host));

    FlowMatchingInputs inputs;
    inputs.x = x;
    inputs.prompt_x = prompt_x;
    inputs.prompt_mask = prompt_mask;
    inputs.pos_tensor = pos_tensor;
    return inputs;
}

struct ggml_tensor* VITSModelCFM::run_ode_loop(
    nn::Context& context,
    const FlowMatchingInputs& inputs,
    struct ggml_tensor* text_embed,
    ggml_backend_t backend
) {
    struct ggml_context* ctx = context.native_handle();
    int n_timesteps = this->cfm_steps;
    if (n_timesteps < 1) {
        n_timesteps = 8;
    }
    float dt = 1.0f / n_timesteps;
    struct ggml_tensor* x = inputs.x;

    // Precompute static timestep embeddings (TS optimization)
    struct ggml_tensor* d_emb = build_timestep_embedding(context, dt, *this, "cfm.estimator.d_embed.", 1024, 256, backend);
    std::vector<struct ggml_tensor*> t_cond_list(n_timesteps);
    for (int j = 0; j < n_timesteps; ++j) {
        float t_val = std::max(0.0f, std::min(1.0f, (float)j * dt));
        struct ggml_tensor* t_emb = build_timestep_embedding(context, t_val, *this, "cfm.estimator.time_embed.", 1024, 256, backend);
        t_cond_list[j] = ggml_add(ctx, t_emb, d_emb);
    }

    for (int j = 0; j < n_timesteps; ++j) {
        struct ggml_tensor* v_pred = build_dit_estimator(ctx, x, inputs.prompt_x, t_cond_list[j], text_embed, *this, backend, inputs.pos_tensor);
        x = ggml_add(ctx, x, ggml_scale(ctx, v_pred, dt));
        x = ggml_mul(ctx, x, ggml_repeat(ctx, inputs.prompt_mask, x));
    }
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-CFM Debug] Completed ODE Euler loop" << std::endl;
    return x;
}

struct ggml_tensor* VITSModelCFM::forward(
    nn::Context& context,
    struct ggml_tensor* phone_ids,
    struct ggml_tensor* phone_lengths,
    struct ggml_tensor* word2ph,
    struct ggml_tensor* bert_features,
    struct ggml_tensor* prompt_semantics,
    struct ggml_tensor* refer_audio,
    VITSRunState& state,
    float speed,
    ggml_backend_t backend
) {
    struct ggml_context* ctx_graph = context.native_handle();
    EncodeResult res = encode_semantic_base(context, phone_ids, prompt_semantics, refer_audio, backend);
    struct ggml_tensor* y2 = res.y2;
    struct ggml_tensor* ge = res.ge;
    int T_y = res.T_y;

    if (!y2) {
        return nullptr;
    }

    // Bridge projection: 192 -> 512
    struct ggml_tensor* bridge_w = get_tensor("bridge.0.weight");
    struct ggml_tensor* bridge_b = get_tensor("bridge.0.bias");
    if (speed > 0.0f && speed != 1.0f) {
        const int64_t speed_adjusted_len = static_cast<int64_t>(T_y / speed) + 1;
        y2 = interp_linear_fractional(context, y2, speed_adjusted_len);
        if (!y2) {
            return nullptr;
        }
        T_y = static_cast<int>(speed_adjusted_len);
    }

    struct ggml_tensor* fea = nn::F::conv1d(ctx_graph, y2, bridge_w, bridge_b, 1, 0, 1, 1, backend);
    fea = ggml_leaky_relu(ctx_graph, fea, 0.01f, false);
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-CFM Debug] Completed Bridge Proj" << std::endl;

    // Interpolate nearest-neighbor to target Mel spectrogram frame rate
    const double feature_scale = profile.feature_rate_scale;
    const int64_t target_len = static_cast<int64_t>(T_y * feature_scale);
    fea = interp_nearest_fractional(context, fea, target_len, feature_scale);
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-CFM Debug] Completed Interp Fractional (len=" << target_len << ")" << std::endl;

    struct ggml_tensor* cond_text = wns1.forward(ctx_graph, fea, ge, backend);
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-CFM Debug] Completed WNS1 Encoder" << std::endl;

    int prompt_len = 0;
    int prompt_start = 0;
    int mel_frames = !state.prompt_mel.empty()
        ? static_cast<int>(state.prompt_mel.size() / profile.prompt_mel_channels) : 0;
    int fea_ref_frames = !state.prompt_features.empty() ? (int)(state.prompt_features.size() / 512) : 0;
    
    int T_min = 0;
    if (mel_frames > 0 && fea_ref_frames > 0) {
        T_min = std::min(mel_frames, fea_ref_frames);
        int Tref = profile.max_prompt_frames;
        if (T_min > Tref) {
            prompt_start = T_min - Tref;
            T_min = Tref;
        }
        prompt_len = T_min;
    }
    int T_mel = prompt_len + (int)target_len;

    struct ggml_tensor* cond_text_padded = cond_text;
    if (prompt_len > 0) {
        const float* sliced_fea_ptr = state.prompt_features.data() + prompt_start * 512;
        struct ggml_tensor* prompt_fea_ref_tensor = context.constant<float>(
            "vits.cfm.prompt_features", {512, prompt_len},
            nn::data::copy(sliced_fea_ptr, static_cast<size_t>(512 * prompt_len)));

        cond_text_padded = ggml_concat(ctx_graph, prompt_fea_ref_tensor, cond_text, 1);
    }

    // Compute static text embeddings + ConvNeXtV2 blocks once before the loop
    struct ggml_tensor* text_embed = precompute_text_embeddings(context, cond_text_padded, T_mel, backend);

    // Build flow matching inputs
    FlowMatchingInputs inputs = build_flow_matching_inputs(context, T_mel, prompt_len, prompt_start, state);

    // Run ODE Euler loop
    struct ggml_tensor* x = run_ode_loop(context, inputs, text_embed, backend);
    // Denormalize Mel spectrogram back to linear range: (x + 1)/2 * 14 - 12
    struct ggml_tensor* cfm_res_denorm = ggml_add_constant(ctx_graph, ggml_scale(ctx_graph, ggml_add_constant(ctx_graph, x, 1.0f), 7.0f), -12.0f);

    struct ggml_tensor* target_mel = cfm_res_denorm;
    if (prompt_len > 0) {
        target_mel = ggml_view_2d(ctx_graph, cfm_res_denorm, 100, (int)target_len, cfm_res_denorm->nb[1], prompt_len * cfm_res_denorm->nb[1]);
    }

    // Feed to final BigVGAN / HiFi-GAN generator vocoder
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-CFM Debug] Running Final Generator Vocoder" << std::endl;
    return build_vits_generator_cfm(ctx_graph, target_mel, ge, *this, backend);
}

} // namespace gpt_sovits
