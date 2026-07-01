#include "vits.h"
#include "ops/ops.h"
#include <iostream>
#include <random>
#include <cstring>
#include <vector>
#include <cmath>

namespace gpt_sovits {

struct ggml_tensor* ggml_snake_beta(
    struct ggml_context* ctx,
    struct ggml_tensor* x,          // [seq_len, channels]
    struct ggml_tensor* alpha,      // [channels]
    struct ggml_tensor* beta        // [channels]
) {
    int64_t C = x->ne[1]; // channels
    int64_t T = x->ne[0]; // seq_len

    struct ggml_tensor* alpha_f32 = force_w_f32(ctx, alpha);
    struct ggml_tensor* beta_f32 = force_w_f32(ctx, beta);

    struct ggml_tensor* alpha_2d = ggml_reshape_2d(ctx, alpha_f32, 1, C);
    struct ggml_tensor* alpha_repeated = ggml_repeat(ctx, alpha_2d, x);

    struct ggml_tensor* beta_2d = ggml_reshape_2d(ctx, beta_f32, 1, C);
    struct ggml_tensor* beta_repeated = ggml_repeat(ctx, beta_2d, x);

    // x_alpha = x * alpha
    struct ggml_tensor* x_alpha = ggml_mul(ctx, x, alpha_repeated);

    // sin_val = sin(x_alpha)
    struct ggml_tensor* sin_val = ggml_sin(ctx, x_alpha);

    // sin_sq = sin_val * sin_val
    struct ggml_tensor* sin_sq = ggml_mul(ctx, sin_val, sin_val);

    // term = sin_sq / beta_repeated
    struct ggml_tensor* term = ggml_div(ctx, sin_sq, beta_repeated);

    // result = x + term
    return ggml_add(ctx, x, term);
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

    struct ggml_tensor* up_filter_rep = model.get_tensor(act_prefix + ".upsample.filter_repeated");
    struct ggml_tensor* upsampled = ggml_ops_conv_transpose_1d(ctx, up_filter_rep, x, 2, 5, 1, channels, backend);

    struct ggml_tensor* act_out = ggml_snake_beta(ctx, upsampled, alpha, beta);

    struct ggml_tensor* down_filter_rep = model.get_tensor(act_prefix + ".downsample.lowpass.filter_repeated");
    struct ggml_tensor* downsampled = ggml_ops_conv_1d(ctx, down_filter_rep, act_out, 2, 5, 1, channels, backend);

    return downsampled;
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
        xt = ggml_conv_1d_with_bias_no_transpose(ctx, xt, c1_w, c1_b, 1, dilation, padding, backend);
        // xt = a2(xt)
        xt = alias_free_activation(ctx, xt, model, block_idx, 2 * l + 1, channels, alpha2, beta2, backend);
        // xt = c2(xt)
        xt = ggml_conv_1d_with_bias_no_transpose(ctx, xt, c2_w, c2_b, 1, 1, padding2, backend);

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

        struct ggml_tensor* c1_w = nullptr;
        int dilation_effective = dilation;
        if (dilation > 1) {
            c1_w = model.get_tensor(prefix1 + ".weight_dilated");
            if (c1_w) {
                dilation_effective = 1;
            } else {
                c1_w = model.get_tensor(prefix1 + ".weight");
            }
        } else {
            c1_w = model.get_tensor(prefix1 + ".weight");
        }
        
        struct ggml_tensor* c1_b = model.get_tensor(prefix1 + ".bias");
        struct ggml_tensor* c2_w = model.get_tensor(prefix2 + ".weight");
        struct ggml_tensor* c2_b = model.get_tensor(prefix2 + ".bias");

        if (!c1_w || !c1_b || !c2_w || !c2_b) {
            continue;
        }

        struct ggml_tensor* xt = ggml_leaky_relu(ctx, current_x, 0.1f, false);
        xt = ggml_conv_1d_with_bias_no_transpose(ctx, xt, c1_w, c1_b, 1, dilation_effective, padding, backend);
        xt = ggml_leaky_relu(ctx, xt, 0.1f, false);
        xt = ggml_conv_1d_with_bias_no_transpose(ctx, xt, c2_w, c2_b, 1, 1, (kernel_size - 1) / 2, backend);

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
    struct ggml_tensor* h = ggml_conv_1d_with_bias_no_transpose(ctx_graph, latent_transposed, dec_conv_pre_w, dec_conv_pre_b, 1, 1, 3, backend);

    std::vector<int> upsample_rates;
    std::vector<int> upsample_kernel_sizes;
    std::vector<int> upsample_channels;

    if (model.version == 3) {
        upsample_rates = {4, 4, 2, 2, 2, 2};
        upsample_kernel_sizes = {8, 8, 4, 4, 4, 4};
        upsample_channels = {768, 384, 192, 96, 48, 24};
    } else { // V4
        upsample_rates = {10, 6, 2, 2, 2};
        upsample_kernel_sizes = {20, 12, 4, 4, 4};
        upsample_channels = {256, 128, 64, 32, 16};
    }

    const std::vector<int> resblock_kernel_sizes = {3, 7, 11};
    const std::vector<int> dilation_sizes = {1, 3, 5};

    for (size_t i = 0; i < upsample_rates.size(); ++i) {
        int stride = upsample_rates[i];
        int kernel_size = upsample_kernel_sizes[i];
        int padding = (kernel_size - stride) / 2;
        int out_channels = upsample_channels[i];

        std::string ups_prefix = "dec.ups." + std::to_string(i) + ".0.";
        struct ggml_tensor* ups_w = model.get_tensor(ups_prefix + "weight");
        struct ggml_tensor* ups_b = model.get_tensor(ups_prefix + "bias");

        if (model.version == 4) {
            h = ggml_leaky_relu(ctx_graph, h, 0.1f, false);
        }

        h = ggml_conv_transpose_1d_with_bias_no_transpose(ctx_graph, h, ups_w, ups_b, stride, padding, backend);

        struct ggml_tensor* res_x = nullptr;
        for (int j = 0; j < 3; ++j) {
            int block_idx = i * 3 + j;
            int r_kernel_size = resblock_kernel_sizes[j];
            
            struct ggml_tensor* block_out = nullptr;
            if (model.version == 3) {
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

    if (model.version == 3) {
        struct ggml_tensor* alpha_post = model.get_tensor("dec.activation_post.act.alpha");
        struct ggml_tensor* beta_post = model.get_tensor("dec.activation_post.act.beta");
        h = ggml_snake_beta(ctx_graph, h, alpha_post, beta_post);
    } else { // V4
        h = ggml_leaky_relu(ctx_graph, h, 0.1f, false);
    }

    struct ggml_tensor* conv_post_w = model.get_tensor("dec.conv_post.weight");
    h = ggml_conv_1d_vits(ctx_graph, conv_post_w, h, 1, 3, 1, backend); // [out_seq_len, 1]

    struct ggml_tensor* audio = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, h));
    return ggml_tanh(ctx_graph, audio);
}

struct ggml_tensor* VITSModelCFM::forward_from_latent(
    struct ggml_context* ctx_graph,
    struct ggml_tensor* latent,
    struct ggml_tensor* speaker_embedding,
    ggml_backend_t backend
) {
    clear_conv_1d_params_pool();
    return build_vits_generator_cfm(ctx_graph, latent, speaker_embedding, *this, backend);
}

static struct ggml_tensor* ggml_add_constant(struct ggml_context* ctx, struct ggml_tensor* a, float value) {
    struct ggml_tensor* c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    c = ggml_fill(ctx, c, value);
    return ggml_add(ctx, a, ggml_repeat(ctx, c, a));
}

static struct ggml_tensor* interp_nearest_fractional(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    int64_t target_len,
    VITSModel& model
) {
    int64_t C = x->ne[0];
    int64_t T = x->ne[1];
    struct ggml_tensor* indices_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, target_len);
    std::vector<int32_t> indices_host(target_len);
    double scale = (double)target_len / T;
    for (int64_t i = 0; i < target_len; ++i) {
        int64_t src_idx = (int64_t)(i / scale);
        if (src_idx >= T) src_idx = T - 1;
        indices_host[i] = (int32_t)src_idx;
    }
    std::vector<uint8_t> raw_bytes(target_len * sizeof(int32_t));
    std::memcpy(raw_bytes.data(), indices_host.data(), raw_bytes.size());
    model.upload_entries.push_back({ indices_tensor, raw_bytes });
    return ggml_get_rows(ctx, x, indices_tensor);
}

static struct ggml_tensor* build_wn_encoder(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* g,
    VITSModel& model,
    const std::string& prefix,
    int in_channels,
    int out_channels,
    int hidden_channels,
    int kernel_size,
    int dilation_rate,
    int n_layers,
    ggml_backend_t backend
) {
    struct ggml_tensor* pre_w = model.get_tensor(prefix + "pre.weight");
    struct ggml_tensor* pre_b = model.get_tensor(prefix + "pre.bias");
    struct ggml_tensor* h = ggml_conv_1d_with_bias(ctx, x, pre_w, pre_b, 1, 1, 0, backend);

    struct ggml_tensor* g_proj = nullptr;
    if (g) {
        struct ggml_tensor* cond_w = model.get_tensor(prefix + "enc.cond_layer.weight");
        struct ggml_tensor* cond_b = model.get_tensor(prefix + "enc.cond_layer.bias");
        g_proj = ggml_conv_1d_with_bias(ctx, g, cond_w, cond_b, 1, 1, 0, backend);
    }

    struct ggml_tensor* WN_out = build_wn(ctx, h, nullptr, g_proj, model, prefix + "enc.",
                                          hidden_channels, kernel_size, dilation_rate, n_layers, backend);

    struct ggml_tensor* proj_w = model.get_tensor(prefix + "proj.weight");
    struct ggml_tensor* proj_b = model.get_tensor(prefix + "proj.bias");
    struct ggml_tensor* out = ggml_conv_1d_with_bias(ctx, WN_out, proj_w, proj_b, 1, 1, 0, backend);

    return out;
}

static struct ggml_tensor* build_timestep_embedding(
    struct ggml_context* ctx,
    float time_val,
    VITSModel& model,
    const std::string& prefix,
    int dim,
    int freq_embed_dim,
    ggml_backend_t backend
) {
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

    struct ggml_tensor* time_hidden = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, freq_embed_dim);
    std::vector<uint8_t> raw_bytes(freq_embed_dim * sizeof(float));
    std::memcpy(raw_bytes.data(), time_hidden_host.data(), raw_bytes.size());
    model.upload_entries.push_back({ time_hidden, raw_bytes });

    struct ggml_tensor* w1 = model.get_tensor(prefix + "time_mlp.0.weight");
    struct ggml_tensor* b1 = model.get_tensor(prefix + "time_mlp.0.bias");
    struct ggml_tensor* h1 = ggml_linear(ctx, time_hidden, w1, b1);
    struct ggml_tensor* h1_act = ggml_silu(ctx, h1);

    struct ggml_tensor* w2 = model.get_tensor(prefix + "time_mlp.2.weight");
    struct ggml_tensor* b2 = model.get_tensor(prefix + "time_mlp.2.bias");
    struct ggml_tensor* out = ggml_linear(ctx, h1_act, w2, b2);

    return out;
}

static struct ggml_tensor* build_text_pos_embed(
    struct ggml_context* ctx,
    int64_t seq_len,
    int64_t text_dim,
    VITSModel& model,
    ggml_backend_t backend
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
    struct ggml_tensor* pos_embed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, text_dim, seq_len);
    std::vector<uint8_t> raw_bytes(seq_len * text_dim * sizeof(float));
    std::memcpy(raw_bytes.data(), freqs_cis_host.data(), raw_bytes.size());
    model.upload_entries.push_back({ pos_embed, raw_bytes });
    return pos_embed;
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
    if (GPT_SOVITS_DEBUG_ENABLED()) {
        std::cout << "[build_convnextv2_block] prefix: " << prefix 
                  << ", x: " << x << " (" << x->ne[0] << ", " << x->ne[1] << ", " << x->ne[2] << ", " << x->ne[3] << ")"
                  << ", dw_w: " << dw_w << " (" << dw_w->ne[0] << ", " << dw_w->ne[1] << ", " << dw_w->ne[2] << ", " << dw_w->ne[3] << ")"
                  << ", dw_b: " << dw_b << std::endl;
    }
    struct ggml_tensor* x_transposed = ggml_cont(ctx, ggml_transpose(ctx, x));
    struct ggml_tensor* x_dw = ggml_conv_1d_dw(ctx, dw_w, x_transposed, 1, 3, 1);
    x_dw = ggml_cont(ctx, ggml_transpose(ctx, x_dw));
    x_dw = ggml_add(ctx, x_dw, ggml_repeat(ctx, dw_b, x_dw));

    struct ggml_tensor* norm_w = model.get_tensor(prefix + "norm.weight");
    struct ggml_tensor* norm_b = model.get_tensor(prefix + "norm.bias");
    struct ggml_tensor* x_norm = ggml_norm(ctx, x_dw, 1e-6f);
    if (norm_w && norm_b) {
        x_norm = ggml_add(ctx, ggml_mul(ctx, x_norm, ggml_repeat(ctx, norm_w, x_norm)), ggml_repeat(ctx, norm_b, x_norm));
    }

    struct ggml_tensor* pw1_w = model.get_tensor(prefix + "pwconv1.weight");
    struct ggml_tensor* pw1_b = model.get_tensor(prefix + "pwconv1.bias");
    struct ggml_tensor* x_pw1 = ggml_linear(ctx, x_norm, pw1_w, pw1_b);

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
    struct ggml_tensor* x_pw2 = ggml_linear(ctx, out, pw2_w, pw2_b);

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
    struct ggml_tensor* emb_proj = ggml_linear(ctx, emb_silu, linear_w, linear_b);

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
    float t_val,
    float d_val,
    struct ggml_tensor* text_embed,  // [512, T_mel]
    VITSModel& model,
    ggml_backend_t backend,
    struct ggml_tensor* pos_tensor
) {
    t_val = std::max(0.0f, std::min(1.0f, t_val)); // Clamp for stability
    struct ggml_tensor* t_emb = build_timestep_embedding(ctx, t_val, model, "cfm.estimator.time_embed.", 1024, 256, backend);
    struct ggml_tensor* d_emb = build_timestep_embedding(ctx, d_val, model, "cfm.estimator.d_embed.", 1024, 256, backend);
    struct ggml_tensor* t_cond = ggml_add(ctx, t_emb, d_emb); // [1024, 1]

    struct ggml_tensor* input_cat = ggml_concat(ctx, ggml_concat(ctx, x, prompt_x, 0), text_embed, 0);
    struct ggml_tensor* proj_w = model.get_tensor("cfm.estimator.input_embed.proj.weight");
    struct ggml_tensor* proj_b = model.get_tensor("cfm.estimator.input_embed.proj.bias");
    struct ggml_tensor* h = ggml_linear(ctx, input_cat, proj_w, proj_b); // [1024, T_mel]

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
        std::string layer_p = "cfm.estimator.transformer_blocks." + std::to_string(l) + ".";
        
        struct ggml_tensor* attn_norm_w = model.get_tensor(layer_p + "attn_norm.linear.weight");
        struct ggml_tensor* attn_norm_b = model.get_tensor(layer_p + "attn_norm.linear.bias");
        
        struct ggml_tensor* qw = model.get_tensor(layer_p + "attn.to_q.weight");
        struct ggml_tensor* qb = model.get_tensor(layer_p + "attn.to_q.bias");
        struct ggml_tensor* kw = model.get_tensor(layer_p + "attn.to_k.weight");
        struct ggml_tensor* kb = model.get_tensor(layer_p + "attn.to_k.bias");
        struct ggml_tensor* vw = model.get_tensor(layer_p + "attn.to_v.weight");
        struct ggml_tensor* vb = model.get_tensor(layer_p + "attn.to_v.bias");
        struct ggml_tensor* ow = model.get_tensor(layer_p + "attn.to_out.0.weight");
        struct ggml_tensor* ob = model.get_tensor(layer_p + "attn.to_out.0.bias");

        struct ggml_tensor* ffn_w1 = model.get_tensor(layer_p + "ff.ff.0.0.weight");
        struct ggml_tensor* ffn_b1 = model.get_tensor(layer_p + "ff.ff.0.0.bias");
        struct ggml_tensor* ffn_w2 = model.get_tensor(layer_p + "ff.ff.2.weight");
        struct ggml_tensor* ffn_b2 = model.get_tensor(layer_p + "ff.ff.2.bias");

        nn::DiTBlock block(
            attn_norm_w, attn_norm_b, 1e-6f,
            qw, qb, kw, kb, vw, vb, ow, ob,
            16, 64, // heads=16, dim_head=64
            nullptr, nullptr, 1e-6f, // ff_norm
            ffn_w1, ffn_b1, ffn_w2, ffn_b2,
            nn::ActivationType::GELU
        );

        h = block.forward(ctx, h, t_cond, nullptr, backend, pos_tensor);
    }

    h = build_adaln_zero_final(ctx, h, t_cond, model, "cfm.estimator.norm_out.", 1024, backend);
    
    struct ggml_tensor* proj_out_w = model.get_tensor("cfm.estimator.proj_out.weight");
    struct ggml_tensor* proj_out_b = model.get_tensor("cfm.estimator.proj_out.bias");
    struct ggml_tensor* v_pred = ggml_linear(ctx, h, proj_out_w, proj_out_b); // [100, T]

    return v_pred;
}

struct ggml_tensor* VITSModelCFM::forward(
    struct ggml_context* ctx_graph,
    struct ggml_tensor* phone_ids,
    struct ggml_tensor* phone_lengths,
    struct ggml_tensor* word2ph,
    struct ggml_tensor* bert_features,
    struct ggml_tensor* prompt_semantics,
    struct ggml_tensor* refer_audio,
    float speed,
    ggml_backend_t backend
) {
    clear_conv_1d_params_pool();
    (void)phone_lengths;
    (void)word2ph;
    (void)bert_features;
    (void)refer_audio;

    int semantic_len = (int)prompt_semantics->ne[0];
    if (GPT_SOVITS_DEBUG_ENABLED()) {
        std::cout << "[VITS-CFM] Inference Graph - semantic_len: " << semantic_len
                  << ", speed: " << speed << std::endl;
    }

    // Step 1: VQ Decode - semantic token IDs -> continuous features [768, N]
    struct ggml_tensor* decoded = vq_decode(ctx_graph, prompt_semantics, *this);
    if (!decoded) {
        std::cerr << "[VITS-CFM] Error: VQ decode failed!" << std::endl;
        return nullptr;
    }
    debug_decoded = decoded;

    // Step 2: Interpolate from 25Hz to 50Hz (2x nearest-neighbor)
    struct ggml_tensor* interp = interp_nearest_2x(ctx_graph, decoded, *this);
    int T_y = (int)interp->ne[1];
    debug_interp = interp;

    // Step 3: SSL Projection - 768 -> 192 channels via enc_p.ssl_proj
    struct ggml_tensor* ssl_proj_w = get_tensor("enc_p.ssl_proj.weight");
    struct ggml_tensor* ssl_proj_b = get_tensor("enc_p.ssl_proj.bias");
    struct ggml_tensor* y = interp;
    if (ssl_proj_w && ssl_proj_b) {
        y = ggml_conv_1d_with_bias(ctx_graph, interp, ssl_proj_w, ssl_proj_b, 1, 1, 0, backend);
        debug_ssl_proj = y;
    }

    // Step 4: Load speaker embedding (ge)
    struct ggml_tensor* ge = refer_audio;
    if (ge) {
        int64_t ge_size = ggml_nelements(ge);
        ge = ggml_reshape_2d(ctx_graph, ge, ge_size, 1);
    } else {
        int64_t ge_dim = 512;
        ge = ggml_new_tensor_2d(ctx_graph, GGML_TYPE_F32, ge_dim, 1);
        ge = ggml_fill(ctx_graph, ge, 0.0f);
    }

    struct ggml_tensor* ge_512 = ge;
    struct ggml_tensor* ge_to512_w = get_tensor("ge_to512.weight");
    struct ggml_tensor* ge_to512_b = get_tensor("ge_to512.bias");
    if (ge_to512_w && ge_to512_b && ge) {
        ge_512 = ggml_linear(ctx_graph, ge, ge_to512_w, ge_to512_b);
    }
    int n_head = 2;
    int d_k = 96;  // 192 / 2

    // Step 5: encoder_ssl (3 layers) on ssl features
    struct ggml_tensor* y_enc = build_encoder(ctx_graph, y, nullptr, *this, "enc_p.encoder_ssl", 3, n_head, d_k, T_y, backend);
    debug_enc_ssl_out = y_enc;

    // Step 6: encoder_text (6 layers) on phone embeddings
    struct ggml_tensor* text_emb_w = get_tensor("enc_p.text_embedding.weight");
    int text_len = (int)phone_ids->ne[0];
    struct ggml_tensor* text_emb = ggml_get_rows(ctx_graph, text_emb_w, phone_ids);  // [192, text_len]

    struct ggml_tensor* text_enc = build_encoder(ctx_graph, text_emb, nullptr, *this, "enc_p.encoder_text", 6, n_head, d_k, text_len, backend);
    debug_enc_text_out = text_enc;

    // Step 7: MRTE - cross-attention between y_enc and text_enc with speaker conditioning
    struct ggml_tensor* mrte_out = build_mrte(ctx_graph, y_enc, nullptr, text_enc, nullptr, ge_512, *this, backend);
    debug_enc_mrte_out = mrte_out;

    // Step 8: encoder2 (3 layers)
    struct ggml_tensor* y2 = build_encoder(ctx_graph, mrte_out, nullptr, *this, "enc_p.encoder2", 3, n_head, d_k, T_y, backend);
    debug_enc_enc2_out = y2;

    // Bridge projection: 192 -> 512
    struct ggml_tensor* bridge_w = get_tensor("bridge.0.weight");
    struct ggml_tensor* bridge_b = get_tensor("bridge.0.bias");
    struct ggml_tensor* fea = ggml_conv_1d_with_bias(ctx_graph, y2, bridge_w, bridge_b, 1, 1, 0, backend);
    fea = ggml_leaky_relu(ctx_graph, fea, 0.01f, false);

    // Interpolate nearest-neighbor to target Mel spectrogram frame rate
    int64_t target_len = T_y * 2;
    if (this->version == 3) {
        target_len = (int64_t)(T_y * 1.875);
    }
    fea = interp_nearest_fractional(ctx_graph, fea, target_len, *this);

    // WNS1: WaveNet-style encoder -> cond_text
    struct ggml_tensor* cond_text = build_wn_encoder(ctx_graph, fea, ge, *this, "wns1.",
                                                     512, 512, 512, 5, 1, 8, backend);

    // Flow Matching ODE loop preparation
    int T_mel = (int)target_len;
    int prompt_len = 0;
    if (prompt_mel.tensor != nullptr && !this->prompt_mel_host.empty()) {
        prompt_len = (int)(this->prompt_mel_host.size() / 100);
    }
    prompt_len = std::min(prompt_len, T_mel);

    // Generate initial noise x_init on host
    std::vector<float> x_host(100 * T_mel);
    std::random_device rd;
    std::mt19937 gen(rd());
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
    struct ggml_tensor* x = ggml_new_tensor_2d(ctx_graph, GGML_TYPE_F32, 100, T_mel);
    std::vector<uint8_t> x_bytes(100 * T_mel * sizeof(float));
    std::memcpy(x_bytes.data(), x_host.data(), x_bytes.size());
    this->upload_entries.push_back({ x, x_bytes });

    // Construct prompt_x
    std::vector<float> prompt_x_host(100 * T_mel, 0.0f);
    if (prompt_mel.tensor != nullptr && !this->prompt_mel_host.empty()) {
        int copy_len = std::min(prompt_len * 100, (int)this->prompt_mel_host.size());
        std::memcpy(prompt_x_host.data(), this->prompt_mel_host.data(), copy_len * sizeof(float));
    }
    struct ggml_tensor* prompt_x = ggml_new_tensor_2d(ctx_graph, GGML_TYPE_F32, 100, T_mel);
    std::vector<uint8_t> px_bytes(100 * T_mel * sizeof(float));
    std::memcpy(px_bytes.data(), prompt_x_host.data(), px_bytes.size());
    this->upload_entries.push_back({ prompt_x, px_bytes });

    // Construct prompt_mask
    std::vector<float> mask_host(T_mel, 1.0f);
    for (int t = 0; t < prompt_len; ++t) {
        mask_host[t] = 0.0f;
    }
    struct ggml_tensor* prompt_mask = ggml_new_tensor_2d(ctx_graph, GGML_TYPE_F32, 1, T_mel);
    std::vector<uint8_t> mask_bytes(T_mel * sizeof(float));
    std::memcpy(mask_bytes.data(), mask_host.data(), mask_bytes.size());
    this->upload_entries.push_back({ prompt_mask, mask_bytes });

    // Construct pos_tensor for RoPE
    struct ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx_graph, GGML_TYPE_I32, T_mel);
    std::vector<int32_t> pos_host(T_mel);
    for (int i = 0; i < T_mel; ++i) pos_host[i] = i;
    std::vector<uint8_t> pos_bytes(T_mel * sizeof(int32_t));
    std::memcpy(pos_bytes.data(), pos_host.data(), pos_bytes.size());
    this->upload_entries.push_back({ pos_tensor, pos_bytes });

    // Compute static text embeddings + ConvNeXtV2 blocks once before the loop
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-CFM] Precomputing static text embeddings with ConvNeXtV2..." << std::endl;
    struct ggml_tensor* text_pos = build_text_pos_embed(ctx_graph, T_mel, 512, *this, backend);
    struct ggml_tensor* text_embed = ggml_add(ctx_graph, cond_text, text_pos);
    for (int l = 0; l < 4; ++l) {
        std::string block_p = "cfm.estimator.text_embed.text_blocks." + std::to_string(l) + ".";
        text_embed = build_convnextv2_block(ctx_graph, text_embed, *this, block_p, 512, 1024, backend);
    }

    // ODE Euler loop (32 steps)
    int n_timesteps = 32;
    float dt = 1.0f / n_timesteps;
    for (int j = 0; j < n_timesteps; ++j) {
        float t_val = j * dt;
        struct ggml_tensor* v_pred = build_dit_estimator(ctx_graph, x, prompt_x, t_val, dt, text_embed, *this, backend, pos_tensor);
        x = ggml_add(ctx_graph, x, ggml_scale(ctx_graph, v_pred, dt));
        x = ggml_mul(ctx_graph, x, ggml_repeat(ctx_graph, prompt_mask, x));
    }

    // Denormalize Mel spectrogram back to linear range: (x + 1)/2 * 14 - 12
    struct ggml_tensor* cfm_res_denorm = ggml_add_constant(ctx_graph, ggml_scale(ctx_graph, ggml_add_constant(ctx_graph, x, 1.0f), 7.0f), -12.0f);

    // Feed to final BigVGAN / HiFi-GAN generator vocoder
    return build_vits_generator_cfm(ctx_graph, cfm_res_denorm, ge, *this, backend);
}

} // namespace gpt_sovits
