#include "vits.h"
#include "ops/ops.h"
#include "nn/nn.h"
#include "ggml.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace gpt_sovits {

// Local helpers implemented in this translation unit
static struct ggml_tensor* mrf_resblock_no_transpose(
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

        struct ggml_tensor* c1_w = model.gen_weights.resblocks[block_idx][l].c1_w;
        struct ggml_tensor* c1_b = model.gen_weights.resblocks[block_idx][l].c1_b;
        struct ggml_tensor* c2_w = model.gen_weights.resblocks[block_idx][l].c2_w;
        struct ggml_tensor* c2_b = model.gen_weights.resblocks[block_idx][l].c2_b;

        if (!c1_w || !c1_b || !c2_w || !c2_b) {
            continue;
        }

        struct ggml_tensor* xt = ggml_leaky_relu(ctx, current_x, 0.1f, false);
        xt = nn::F::conv1d_no_transpose(ctx, xt, c1_w, c1_b, 1, padding, dilation, 1, backend);
        xt = ggml_leaky_relu(ctx, xt, 0.1f, false);
        xt = nn::F::conv1d_no_transpose(ctx, xt, c2_w, c2_b, 1, (kernel_size - 1) / 2, 1, 1, backend);

        current_x = ggml_add(ctx, xt, current_x);
    }

    return current_x;
}

static struct ggml_tensor* conv1d_glu(
    struct ggml_context* ctx,
    struct ggml_tensor* x,     // [C, T]
    struct ggml_tensor* w,     // [kernel, C, 2*C]
    struct ggml_tensor* b,     // [2*C]
    ggml_backend_t backend
) {
    int in_ch = (int)x->ne[0];
    struct ggml_tensor* x_t = ggml_cont(ctx, ggml_transpose(ctx, x));
    int pad = (int)((ggml_is_quantized(w->type) ? w->ne[1] : w->ne[0]) - 1) / 2;
    struct ggml_tensor* conv = nn::F::conv1d_no_transpose(ctx, x_t, w, nullptr, 1, pad, 1, 1, backend);
    struct ggml_tensor* conv_t = ggml_cont(ctx, ggml_transpose(ctx, conv));  // [2*C, T]

    struct ggml_tensor* b2d = ggml_reshape_2d(ctx, b, b->ne[0], 1);
    conv_t = ggml_add(ctx, conv_t, b2d);

    struct ggml_tensor* x1 = ggml_view_2d(ctx, conv_t, in_ch, conv_t->ne[1], conv_t->nb[1], 0);
    struct ggml_tensor* x2 = ggml_view_2d(ctx, conv_t, in_ch, conv_t->ne[1], conv_t->nb[1], in_ch * sizeof(float));

    struct ggml_tensor* glu = ggml_mul(ctx, x1, ggml_sigmoid(ctx, x2));
    return ggml_add(ctx, x, glu);
}

static struct ggml_tensor* build_encoder_layer(
    struct ggml_context* ctx,
    struct ggml_tensor* x,        // [C, T], ne0=C=192, ne1=T
    VITSModel& model,
    const std::string& prefix,    // e.g., "enc_p.encoder_ssl.attn_layers.0."
    const std::string& norm1_prefix,
    const std::string& norm2_prefix,
    const std::string& ffn1_prefix,
    const std::string& ffn2_prefix,
    int n_head,
    int d_k,
    int T,
    struct ggml_tensor* emb_rel_k,
    struct ggml_tensor* emb_rel_v,
    ggml_backend_t backend
) {
    nn::Context graph_context = nn::Context::borrow(ctx);
    struct ggml_tensor* q_w = model.get_tensor(prefix + "conv_q.weight"); // [1, C, C]
    struct ggml_tensor* q_b = model.get_tensor(prefix + "conv_q.bias");
    struct ggml_tensor* k_w = model.get_tensor(prefix + "conv_k.weight");
    struct ggml_tensor* k_b = model.get_tensor(prefix + "conv_k.bias");
    struct ggml_tensor* v_w = model.get_tensor(prefix + "conv_v.weight");
    struct ggml_tensor* v_b = model.get_tensor(prefix + "conv_v.bias");
    struct ggml_tensor* o_w = model.get_tensor(prefix + "conv_o.weight");
    struct ggml_tensor* o_b = model.get_tensor(prefix + "conv_o.bias");

    struct ggml_tensor* q = nn::F::conv1d(ctx, x, q_w, q_b, 1, 0, 1, 1, backend);
    struct ggml_tensor* k = nn::F::conv1d(ctx, x, k_w, k_b, 1, 0, 1, 1, backend);
    struct ggml_tensor* v = nn::F::conv1d(ctx, x, v_w, v_b, 1, 0, 1, 1, backend);

    q = ggml_cont(ctx, ggml_reshape_4d(ctx, q, d_k, n_head, T, 1));
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3)); // [d_k, T, n_head, 1]
    k = ggml_cont(ctx, ggml_reshape_4d(ctx, k, d_k, n_head, T, 1));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_reshape_4d(ctx, v, d_k, n_head, T, 1));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);
    struct ggml_tensor* k_f32 = (k->type == GGML_TYPE_F32) ? k : ggml_cast(ctx, k, GGML_TYPE_F32);
    struct ggml_tensor* q_f32 = (q->type == GGML_TYPE_F32) ? q : ggml_cast(ctx, q, GGML_TYPE_F32);
    struct ggml_tensor* relative_bias = emb_rel_k
        ? ggml_ops_relative_pe_keys(ctx, q_f32, emb_rel_k, inv_sqrt_dk, 4, backend)
        : nullptr;
    struct ggml_tensor* attn_w = graph_context.empty<float>(
        "vits.attn.weights", {T, T, n_head, 1});

    struct ggml_tensor* v_f32 = (v->type == GGML_TYPE_F32) ? v : ggml_cast(ctx, v, GGML_TYPE_F32);
    struct ggml_tensor* out = ggml_ops_attention(
        ctx, q_f32, k_f32, v_f32, relative_bias, attn_w, inv_sqrt_dk, 4, backend);

    out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
    struct ggml_tensor* attn_raw = ggml_reshape_2d(ctx, out, d_k * n_head, T);

    if (emb_rel_v != nullptr) {
        attn_raw = ggml_add(
            ctx, attn_raw, ggml_ops_relative_pe_values(ctx, attn_w, emb_rel_v, out, 4, backend));
    }

    struct ggml_tensor* attn_proj = nn::F::conv1d(ctx, attn_raw, o_w, o_b, 1, 0, 1, 1, backend);
    struct ggml_tensor* x_attn = ggml_add(ctx, x, attn_proj);

    struct ggml_tensor* ln1_g = model.get_tensor(norm1_prefix + ".gamma");
    struct ggml_tensor* ln1_b = model.get_tensor(norm1_prefix + ".beta");
    x_attn = nn::F::layer_norm(ctx, x_attn, ln1_g, ln1_b, 1e-5f, backend);

    struct ggml_tensor* ffn_w1 = model.get_tensor(ffn1_prefix + ".weight");
    struct ggml_tensor* ffn_b1 = model.get_tensor(ffn1_prefix + ".bias");
    struct ggml_tensor* ffn_w2 = model.get_tensor(ffn2_prefix + ".weight");
    struct ggml_tensor* ffn_b2 = model.get_tensor(ffn2_prefix + ".bias");

    struct ggml_tensor* ffn_out = nn::F::conv1d(ctx, x_attn, ffn_w1, ffn_b1, 1, 1, 1, 1, backend);  // kernel=3, pad=1
    ffn_out = ggml_relu(ctx, ffn_out);
    ffn_out = nn::F::conv1d(ctx, ffn_out, ffn_w2, ffn_b2, 1, 1, 1, 1, backend);

    struct ggml_tensor* x_out = ggml_add(ctx, x_attn, ffn_out);
    struct ggml_tensor* ln2_g = model.get_tensor(norm2_prefix + ".gamma");
    struct ggml_tensor* ln2_b = model.get_tensor(norm2_prefix + ".beta");
    struct ggml_tensor* result = nn::F::layer_norm(ctx, x_out, ln2_g, ln2_b, 1e-5f, backend);

    return result;
}

struct ggml_tensor* build_encoder(
    struct ggml_context* ctx,
    struct ggml_tensor* x,        // [C, T]
    VITSModel& model,
    const std::string& base_prefix,  // "enc_p.encoder_ssl"
    int n_layers,
    int n_head,
    int d_k,
    int T,
    ggml_backend_t backend
) {
    for (int l = 0; l < n_layers; ++l) {
        std::string lp = base_prefix + ".attn_layers." + std::to_string(l) + ".";
        std::string n1p = base_prefix + ".norm_layers_1." + std::to_string(l);
        std::string n2p = base_prefix + ".norm_layers_2." + std::to_string(l);
        std::string f1p = base_prefix + ".ffn_layers." + std::to_string(l) + ".conv_1";
        std::string f2p = base_prefix + ".ffn_layers." + std::to_string(l) + ".conv_2";

        struct ggml_tensor* emb_rel_k = model.get_tensor(lp + "emb_rel_k");
        struct ggml_tensor* emb_rel_v = model.get_tensor(lp + "emb_rel_v");
        
        if (emb_rel_k && emb_rel_k->type != GGML_TYPE_F32) {
            emb_rel_k = ggml_cast(ctx, emb_rel_k, GGML_TYPE_F32);
        }
        if (emb_rel_v && emb_rel_v->type != GGML_TYPE_F32) {
            emb_rel_v = ggml_cast(ctx, emb_rel_v, GGML_TYPE_F32);
        }

        x = build_encoder_layer(ctx, x, model, lp, n1p, n2p, f1p, f2p, n_head, d_k, T, emb_rel_k, emb_rel_v, backend);
    }
    return x;
}

struct ggml_tensor* build_mrte(
    struct ggml_context* ctx,
    struct ggml_tensor* y,        // SSL features [192, T_y]
    struct ggml_tensor* text,     // Text features [192, T_x]
    struct ggml_tensor* ge,       // Speaker embedding [512, 1]
    VITSModel& model,
    ggml_backend_t backend
) {
    int T_y = (int)y->ne[1];
    int T_x = (int)text->ne[1];

    struct ggml_tensor* c_pre_w = model.get_tensor("enc_p.mrte.c_pre.weight");
    struct ggml_tensor* c_pre_b = model.get_tensor("enc_p.mrte.c_pre.bias");
    struct ggml_tensor* y_proj = nn::F::conv1d(ctx, y, c_pre_w, c_pre_b, 1, 0, 1, 1, backend);  // [512, T_y]

    struct ggml_tensor* text_pre_w = model.get_tensor("enc_p.mrte.text_pre.weight");
    struct ggml_tensor* text_pre_b = model.get_tensor("enc_p.mrte.text_pre.bias");
    struct ggml_tensor* text_proj = nn::F::conv1d(ctx, text, text_pre_w, text_pre_b, 1, 0, 1, 1, backend);  // [512, T_x]

    int mrte_n_head = 4;
    int mrte_d_k = 128;

    struct ggml_tensor* q_w = model.get_tensor("enc_p.mrte.cross_attention.conv_q.weight");
    struct ggml_tensor* q_b = model.get_tensor("enc_p.mrte.cross_attention.conv_q.bias");
    struct ggml_tensor* k_w = model.get_tensor("enc_p.mrte.cross_attention.conv_k.weight");
    struct ggml_tensor* k_b = model.get_tensor("enc_p.mrte.cross_attention.conv_k.bias");
    struct ggml_tensor* v_w = model.get_tensor("enc_p.mrte.cross_attention.conv_v.weight");
    struct ggml_tensor* v_b = model.get_tensor("enc_p.mrte.cross_attention.conv_v.bias");
    struct ggml_tensor* o_w = model.get_tensor("enc_p.mrte.cross_attention.conv_o.weight");
    struct ggml_tensor* o_b = model.get_tensor("enc_p.mrte.cross_attention.conv_o.bias");

    struct ggml_tensor* q_mrte = nn::F::conv1d(ctx, y_proj, q_w, q_b, 1, 0, 1, 1, backend);  // [512, T_y]
    struct ggml_tensor* k_mrte = nn::F::conv1d(ctx, text_proj, k_w, k_b, 1, 0, 1, 1, backend);  // [512, T_x]
    struct ggml_tensor* v_mrte = nn::F::conv1d(ctx, text_proj, v_w, v_b, 1, 0, 1, 1, backend);  // [512, T_x]

    q_mrte = ggml_cont(ctx, ggml_reshape_3d(ctx, q_mrte, mrte_d_k, mrte_n_head, T_y));
    q_mrte = ggml_cont(ctx, ggml_permute(ctx, q_mrte, 0, 2, 1, 3));  // [128, T_y, 4]

    k_mrte = ggml_cont(ctx, ggml_reshape_3d(ctx, k_mrte, mrte_d_k, mrte_n_head, T_x));
    k_mrte = ggml_cont(ctx, ggml_permute(ctx, k_mrte, 0, 2, 1, 3));  // [128, T_x, 4]

    v_mrte = ggml_cont(ctx, ggml_reshape_3d(ctx, v_mrte, mrte_d_k, mrte_n_head, T_x));
    v_mrte = ggml_cont(ctx, ggml_permute(ctx, v_mrte, 0, 2, 1, 3));  // [128, T_x, 4]

    float mrte_scale = 1.0f / sqrtf((float)mrte_d_k);
    struct ggml_tensor* k_mrte_f32 = (k_mrte->type == GGML_TYPE_F32) ? k_mrte : ggml_cast(ctx, k_mrte, GGML_TYPE_F32);
    struct ggml_tensor* q_mrte_f32 = (q_mrte->type == GGML_TYPE_F32) ? q_mrte : ggml_cast(ctx, q_mrte, GGML_TYPE_F32);
    struct ggml_tensor* v_mrte_f32 = (v_mrte->type == GGML_TYPE_F32) ? v_mrte : ggml_cast(ctx, v_mrte, GGML_TYPE_F32);
    struct ggml_tensor* out_mrte = ggml_ops_attention(
        ctx, q_mrte_f32, k_mrte_f32, v_mrte_f32, nullptr, nullptr,
        mrte_scale, -1, backend);

    out_mrte = ggml_cont(ctx, ggml_permute(ctx, out_mrte, 0, 2, 1, 3));
    struct ggml_tensor* cross_out = ggml_reshape_2d(ctx, out_mrte, mrte_d_k * mrte_n_head, T_y);

    struct ggml_tensor* cross_proj = nn::F::conv1d(ctx, cross_out, o_w, o_b, 1, 0, 1, 1, backend);  // [512, T_y]

    struct ggml_tensor* ge_2d = ggml_reshape_2d(ctx, ge, ge->ne[0], 1);
    struct ggml_tensor* mrte_res = ggml_add(ctx, y_proj, cross_proj);
    mrte_res = ggml_add(ctx, mrte_res, ge_2d);

    struct ggml_tensor* c_post_w = model.get_tensor("enc_p.mrte.c_post.weight");
    struct ggml_tensor* c_post_b = model.get_tensor("enc_p.mrte.c_post.bias");
    struct ggml_tensor* result = nn::F::conv1d(ctx, mrte_res, c_post_w, c_post_b, 1, 0, 1, 1, backend);  // [192, T_y]

    return result;
}

static struct ggml_tensor* build_ref_enc(
    struct ggml_context* ctx,
    struct ggml_tensor* mel_spec,
    VITSModel& model,
    ggml_backend_t backend
) {
    int64_t T = mel_spec->ne[1];

    struct ggml_tensor* x = mel_spec;
    struct ggml_tensor* s0_w = model.get_tensor("ref_enc.spectral.0.fc.weight");
    struct ggml_tensor* s0_b = model.get_tensor("ref_enc.spectral.0.fc.bias");
    x = nn::F::linear(ctx, x, s0_w, s0_b, backend);
    x = ggml_ops_mish(ctx, x, backend);

    struct ggml_tensor* s3_w = model.get_tensor("ref_enc.spectral.3.fc.weight");
    struct ggml_tensor* s3_b = model.get_tensor("ref_enc.spectral.3.fc.bias");
    x = nn::F::linear(ctx, x, s3_w, s3_b, backend);
    x = ggml_ops_mish(ctx, x, backend);

    struct ggml_tensor* t0_w = model.get_tensor("ref_enc.temporal.0.conv1.conv.weight");
    struct ggml_tensor* t0_b = model.get_tensor("ref_enc.temporal.0.conv1.conv.bias");
    x = conv1d_glu(ctx, x, t0_w, t0_b, backend);

    struct ggml_tensor* t1_w = model.get_tensor("ref_enc.temporal.1.conv1.conv.weight");
    struct ggml_tensor* t1_b = model.get_tensor("ref_enc.temporal.1.conv1.conv.bias");
    x = conv1d_glu(ctx, x, t1_w, t1_b, backend);

    {
        int n_head = 2;
        int d_k = 64;
        int d_v = 64;

        struct ggml_tensor* residual = x;

        struct ggml_tensor* w_qs = model.get_tensor("ref_enc.slf_attn.w_qs.weight");
        struct ggml_tensor* b_qs = model.get_tensor("ref_enc.slf_attn.w_qs.bias");
        struct ggml_tensor* w_ks = model.get_tensor("ref_enc.slf_attn.w_ks.weight");
        struct ggml_tensor* b_ks = model.get_tensor("ref_enc.slf_attn.w_ks.bias");
        struct ggml_tensor* w_vs = model.get_tensor("ref_enc.slf_attn.w_vs.weight");
        struct ggml_tensor* b_vs = model.get_tensor("ref_enc.slf_attn.w_vs.bias");
        struct ggml_tensor* attn_fc_w = model.get_tensor("ref_enc.slf_attn.fc.weight");
        struct ggml_tensor* attn_fc_b = model.get_tensor("ref_enc.slf_attn.fc.bias");

        struct ggml_tensor* q = nn::F::linear(ctx, x, w_qs, b_qs, backend);
        struct ggml_tensor* k = nn::F::linear(ctx, x, w_ks, b_ks, backend);
        struct ggml_tensor* v = nn::F::linear(ctx, x, w_vs, b_vs, backend);

        q = ggml_cont(ctx, ggml_reshape_3d(ctx, q, d_k, n_head, T));
        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        q = ggml_reshape_4d(ctx, q, d_k, T, n_head, 1);

        k = ggml_cont(ctx, ggml_reshape_3d(ctx, k, d_k, n_head, T));
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        k = ggml_reshape_4d(ctx, k, d_k, T, n_head, 1);

        v = ggml_cont(ctx, ggml_reshape_3d(ctx, v, d_v, n_head, T));
        v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
        v = ggml_reshape_4d(ctx, v, d_v, T, n_head, 1);

        float scale = 1.0f / sqrtf(128.0f);
        struct ggml_tensor* k_f32 = (k->type == GGML_TYPE_F32) ? k : ggml_cast(ctx, k, GGML_TYPE_F32);
        struct ggml_tensor* q_f32 = (q->type == GGML_TYPE_F32) ? q : ggml_cast(ctx, q, GGML_TYPE_F32);
        struct ggml_tensor* v_f32 = (v->type == GGML_TYPE_F32) ? v : ggml_cast(ctx, v, GGML_TYPE_F32);
        struct ggml_tensor* attn_out = ggml_ops_attention(
            ctx, q_f32, k_f32, v_f32, nullptr, nullptr, scale, -1, backend);

        // PyTorch flattens [head, d_v] for each time step, not [time, d_v].
        attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));
        attn_out = ggml_cont(ctx, ggml_reshape_2d(ctx, attn_out, d_v * n_head, T));
        struct ggml_tensor* output = nn::F::linear(ctx, attn_out, attn_fc_w, attn_fc_b, backend);
        x = ggml_add(ctx, output, residual);
    }
    struct ggml_tensor* fc_w = model.get_tensor("ref_enc.fc.fc.weight");
    struct ggml_tensor* fc_b = model.get_tensor("ref_enc.fc.fc.bias");
    x = nn::F::linear(ctx, x, fc_w, fc_b, backend);

    x = ggml_cont(ctx, ggml_transpose(ctx, x));
    struct ggml_tensor* summed = ggml_sum_rows(ctx, x);
    struct ggml_tensor* ge = ggml_scale(ctx, summed, 1.0f / (float)T);
    ge = ggml_cont(ctx, ge);
    return ge;
}

struct ggml_tensor* VITSModel::compute_speaker_embedding(
    struct ggml_context* ctx_graph,
    struct ggml_tensor* mel_spec,
    struct ggml_tensor* sv_emb,
    ggml_backend_t backend
) {
    struct ggml_tensor* ge = build_ref_enc(ctx_graph, mel_spec, *this, backend);
    
    struct ggml_tensor* sv_emb_b = get_tensor("sv_emb.bias");
    struct ggml_tensor* sv_emb_w = get_tensor("sv_emb.weight");
    struct ggml_tensor* prelu_w = get_tensor("prelu.weight");
    if (sv_emb_b && prelu_w) {
        int64_t dim = sv_emb_b->ne[0];
        
        struct ggml_tensor* sv_proj = nullptr;
        if (sv_emb && sv_emb_w) {
            nn::Linear sv_emb_layer(sv_emb_w, sv_emb_b); sv_emb_layer.to(backend);
            sv_proj = sv_emb_layer(ctx_graph, sv_emb);
            sv_proj = ggml_reshape_2d(ctx_graph, sv_proj, 1, dim);
        } else {
            sv_proj = ggml_reshape_2d(ctx_graph, sv_emb_b, 1, dim);
        }
        
        ge = ggml_add(ctx_graph, ge, sv_proj);
        
        nn::PReLU prelu_layer(prelu_w); prelu_layer.to(backend);
        ge = prelu_layer(ctx_graph, ge);
    }
    
    return ge;
}

struct ggml_tensor* WN::forward(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* x_mask,
    struct ggml_tensor* g,
    ggml_backend_t backend
) {
    ggml_backend_t b = backend ? backend : this->backend;
    nn::Context graph_context = nn::Context::borrow(ctx);
    struct ggml_tensor* output = graph_context.empty<float>("vits.wn.output", {hidden_channels, x->ne[1]});
    output = ggml_fill(ctx, output, 0.0f);

    for (int i = 0; i < n_layers; ++i) {
        struct ggml_tensor* x_in = in_layers[i]->forward(ctx, x, b);

        if (g) {
            int cond_offset = i * 2 * hidden_channels;
            struct ggml_tensor* g_l = ggml_view_2d(ctx, g, 2 * hidden_channels, g->ne[1],
                g->nb[1], cond_offset * sizeof(float));
            x_in = ggml_add(ctx, x_in, ggml_cont(ctx, g_l));
        }

        struct ggml_tensor* acts = ggml_ops_gated_tanh_sigmoid(ctx, x_in, hidden_channels, b);

        struct ggml_tensor* res_skip = res_skip_layers[i]->forward(ctx, acts, b);

        if (i < n_layers - 1) {
            size_t rs_stride = res_skip->ne[0] * sizeof(float);
            struct ggml_tensor* res_acts = ggml_cont(ctx, ggml_view_2d(ctx, res_skip, hidden_channels,
                res_skip->ne[1], rs_stride, 0));
            x = ggml_add(ctx, x, res_acts);
            if (x_mask) {
                x = ggml_mul(ctx, x, ggml_repeat(ctx, x_mask, x));
            }
            struct ggml_tensor* skip_acts = ggml_cont(ctx, ggml_view_2d(ctx, res_skip, hidden_channels,
                res_skip->ne[1], rs_stride, hidden_channels * sizeof(float)));
            output = ggml_add(ctx, output, skip_acts);
        } else {
            output = ggml_add(ctx, output, res_skip);
        }
    }
    if (x_mask) {
        output = ggml_mul(ctx, output, ggml_repeat(ctx, x_mask, output));
    }
    return output;
}

struct ggml_tensor* ResidualCouplingLayer::forward(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* x_mask,
    struct ggml_tensor* g,
    ggml_backend_t backend
) {
    ggml_backend_t b = backend ? backend : this->backend;
    int channels = x->ne[0];
    int half_c = channels / 2;

    size_t x_stride = x->ne[0] * sizeof(float);
    struct ggml_tensor* x0 = ggml_view_2d(ctx, x, half_c, x->ne[1], x_stride, 0);
    struct ggml_tensor* x1 = ggml_view_2d(ctx, x, half_c, x->ne[1], x_stride, half_c * sizeof(float));

    struct ggml_tensor* h = pre.forward(ctx, x0, b);
    if (x_mask) {
        h = ggml_mul(ctx, h, ggml_repeat(ctx, x_mask, h));
    }

    struct ggml_tensor* g_proj = nullptr;
    if (cond_layer.weight.is_bound() && g) {
        g_proj = cond_layer.forward(ctx, g, b);
    }

    h = wn.forward(ctx, h, x_mask, g_proj, b);

    struct ggml_tensor* mean = post.forward(ctx, h, b);
    if (x_mask) {
        mean = ggml_mul(ctx, mean, ggml_repeat(ctx, x_mask, mean));
    }
    struct ggml_tensor* new_x1 = ggml_sub(ctx, ggml_cont(ctx, x1), mean);
    return ggml_cont(ctx, ggml_concat(ctx, x0, new_x1, 0));
}

struct ggml_tensor* WNEncoder::forward(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* g,
    ggml_backend_t backend
) {
    ggml_backend_t b = backend ? backend : this->backend;
    struct ggml_tensor* h = pre.forward(ctx, x, b);

    struct ggml_tensor* g_proj = nullptr;
    if (g) {
        g_proj = cond_layer.forward(ctx, g, b);
    }

    struct ggml_tensor* WN_out = wn.forward(ctx, h, nullptr, g_proj, b);
    struct ggml_tensor* out = proj.forward(ctx, WN_out, b);
    return out;
}

struct ggml_tensor* build_vits_generator(
    struct ggml_context* ctx_graph,
    struct ggml_tensor* latent,
    struct ggml_tensor* speaker_embedding,
    VITSModel& model,
    ggml_backend_t backend
) {
    struct ggml_tensor* dec_conv_pre_w = model.gen_weights.conv_pre_w;
    struct ggml_tensor* dec_conv_pre_b = model.gen_weights.conv_pre_b;
    if (!dec_conv_pre_w || !dec_conv_pre_b) {
        std::cerr << "[VITS] Error: Missing dec.conv_pre weights!" << std::endl;
        return nullptr;
    }

    struct ggml_tensor* latent_transposed = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, latent));
    struct ggml_tensor* h = nn::F::conv1d_no_transpose(ctx_graph, latent_transposed, dec_conv_pre_w, dec_conv_pre_b, 1, 3, 1, 1, backend);
    if (!h) {
        std::cerr << "[VITS] Error: dec.conv_pre contract rejected its packed weight." << std::endl;
        return nullptr;
    }

    if (speaker_embedding != nullptr) {
        struct ggml_tensor* cond_w = model.gen_weights.cond_w;
        struct ggml_tensor* cond_b = model.gen_weights.cond_b;
        if (cond_w && cond_b) {
            struct ggml_tensor* g_proj_t = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, speaker_embedding));
            struct ggml_tensor* g_proj = nn::F::conv1d_no_transpose(ctx_graph, g_proj_t, cond_w, cond_b, 1, 0, 1, 1, backend); // [1, channels]
            if (!g_proj) {
                std::cerr << "[VITS] Error: dec.cond contract rejected its packed weight." << std::endl;
                return nullptr;
            }
            h = ggml_add(ctx_graph, h, g_proj);
        }
    }

    const std::vector<int> dilations = {1, 3, 5};
    const auto upsample_padding = [](const struct ggml_tensor* weight, int stride) {
        const int kernel = (int)(ggml_is_quantized(weight->type) ? weight->ne[1] : weight->ne[0]);
        return (kernel - stride) / 2;
    };

    struct ggml_tensor* ups0_w = model.gen_weights.ups_w[0];
    struct ggml_tensor* ups0_b = model.gen_weights.ups_b[0];
    struct ggml_tensor* ups1_w = model.gen_weights.ups_w[1];
    struct ggml_tensor* ups1_b = model.gen_weights.ups_b[1];
    struct ggml_tensor* ups2_w = model.gen_weights.ups_w[2];
    struct ggml_tensor* ups2_b = model.gen_weights.ups_b[2];
    struct ggml_tensor* ups3_w = model.gen_weights.ups_w[3];
    struct ggml_tensor* ups3_b = model.gen_weights.ups_b[3];
    struct ggml_tensor* ups4_w = model.gen_weights.ups_w[4];
    struct ggml_tensor* ups4_b = model.gen_weights.ups_b[4];

    if (ups0_w && ups0_b) {
        h = ggml_leaky_relu(ctx_graph, h, 0.1f, false);
        h = nn::F::conv_transpose1d_no_transpose(
            ctx_graph, h, ups0_w, ups0_b, 10, upsample_padding(ups0_w, 10), backend);
        
        struct ggml_tensor* r0 = mrf_resblock_no_transpose(ctx_graph, h, model, 0, 256, 3, dilations, backend);
        struct ggml_tensor* r1 = mrf_resblock_no_transpose(ctx_graph, h, model, 1, 256, 7, dilations, backend);
        struct ggml_tensor* r2 = mrf_resblock_no_transpose(ctx_graph, h, model, 2, 256, 11, dilations, backend);
        
        
        struct ggml_tensor* xs = ggml_add(ctx_graph, r0, r1);
        xs = ggml_add(ctx_graph, xs, r2);
        h = ggml_scale(ctx_graph, xs, 1.0f / 3.0f);
    }

    if (ups1_w && ups1_b) {
        h = ggml_leaky_relu(ctx_graph, h, 0.1f, false);
        h = nn::F::conv_transpose1d_no_transpose(
            ctx_graph, h, ups1_w, ups1_b, 8, upsample_padding(ups1_w, 8), backend);
        
        struct ggml_tensor* r3 = mrf_resblock_no_transpose(ctx_graph, h, model, 3, 128, 3, dilations, backend);
        struct ggml_tensor* r4 = mrf_resblock_no_transpose(ctx_graph, h, model, 4, 128, 7, dilations, backend);
        struct ggml_tensor* r5 = mrf_resblock_no_transpose(ctx_graph, h, model, 5, 128, 11, dilations, backend);
        
        
        struct ggml_tensor* xs = ggml_add(ctx_graph, r3, r4);
        xs = ggml_add(ctx_graph, xs, r5);
        h = ggml_scale(ctx_graph, xs, 1.0f / 3.0f);
    }

    if (ups2_w && ups2_b) {
        h = ggml_leaky_relu(ctx_graph, h, 0.1f, false);
        h = nn::F::conv_transpose1d_no_transpose(
            ctx_graph, h, ups2_w, ups2_b, 2, upsample_padding(ups2_w, 2), backend);
        
        struct ggml_tensor* r6 = mrf_resblock_no_transpose(ctx_graph, h, model, 6, 64, 3, dilations, backend);
        struct ggml_tensor* r7 = mrf_resblock_no_transpose(ctx_graph, h, model, 7, 64, 7, dilations, backend);
        struct ggml_tensor* r8 = mrf_resblock_no_transpose(ctx_graph, h, model, 8, 64, 11, dilations, backend);
        
        
        struct ggml_tensor* xs = ggml_add(ctx_graph, r6, r7);
        xs = ggml_add(ctx_graph, xs, r8);
        h = ggml_scale(ctx_graph, xs, 1.0f / 3.0f);
    }

    if (ups3_w && ups3_b) {
        h = ggml_leaky_relu(ctx_graph, h, 0.1f, false);
        h = nn::F::conv_transpose1d_no_transpose(
            ctx_graph, h, ups3_w, ups3_b, 2, upsample_padding(ups3_w, 2), backend);
        
        struct ggml_tensor* r9 = mrf_resblock_no_transpose(ctx_graph, h, model, 9, 32, 3, dilations, backend);
        struct ggml_tensor* r10 = mrf_resblock_no_transpose(ctx_graph, h, model, 10, 32, 7, dilations, backend);
        struct ggml_tensor* r11 = mrf_resblock_no_transpose(ctx_graph, h, model, 11, 32, 11, dilations, backend);
        
        
        struct ggml_tensor* xs = ggml_add(ctx_graph, r9, r10);
        xs = ggml_add(ctx_graph, xs, r11);
        h = ggml_scale(ctx_graph, xs, 1.0f / 3.0f);
    }

    if (ups4_w && ups4_b) {
        h = ggml_leaky_relu(ctx_graph, h, 0.1f, false);
        h = nn::F::conv_transpose1d_no_transpose(
            ctx_graph, h, ups4_w, ups4_b, 2, upsample_padding(ups4_w, 2), backend);
        
        struct ggml_tensor* r12 = mrf_resblock_no_transpose(ctx_graph, h, model, 12, 16, 3, dilations, backend);
        struct ggml_tensor* r13 = mrf_resblock_no_transpose(ctx_graph, h, model, 13, 16, 7, dilations, backend);
        struct ggml_tensor* r14 = mrf_resblock_no_transpose(ctx_graph, h, model, 14, 16, 11, dilations, backend);
        
        
        struct ggml_tensor* xs = ggml_add(ctx_graph, r12, r13);
        xs = ggml_add(ctx_graph, xs, r14);
        h = ggml_scale(ctx_graph, xs, 1.0f / 3.0f);
    }

    struct ggml_tensor* conv_post_w = model.gen_weights.conv_post_w;
    if (!conv_post_w) {
        std::cerr << "[VITS] Error: Missing dec.conv_post.weight!" << std::endl;
        return nullptr;
    }

    h = ggml_leaky_relu(ctx_graph, h, 0.01f, false);
    struct ggml_tensor* conv = nn::F::conv1d_no_transpose(ctx_graph, h, conv_post_w, nullptr, 1, 3, 1, 1, backend); // [out_seq_len, 1]
    struct ggml_tensor* audio = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, conv));
    return ggml_tanh(ctx_graph, audio);
}

} // namespace gpt_sovits
