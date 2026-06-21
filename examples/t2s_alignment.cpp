#include "models.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace {

static ggml_backend_t pick_backend(bool use_gpu) {
    ggml_backend_load_all();
    if (!use_gpu) {
        return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }

    const size_t n_devs = ggml_backend_dev_count();
    for (size_t i = 0; i < n_devs; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) {
            continue;
        }
        const std::string name = ggml_backend_dev_name(dev);
        if (name.rfind("CUDA", 0) == 0 || name.rfind("SYCL", 0) == 0) {
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            if (backend) {
                return backend;
            }
        }
    }
    return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
}

static std::vector<float> compute_positional_embeddings(int seq_len, int hidden_dim, float alpha) {
    std::vector<float> data(seq_len * hidden_dim, 0.0f);
    for (int pos = 0; pos < seq_len; ++pos) {
        for (int i = 0; i < hidden_dim / 2; ++i) {
            float denom = std::pow(10000.0f, (2.0f * i) / (float)hidden_dim);
            float val = (float)pos / denom;
            data[pos * hidden_dim + 2 * i] = alpha * std::sin(val);
            data[pos * hidden_dim + 2 * i + 1] = alpha * std::cos(val);
        }
    }
    return data;
}

static std::vector<float> compute_prefix_causal_mask(int seq_len, int text_len, int n_heads) {
    std::vector<float> data(seq_len * seq_len * n_heads, 0.0f);
    for (int h = 0; h < n_heads; ++h) {
        for (int r = 0; r < seq_len; ++r) {
            for (int c = 0; c < seq_len; ++c) {
                int64_t idx = h * (seq_len * seq_len) + r * seq_len + c;
                if (r < text_len) {
                    data[idx] = (c < text_len) ? 0.0f : -1e4f;
                } else {
                    data[idx] = (c <= r) ? 0.0f : -1e4f;
                }
            }
        }
    }
    return data;
}

static void write_f32_binary(const std::string & path, const std::vector<float> & data) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[T2S Align] Failed to open binary output: " << path << "\n";
        return;
    }
    file.write(reinterpret_cast<const char *>(data.data()), (std::streamsize)(data.size() * sizeof(float)));
}

static void save_tensor(const std::string & path, struct ggml_tensor * tensor) {
    if (!tensor) return;
    int64_t out_elements = ggml_nelements(tensor);
    std::vector<float> host_out(out_elements);
    ggml_backend_tensor_get(tensor, host_out.data(), 0, out_elements * sizeof(float));
    write_f32_binary(path, host_out);
}

static void print_top_logits(const std::string & label, const std::vector<float> & logits) {
    std::vector<std::pair<float, int>> top_logits;
    top_logits.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        top_logits.push_back({logits[i], (int)i});
    }
    int n = (int)top_logits.size();
    int k = std::min(5, n);
    for (int i = 0; i < k; ++i) {
        int max_idx = i;
        for (int j = i + 1; j < n; ++j) {
            if (top_logits[j].first > top_logits[max_idx].first) {
                max_idx = j;
            }
        }
        if (max_idx != i) {
            std::swap(top_logits[i], top_logits[max_idx]);
        }
    }
    std::cout << "[T2S Align] " << label << " Top 5 predicted tokens:\n";
    for (int i = 0; i < k; ++i) {
        std::cout << "  Token " << top_logits[i].second << ": Logit " << top_logits[i].first << "\n";
    }
}


// DoubleSwish activation function matching PyTorch: x * sigmoid(x - 1.0f)
static struct ggml_tensor* ggml_double_swish(struct ggml_context* ctx, struct ggml_tensor* x) {
    struct ggml_tensor* ones = ggml_new_tensor(ctx, GGML_TYPE_F32, ggml_n_dims(x), x->ne);
    ones = ggml_fill(ctx, ones, 1.0f);
    struct ggml_tensor* x_minus_1 = ggml_sub(ctx, x, ones);
    struct ggml_tensor* sig = ggml_sigmoid(ctx, x_minus_1);
    return ggml_mul(ctx, x, sig);
}

} // namespace

int main(int argc, char ** argv) {
    std::string t2s_path = "models/speech/Firekeeper_v2-e50_t2s.gguf";
    std::string out_prefix = "scratch/t2s_alignment_cpp";
    bool use_gpu = true;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--t2s" && i + 1 < argc) {
            t2s_path = argv[++i];
        } else if (arg == "--out-prefix" && i + 1 < argc) {
            out_prefix = argv[++i];
        } else if (arg == "--cpu") {
            use_gpu = false;
        }
    }

    ggml_backend_t backend = pick_backend(use_gpu);
    if (!backend) {
        std::cerr << "[T2S Align] Failed to initialize backend\n";
        return 1;
    }

    gpt_sovits::T2SModel model;
    if (!model.load(t2s_path, backend)) {
        std::cerr << "[T2S Align] Failed to load model: " << t2s_path << "\n";
        ggml_backend_free(backend);
        return 1;
    }

    // Deterministic mock inputs
    std::vector<int32_t> prompt_phones = {3,5,8,12,15};
    std::vector<int32_t> target_phones = {6,9,11,14,18,20};
    std::vector<int32_t> prompt_semantics = {120,345,678};
    int text_len = (int)(prompt_phones.size() + target_phones.size());
    int prompt_audio_len = (int)prompt_semantics.size();

    // Dummy BERT features (1024 x text_len)
    std::vector<float> bert_features_host(1024 * text_len);
    for (int s = 0; s < text_len; ++s) {
        for (int c = 0; c < 1024; ++c) {
            bert_features_host[s * 1024 + c] = std::sin((float)(s * c)) * 0.1f;
        }
    }

    // Retrieve required model tensors
    struct ggml_tensor* text_embed = model.get_tensor("ar_text_embedding.word_embeddings.weight");
    struct ggml_tensor* audio_embed = model.get_tensor("ar_audio_embedding.word_embeddings.weight");
    struct ggml_tensor* bert_proj_w = model.get_tensor("bert_proj.weight");
    struct ggml_tensor* bert_proj_b = model.get_tensor("bert_proj.bias");
    struct ggml_tensor* ar_text_position_alpha = model.get_tensor("ar_text_position.alpha");
    struct ggml_tensor* ar_audio_position_alpha = model.get_tensor("ar_audio_position.alpha");
    struct ggml_tensor* predict_w = model.get_tensor("ar_predict_layer.weight");
    if (!text_embed || !audio_embed || !bert_proj_w || !bert_proj_b || !ar_text_position_alpha || !ar_audio_position_alpha || !predict_w) {
        std::cerr << "[T2S Align] Error: Missing model weights in GGUF mapping!\n";
        ggml_backend_free(backend);
        return 1;
    }

    float text_alpha = 1.0f;
    float audio_alpha = 1.0f;
    ggml_backend_tensor_get(ar_text_position_alpha, &text_alpha, 0, sizeof(float));
    ggml_backend_tensor_get(ar_audio_position_alpha, &audio_alpha, 0, sizeof(float));
    std::cout << "[T2S Align] Read text_alpha = " << text_alpha << ", audio_alpha = " << audio_alpha << "\n";


    // Build input id sequences
    std::vector<int32_t> text_ids;
    text_ids.insert(text_ids.end(), prompt_phones.begin(), prompt_phones.end());
    text_ids.insert(text_ids.end(), target_phones.begin(), target_phones.end());
    std::vector<int32_t> current_audio_ids = {1024};
    current_audio_ids.insert(current_audio_ids.end(), prompt_semantics.begin(), prompt_semantics.end());
    std::vector<int32_t> force_feed_tokens = {512, 789, 230, 450};

    // Debug containers for layer‑0 diagnostics
    struct ggml_tensor* debug_layer0_Q = nullptr;
    struct ggml_tensor* debug_layer0_K = nullptr;
    struct ggml_tensor* debug_layer0_V = nullptr;
    struct ggml_tensor* debug_layer0_kq_raw = nullptr;
    struct ggml_tensor* debug_layer0_kq_scaled = nullptr;
    struct ggml_tensor* debug_layer0_kq_masked = nullptr;
    struct ggml_tensor* debug_layer0_mask = nullptr;
    struct ggml_tensor* debug_layer0_kq = nullptr;
    struct ggml_tensor* debug_layer0_kqv = nullptr;
    struct ggml_tensor* debug_layer0_attn_out = nullptr;
    struct ggml_tensor* debug_layer0_x_attn = nullptr;
    struct ggml_tensor* debug_layer0_mlp_out = nullptr;
    struct ggml_tensor* debug_layer0_x_out = nullptr;
    struct ggml_tensor* debug_layer0_x_in = nullptr;
    std::vector<struct ggml_tensor*> debug_layers_out(24, nullptr);


    int total_decoded = 0;
    while (total_decoded <= 4) { // deterministic demo – 5 steps total
        int audio_len = (int)current_audio_ids.size();
        int total_len = text_len + audio_len;
        if (total_len >= 512) break;

        // Per‑step graph context
        struct ggml_init_params init_params = {96 * 1024 * 1024, nullptr, true};
        struct ggml_context* ctx_step = ggml_init(init_params);
        struct ggml_cgraph* cgraph = ggml_new_graph(ctx_step);

        struct ggml_tensor* x = nullptr;
        struct ggml_tensor* bert_features_local = nullptr;
        struct ggml_tensor* text_ids_tensor = nullptr;
        struct ggml_tensor* bert_proj_aligned = nullptr;
        struct ggml_tensor* audio_ids_tensor = nullptr;
        struct ggml_tensor* token_tensor = nullptr;
        struct ggml_tensor* text_pe = nullptr;
        struct ggml_tensor* audio_pe = nullptr;
        struct ggml_tensor* mask = nullptr;

        std::vector<float> text_pe_data;
        std::vector<float> audio_pe_data;
        std::vector<float> mask_data;

        if (total_decoded == 0) {
            // Initial full context
            bert_features_local = ggml_new_tensor_2d(ctx_step, GGML_TYPE_F32, 1024, text_len);
            bert_proj_aligned = ggml_add(ctx_step,
                ggml_mul_mat(ctx_step, bert_proj_w, bert_features_local),
                ggml_reshape_2d(ctx_step, bert_proj_b, ggml_nelements(bert_proj_b), 1));

            // Text embeddings
            text_ids_tensor = ggml_new_tensor_1d(ctx_step, GGML_TYPE_I32, text_len);
            struct ggml_tensor* t_emb = ggml_get_rows(ctx_step, text_embed, text_ids_tensor);
            struct ggml_tensor* text_fused = ggml_add(ctx_step, t_emb, bert_proj_aligned);
            text_pe = ggml_new_tensor_2d(ctx_step, GGML_TYPE_F32, 512, text_len);
            text_pe_data = compute_positional_embeddings(text_len, 512, text_alpha);
            struct ggml_tensor* text_rep = ggml_add(ctx_step, text_fused, text_pe);

            // Audio embeddings
            audio_ids_tensor = ggml_new_tensor_1d(ctx_step, GGML_TYPE_I32, audio_len);
            struct ggml_tensor* a_emb = ggml_get_rows(ctx_step, audio_embed, audio_ids_tensor);
            audio_pe = ggml_new_tensor_2d(ctx_step, GGML_TYPE_F32, 512, audio_len);
            audio_pe_data = compute_positional_embeddings(audio_len, 512, audio_alpha);
            struct ggml_tensor* audio_rep = ggml_add(ctx_step, a_emb, audio_pe);

            // Concatenate
            x = ggml_concat(ctx_step, text_rep, audio_rep, 1);
            debug_layer0_x_in = x;

        } else {
            // Decode single token using KV cache
            token_tensor = ggml_new_tensor_1d(ctx_step, GGML_TYPE_I32, 1);
            x = ggml_get_rows(ctx_step, audio_embed, token_tensor);
            int pos_idx = audio_len - 1;
            audio_pe = ggml_new_tensor_2d(ctx_step, GGML_TYPE_F32, 512, pos_idx + 1);
            audio_pe_data = compute_positional_embeddings(pos_idx + 1, 512, audio_alpha);
            struct ggml_tensor* audio_pe_single = ggml_view_2d(ctx_step, audio_pe, 512, 1, audio_pe->nb[1], pos_idx * audio_pe->nb[1]);
            x = ggml_add(ctx_step, x, audio_pe_single);
        }

        // Causal mask for first step
        if (total_decoded == 0) {
            mask = ggml_new_tensor_3d(ctx_step, GGML_TYPE_F32, total_len, total_len, model.n_heads);
            mask_data = compute_prefix_causal_mask(total_len, text_len, model.n_heads);
        }

        int q_len = (total_decoded == 0) ? total_len : 1;
        const int n_heads = model.n_heads;
        const int head_dim = model.head_dim;

        // Transformer layers
        for (int layer = 0; layer < 24; ++layer) {
            std::string prefix = "h.layers." + std::to_string(layer) + ".";
            struct ggml_tensor* qw = model.get_tensor(prefix + "self_attn.q.weight");
            struct ggml_tensor* qb = model.get_tensor(prefix + "self_attn.q.bias");
            struct ggml_tensor* kw = model.get_tensor(prefix + "self_attn.k.weight");
            struct ggml_tensor* kb = model.get_tensor(prefix + "self_attn.k.bias");
            struct ggml_tensor* vw = model.get_tensor(prefix + "self_attn.v.weight");
            struct ggml_tensor* vb = model.get_tensor(prefix + "self_attn.v.bias");
            struct ggml_tensor* out_w = model.get_tensor(prefix + "self_attn.out_proj.weight");
            struct ggml_tensor* out_b = model.get_tensor(prefix + "self_attn.out_proj.bias");
            struct ggml_tensor* ln1_w = model.get_tensor(prefix + "norm1.weight");
            struct ggml_tensor* ln1_b = model.get_tensor(prefix + "norm1.bias");
            struct ggml_tensor* ln2_w = model.get_tensor(prefix + "norm2.weight");
            struct ggml_tensor* ln2_b = model.get_tensor(prefix + "norm2.bias");
            struct ggml_tensor* ffn_w1 = model.get_tensor(prefix + "linear1.weight");
            struct ggml_tensor* ffn_b1 = model.get_tensor(prefix + "linear1.bias");
            struct ggml_tensor* ffn_w2 = model.get_tensor(prefix + "linear2.weight");
            struct ggml_tensor* ffn_b2 = model.get_tensor(prefix + "linear2.bias");
            if (!qw || !qb || !kw || !kb || !vw || !vb || !out_w || !out_b || !ln1_w || !ln1_b || !ln2_w || !ln2_b || !ffn_w1 || !ffn_b1 || !ffn_w2 || !ffn_b2) {
                std::cerr << "[T2S] Error: Missing layer " << layer << " weights!\n";
                ggml_free(ctx_step);
                return 1;
            }

            // Q,K,V projections
            struct ggml_tensor* Q = ggml_add(ctx_step, ggml_mul_mat(ctx_step, qw, x), qb);
            struct ggml_tensor* K = ggml_add(ctx_step, ggml_mul_mat(ctx_step, kw, x), kb);
            struct ggml_tensor* V = ggml_add(ctx_step, ggml_mul_mat(ctx_step, vw, x), vb);

            Q = ggml_reshape_3d(ctx_step, Q, head_dim, n_heads, q_len);
            K = ggml_reshape_3d(ctx_step, K, head_dim, n_heads, q_len);
            V = ggml_reshape_3d(ctx_step, V, head_dim, n_heads, q_len);

            // Permute and copy into KV cache (unchanged)
            struct ggml_tensor* K_perm = ggml_permute(ctx_step, K, 0, 2, 1, 3);
            struct ggml_tensor* K_cont = ggml_cont(ctx_step, K_perm);
            struct ggml_tensor* V_perm = ggml_permute(ctx_step, V, 0, 2, 1, 3);
            struct ggml_tensor* V_cont = ggml_cont(ctx_step, V_perm);

            struct ggml_tensor* K_dest = nullptr;
            struct ggml_tensor* V_dest = nullptr;
            if (total_decoded == 0) {
                int64_t offset_bytes = layer * model.kv_k->nb[3];
                K_dest = ggml_view_3d(ctx_step, model.kv_k, head_dim, total_len, n_heads,
                    model.kv_k->nb[1], model.kv_k->nb[2], offset_bytes);
                V_dest = ggml_view_3d(ctx_step, model.kv_v, head_dim, total_len, n_heads,
                    model.kv_v->nb[1], model.kv_v->nb[2], offset_bytes);
            } else {
                int pos_idx = total_len - 1;
                int64_t offset_bytes = layer * model.kv_k->nb[3] + pos_idx * model.kv_k->nb[1];
                K_dest = ggml_view_3d(ctx_step, model.kv_k, head_dim, 1, n_heads,
                    model.kv_k->nb[1], model.kv_k->nb[2], offset_bytes);
                V_dest = ggml_view_3d(ctx_step, model.kv_v, head_dim, 1, n_heads,
                    model.kv_v->nb[1], model.kv_v->nb[2], offset_bytes);
            }
            struct ggml_tensor* K_cpy = ggml_cpy(ctx_step, K_cont, K_dest);
            struct ggml_tensor* V_cpy = ggml_cpy(ctx_step, V_cont, V_dest);
            ggml_build_forward_expand(cgraph, K_cpy);
            ggml_build_forward_expand(cgraph, V_cpy);

            // Cached views for attention
            struct ggml_tensor* K_cached = ggml_view_3d(ctx_step, model.kv_k, head_dim, total_len, n_heads,
                model.kv_k->nb[1], model.kv_k->nb[2], layer * model.kv_k->nb[3]);
            struct ggml_tensor* V_cached = ggml_view_3d(ctx_step, model.kv_v, head_dim, total_len, n_heads,
                model.kv_v->nb[1], model.kv_v->nb[2], layer * model.kv_v->nb[3]);

            struct ggml_tensor* Q_perm = ggml_permute(ctx_step, Q, 0, 2, 1, 3);
            struct ggml_tensor* K_cached_perm = K_cached;
            struct ggml_tensor* V_cached_perm = ggml_permute(ctx_step, V_cached, 1, 0, 2, 3);

            struct ggml_tensor* Q_cont = ggml_cont(ctx_step, Q_perm);
            struct ggml_tensor* K_cont_cached = ggml_cont(ctx_step, K_cached_perm);
            struct ggml_tensor* kq = ggml_mul_mat(ctx_step, Q_cont, K_cont_cached); // [Query, Key, Head]
            kq = ggml_transpose(ctx_step, kq); // swap to [Key, Query, Head]
            kq = ggml_cont(ctx_step, kq);
            struct ggml_tensor* kq_scaled = ggml_scale(ctx_step, kq, 1.0f / std::sqrt((float)head_dim));
            struct ggml_tensor* kq_masked = mask ? ggml_add(ctx_step, kq_scaled, mask) : kq_scaled;
            struct ggml_tensor* kq_soft = ggml_soft_max(ctx_step, kq_masked);
            struct ggml_tensor* V_cont_cached = ggml_cont(ctx_step, V_cached_perm);
            if (layer == 0) {
                std::cout << "[T2S Align Debug] Step " << total_decoded << " layer " << layer 
                          << " V_cont_cached shape: " << V_cont_cached->ne[0] << "x" << V_cont_cached->ne[1] 
                          << "x" << V_cont_cached->ne[2] << "x" << V_cont_cached->ne[3] << "\n";
                std::cout << "[T2S Align Debug] Step " << total_decoded << " layer " << layer 
                          << " kq_soft shape: " << kq_soft->ne[0] << "x" << kq_soft->ne[1] 
                          << "x" << kq_soft->ne[2] << "x" << kq_soft->ne[3] << "\n";
            }
            struct ggml_tensor* kqv = ggml_mul_mat(ctx_step, V_cont_cached, kq_soft);
            kqv = ggml_permute(ctx_step, kqv, 0, 2, 1, 3);
            kqv = ggml_cont(ctx_step, kqv);
            kqv = ggml_reshape_2d(ctx_step, kqv, 512, q_len);

            // Attention output projection
            struct ggml_tensor* attn_out = ggml_add(ctx_step, ggml_mul_mat(ctx_step, out_w, kqv), out_b);

            // Residual + LayerNorm 1
            struct ggml_tensor* x_attn = ggml_add(ctx_step, x, attn_out);
            x_attn = ggml_norm(ctx_step, x_attn, 1e-5f);
            x_attn = ggml_add(ctx_step, ggml_mul(ctx_step, x_attn, ln1_w), ln1_b);

            // Feed‑Forward MLP
            struct ggml_tensor* h = ggml_add(ctx_step, ggml_mul_mat(ctx_step, ffn_w1, x_attn), ffn_b1);
            struct ggml_tensor* h_act = ggml_relu(ctx_step, h); // ReLU activation
            struct ggml_tensor* mlp_out = ggml_add(ctx_step, ggml_mul_mat(ctx_step, ffn_w2, h_act), ffn_b2);

            // Residual + LayerNorm 2
            x = ggml_add(ctx_step, x_attn, mlp_out);
            x = ggml_norm(ctx_step, x, 1e-5f);
            x = ggml_add(ctx_step, ggml_mul(ctx_step, x, ln2_w), ln2_b);

            // Capture diagnostics for layer 0 at step 0
            if (layer == 0 && total_decoded == 0) {
                debug_layer0_Q = Q;
                debug_layer0_K = K;
                debug_layer0_V = V;
                debug_layer0_kq_raw = kq;
                debug_layer0_kq_scaled = kq_scaled;
                debug_layer0_kq_masked = kq_masked;
                debug_layer0_mask = mask;
                debug_layer0_kq = kq_soft;
                debug_layer0_kqv = kqv;
                debug_layer0_attn_out = attn_out;
                debug_layer0_x_attn = x_attn;
                debug_layer0_mlp_out = mlp_out;
                debug_layer0_x_out = x;
            }
            if (total_decoded == 0) debug_layers_out[layer] = x;
        }

        // Predict logits for the last token
        struct ggml_tensor* last_token_rep = ggml_view_2d(ctx_step, x, head_dim * n_heads, 1, x->nb[1], (q_len - 1) * x->nb[1]);
        struct ggml_tensor* logits_tensor = ggml_mul_mat(ctx_step, predict_w, last_token_rep);
        ggml_build_forward_expand(cgraph, logits_tensor);

        // Allocate backend buffers and set inputs
        ggml_backend_buffer_t step_buf = ggml_backend_alloc_ctx_tensors(ctx_step, backend);
        if (step_buf) {
            if (total_decoded == 0) {
                ggml_backend_tensor_set(text_ids_tensor, text_ids.data(), 0, text_len * sizeof(int32_t));
                ggml_backend_tensor_set(bert_features_local, bert_features_host.data(), 0, bert_features_host.size() * sizeof(float));
                ggml_backend_tensor_set(audio_ids_tensor, current_audio_ids.data(), 0, audio_len * sizeof(int32_t));
                ggml_backend_tensor_set(text_pe, text_pe_data.data(), 0, text_pe_data.size() * sizeof(float));
                ggml_backend_tensor_set(audio_pe, audio_pe_data.data(), 0, audio_pe_data.size() * sizeof(float));
                ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(float));
            } else {
                int latest_token = current_audio_ids.back();
                ggml_backend_tensor_set(token_tensor, &latest_token, 0, sizeof(int32_t));
                ggml_backend_tensor_set(audio_pe, audio_pe_data.data(), 0, audio_pe_data.size() * sizeof(float));
            }
        }

        ggml_backend_graph_compute(backend, cgraph);

        // Save diagnostic tensors after first step
        if (total_decoded == 0) {
            save_tensor(out_prefix + "_debug_layer0_x_in.f32", debug_layer0_x_in);
            save_tensor(out_prefix + "_debug_layer0_Q.f32", debug_layer0_Q);

            save_tensor(out_prefix + "_debug_layer0_K.f32", debug_layer0_K);
            save_tensor(out_prefix + "_debug_layer0_V.f32", debug_layer0_V);
            save_tensor(out_prefix + "_debug_layer0_kq_raw.f32", debug_layer0_kq_raw);
            save_tensor(out_prefix + "_debug_layer0_kq_scaled.f32", debug_layer0_kq_scaled);
            save_tensor(out_prefix + "_debug_layer0_kq_masked.f32", debug_layer0_kq_masked);
            save_tensor(out_prefix + "_debug_layer0_mask.f32", debug_layer0_mask);
            save_tensor(out_prefix + "_debug_layer0_kq.f32", debug_layer0_kq);
            save_tensor(out_prefix + "_debug_layer0_kqv.f32", debug_layer0_kqv);
            save_tensor(out_prefix + "_debug_layer0_attn_out.f32", debug_layer0_attn_out);
            save_tensor(out_prefix + "_debug_layer0_x_attn.f32", debug_layer0_x_attn);
            save_tensor(out_prefix + "_debug_layer0_mlp_out.f32", debug_layer0_mlp_out);
            save_tensor(out_prefix + "_debug_layer0_x_out.f32", debug_layer0_x_out);
            for (int l = 0; l < 24; ++l) {
                if (debug_layers_out[l]) {
                    save_tensor(out_prefix + "_debug_layer" + std::to_string(l) + "_x_out.f32", debug_layers_out[l]);
                }
            }
            std::cout << "[T2S Align] Saved debug tensors for t=0.\n";
        }

        // Retrieve logits
        std::vector<float> logits(1025);
        ggml_backend_tensor_get(logits_tensor, logits.data(), 0, 1025 * sizeof(float));

        // Cleanup step
        if (step_buf) ggml_backend_buffer_free(step_buf);
        ggml_free(ctx_step);

        std::string step_label = "Step t=" + std::to_string(total_decoded);
        print_top_logits(step_label, logits);
        write_f32_binary(out_prefix + "_logits_t" + std::to_string(total_decoded) + ".f32", logits);

        // Prepare next token (deterministic demo)
        if (total_decoded < 4) {
            current_audio_ids.push_back(force_feed_tokens[total_decoded]);
        }
        ++total_decoded;
    }

    std::cout << "[T2S Align] Finished deterministic alignment demo.\n";
    ggml_backend_free(backend);
    return 0;
}
