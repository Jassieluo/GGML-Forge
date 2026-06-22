#include "gpt_t2s.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ops/ops.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <random>
#include <map>

namespace gpt_sovits {

namespace {

static std::vector<float> compute_positional_embeddings(int seq_len, int hidden_dim, float alpha, int pos_offset = 0) {
    std::vector<float> data(seq_len * hidden_dim, 0.0f);
    for (int i_pos = 0; i_pos < seq_len; ++i_pos) {
        int pos = i_pos + pos_offset;
        for (int i = 0; i < hidden_dim / 2; ++i) {
            float denom = std::pow(10000.0f, (2.0f * i) / (float)hidden_dim);
            float val = (float)pos / denom;
            data[i_pos * hidden_dim + 2 * i] = alpha * std::sin(val);
            data[i_pos * hidden_dim + 2 * i + 1] = alpha * std::cos(val);
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

// DoubleSwish activation function matching PyTorch: x * sigmoid(x - 1.0f)
static struct ggml_tensor* ggml_double_swish(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend) {
    return ggml_ops_double_swish(ctx, x, backend);
}

static void save_tensor_binary(const std::string& path, struct ggml_tensor* tensor) {
    if (!tensor) return;
    int64_t out_elements = ggml_nelements(tensor);
    std::vector<float> host_out(out_elements);
    ggml_backend_tensor_get(tensor, host_out.data(), 0, out_elements * sizeof(float));
    std::ofstream file(path, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(host_out.data()), out_elements * sizeof(float));
    }
}

// Low-level sampler for top-k/top-p logits sampling
static int32_t sample_logits(
    std::vector<float>& logits, 
    const std::vector<int32_t>& current_audio_ids,
    float temp, int top_k, float top_p, float rep_penalty, bool is_greedy
) {
    // 1. Repetition penalty
    if (rep_penalty != 1.0f) {
        std::vector<bool> seen(1025, false);
        for (int id : current_audio_ids) {
            if (id >= 0 && id < 1025) seen[id] = true;
        }
        for (int i = 0; i < (int)logits.size(); ++i) {
            if (seen[i]) {
                if (logits[i] < 0.0f) {
                    logits[i] *= rep_penalty;
                } else {
                    logits[i] /= rep_penalty;
                }
            }
        }
    }

    // 2. Greedy decoding
    if (is_greedy) {
        return (int32_t)(std::max_element(logits.begin(), logits.end()) - logits.begin());
    }

    // 3. Top-k filtering
    std::vector<std::pair<float, int>> indexed_logits(logits.size());
    for (int i = 0; i < (int)logits.size(); ++i) {
        indexed_logits[i] = {logits[i], i};
    }
    std::sort(indexed_logits.begin(), indexed_logits.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

    if (top_k > 0 && top_k < (int)logits.size()) {
        indexed_logits.resize(top_k);
    }

    // Apply temperature scaling
    float sum_exp = 0.0f;
    std::vector<float> probs(indexed_logits.size());
    float max_l = indexed_logits[0].first;
    for (int i = 0; i < (int)indexed_logits.size(); ++i) {
        probs[i] = std::exp((indexed_logits[i].first - max_l) / temp);
        sum_exp += probs[i];
    }
    for (int i = 0; i < (int)indexed_logits.size(); ++i) {
        probs[i] /= sum_exp;
    }

    // 4. Top-p filtering
    if (top_p > 0.0f && top_p < 1.0f) {
        float cumsum = 0.0f;
        int cutoff = 0;
        for (int i = 0; i < (int)probs.size(); ++i) {
            cumsum += probs[i];
            if (cumsum >= top_p) {
                cutoff = i;
                break;
            }
        }
        probs.resize(cutoff + 1);
        indexed_logits.resize(cutoff + 1);
        float p_sum = 0.0f;
        for (float p : probs) p_sum += p;
        for (float& p : probs) p /= p_sum;
    }

    // 5. Categorical sampling
    static constexpr uint32_t default_seed = 42u;
    static std::mt19937 gen(default_seed);
    static thread_local std::string last_seed_env;
    const char * env_seed = std::getenv("T2S_RANDOM_SEED");
    const std::string seed_env = env_seed ? env_seed : "";
    if (seed_env != last_seed_env) {
        if (!seed_env.empty()) {
            gen.seed((uint32_t) std::strtoul(seed_env.c_str(), nullptr, 10));
        } else {
            gen.seed(default_seed);
        }
        last_seed_env = seed_env;
    }
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int sampled_idx = dist(gen);
    return indexed_logits[sampled_idx].second;
}

} // namespace

void T2SModel::on_read_metadata(struct gguf_context* ctx_gguf) {
    int kid_layers = gguf_find_key(ctx_gguf, "gpt_sovits.t2s.n_layers");
    if (kid_layers >= 0) {
        n_layers = (int)gguf_get_val_u32(ctx_gguf, kid_layers);
    } else {
        n_layers = 24;
    }
    int kid_heads = gguf_find_key(ctx_gguf, "attention.head_count");
    if (kid_heads >= 0) {
        n_heads = (int)gguf_get_val_u32(ctx_gguf, kid_heads);
    }
}

bool T2SModel::load(const std::string& path, ggml_backend_t backend) {
    if (!load_gguf_model(path, *this, backend)) {
        return false;
    }

    // Pre-convert FP16 weights to FP32 for non-CUDA (CPU/SYCL) backends to eliminate 144 dynamic casts per step!
    bool is_cuda = false;
    if (backend) {
        const char * bname = ggml_backend_name(backend);
        if (bname && strncmp(bname, "CUDA", 4) == 0) {
            is_cuda = true;
        }
    }

    if (!is_cuda) {
        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[T2S load] Non-CUDA backend detected. Pre-converting loaded FP16 weights to FP32..." << std::endl;
        struct ggml_init_params custom_params = {
            /* .mem_size   = */ 16 * 1024 * 1024, // 16MB metadata pool, extremely safe!
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true
        };
        custom_ctx = ggml_init(custom_params);
        if (!custom_ctx) {
            std::cerr << "[T2S load] Error: Failed to initialize custom_ctx for FP32 weights!" << std::endl;
            return false;
        }

        struct UploadF32Entry {
            std::string name;
            std::vector<float> data;
        };
        std::vector<UploadF32Entry> fp32_upload_list;
        std::vector<struct ggml_tensor*> fp32_tensors_list;

        for (const auto& pair : tensors) {
            struct ggml_tensor* old_w = pair.second;
            if (!old_w || old_w->type != GGML_TYPE_F16) continue;

            // Skip embedding tensors to let get_rows run on FP16 (as SYCL get_rows crashes on FP32)
            if (pair.first.find("embedding") != std::string::npos) {
                continue;
            }

            // Also skip position embedding tables
            if (pair.first.find("pos_embed") != std::string::npos || pair.first.find("position_embeddings") != std::string::npos) {
                continue;
            }

            int64_t nelems = ggml_nelements(old_w);
            std::vector<float> w_f32_data(nelems);
            if (old_w->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> f16_buf(nelems);
                ggml_backend_tensor_get(old_w, f16_buf.data(), 0, nelems * sizeof(ggml_fp16_t));
                ggml_fp16_to_fp32_row(f16_buf.data(), w_f32_data.data(), nelems);
            }

            struct ggml_tensor* new_w = ggml_new_tensor(custom_ctx, GGML_TYPE_F32, ggml_n_dims(old_w), old_w->ne);
            ggml_set_name(new_w, old_w->name);

            fp32_tensors_list.push_back(new_w);
            fp32_upload_list.push_back({pair.first, w_f32_data});
        }

        custom_buffer = ggml_backend_alloc_ctx_tensors(custom_ctx, backend);
        if (!custom_buffer) {
            std::cerr << "[T2S load] Error: Failed to allocate custom_buffer for FP32 weights!" << std::endl;
            return false;
        }

        for (size_t i = 0; i < fp32_tensors_list.size(); ++i) {
            struct ggml_tensor* nt = fp32_tensors_list[i];
            const auto& upload_entry = fp32_upload_list[i];
            ggml_backend_tensor_set(nt, upload_entry.data.data(), 0, upload_entry.data.size() * sizeof(float));
            tensors[upload_entry.name] = nt;
        }
        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[T2S load] Pre-converted and uploaded " << fp32_tensors_list.size() << " weights to FP32 successfully." << std::endl;
    }

    // Dynamically retrieve head dim
    struct ggml_tensor* qw = get_tensor("h.layers.0.self_attn.q.weight");
    if (qw) {
        int hidden_dim = (int)qw->ne[0];
        head_dim = hidden_dim / n_heads;
        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[T2S] Dynamically configured attention heads: " << n_heads << ", head_dim: " << head_dim << " (hidden_dim=" << hidden_dim << ")" << std::endl;
    } else {
        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[T2S] Warning: self_attn.q.weight not found. Defaulting to head_dim: " << head_dim << ", n_heads: " << n_heads << std::endl;
    }

    // Allocate GPU resident Keys and Values KV Cache (Native [head_dim, 512, n_heads, n_layers] shapes)
    struct ggml_init_params kv_params = {
        /* .mem_size   = */ 2 * 1024 * 1024, // Metadata size
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true
    };
    kv_ctx = ggml_init(kv_params);
    kv_k = ggml_new_tensor_4d(kv_ctx, GGML_TYPE_F32, head_dim, 512, n_heads, n_layers);
    kv_v = ggml_new_tensor_4d(kv_ctx, GGML_TYPE_F32, head_dim, 512, n_heads, n_layers);

    kv_buffer = ggml_backend_alloc_ctx_tensors(kv_ctx, backend);
    if (kv_buffer) {
        size_t total_elements = (size_t)head_dim * 512 * n_heads * n_layers;
        std::vector<float> zero_buf(total_elements, 0.0f);
        ggml_backend_tensor_set(kv_k, zero_buf.data(), 0, total_elements * sizeof(float));
        ggml_backend_tensor_set(kv_v, zero_buf.data(), 0, total_elements * sizeof(float));
        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[T2S] GPU Resident KV Cache (" << n_layers << " layers, " << (total_elements * sizeof(float) * 2 / (1024 * 1024)) << " MB VRAM) allocated and zeroed out successfully.\n";
    } else {
        std::cerr << "[T2S] Failed to allocate GPU resident KV Cache!\n";
        return false;
    }

    return true;
}

std::vector<int32_t> T2SModel::forward(
    struct ggml_context* ctx_graph, 
    const std::vector<int32_t>& prompt_phones,
    const std::vector<int32_t>& target_phones,
    const std::vector<int32_t>& prompt_semantics,
    struct ggml_tensor* bert_features,
    const std::vector<int32_t>& target_word2ph,
    int max_len,
    ggml_backend_t backend
) {
    bool align_mode = (std::getenv("T2S_ALIGNMENT") != nullptr);
    std::cout << "[GPT-SoVITS Debug] Running T2S forward...\n";

    // Setup models tensors
    struct ggml_tensor* text_embed = get_tensor("ar_text_embedding.word_embeddings.weight");
    struct ggml_tensor* audio_embed = get_tensor("ar_audio_embedding.word_embeddings.weight");
    struct ggml_tensor* bert_proj_w = get_tensor("bert_proj.weight");
    struct ggml_tensor* bert_proj_b = get_tensor("bert_proj.bias");
    struct ggml_tensor* ar_text_position_alpha = get_tensor("ar_text_position.alpha");
    struct ggml_tensor* ar_audio_position_alpha = get_tensor("ar_audio_position.alpha");
    struct ggml_tensor* predict_w = get_tensor("ar_predict_layer.weight");
    if (!text_embed || !audio_embed || !bert_proj_w || !bert_proj_b || !ar_text_position_alpha || !ar_audio_position_alpha || !predict_w) {
        std::cerr << "[T2S] Error: Missing model weights in GGUF weight mapping!\n";
        return {};
    }

    float text_alpha = 1.0f;
    float audio_alpha = 1.0f;
    ggml_backend_tensor_get(ar_text_position_alpha, &text_alpha, 0, sizeof(float));
    ggml_backend_tensor_get(ar_audio_position_alpha, &audio_alpha, 0, sizeof(float));
    int text_len = (int)(prompt_phones.size() + target_phones.size());
    std::vector<int32_t> text_ids;
    text_ids.reserve(text_len);
    text_ids.insert(text_ids.end(), prompt_phones.begin(), prompt_phones.end());
    text_ids.insert(text_ids.end(), target_phones.begin(), target_phones.end());

    std::vector<int32_t> current_audio_ids = prompt_semantics;

    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[T2S Debug] forward start. text_len=" << text_len << " audio_len=" << current_audio_ids.size() << std::endl; std::fflush(stdout);

    std::vector<int32_t> generated_semantics;
    int total_decoded = 0;

    // Target debug tensors
    struct ggml_tensor* layer0_Q = nullptr;
    struct ggml_tensor* layer0_K = nullptr;
    struct ggml_tensor* layer0_V = nullptr;
    struct ggml_tensor* layer0_kq = nullptr;
    struct ggml_tensor* layer0_kqv = nullptr;
    struct ggml_tensor* layer0_attn_out = nullptr;
    struct ggml_tensor* layer0_x_attn = nullptr;
    struct ggml_tensor* layer0_mlp_out = nullptr;
    struct ggml_tensor* layer0_out = nullptr;

    struct ggml_tensor* a_emb = nullptr;
    struct ggml_tensor* audio_rep = nullptr;

    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[T2S Debug] Creating graph allocator..." << std::endl; std::fflush(stdout);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!galloc) {
        std::cerr << "[T2S] Error: Failed to create graph allocator (gallocr)!\n";
        return {};
    }

    // Context size 4MB, allocated once and reset per step to stabilize graph keys
    struct ggml_init_params init_params = { 4 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_step = ggml_init(init_params);
    
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[T2S Debug] Entering autoregressive loop..." << std::endl; std::fflush(stdout);

    while (total_decoded < max_len) {
        int audio_len = (int)current_audio_ids.size();
        int total_len = text_len + audio_len;
        if (total_len >= 512) break;

        ggml_reset(ctx_step);
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

        struct ggml_tensor* t_emb = nullptr;
        struct ggml_tensor* text_fused = nullptr;
        struct ggml_tensor* text_rep = nullptr;
        struct ggml_tensor* x_concat = nullptr;

        if (total_decoded == 0) {
            // First step: Process prompt phones and prompt semantics entirely
            bert_features_local = ggml_new_tensor_2d(ctx_step, GGML_TYPE_F32, 1024, text_len);
            
            bert_proj_aligned = ggml_add(ctx_step,
                ggml_mul_mat(ctx_step, bert_proj_w, bert_features_local),
                ggml_reshape_2d(ctx_step, bert_proj_b, ggml_nelements(bert_proj_b), 1)
            );

            // Text embeddings
            text_ids_tensor = ggml_new_tensor_1d(ctx_step, GGML_TYPE_I32, text_len);
            t_emb = ggml_get_rows(ctx_step, text_embed, text_ids_tensor);
            text_fused = ggml_add(ctx_step, t_emb, bert_proj_aligned);
            text_pe = ggml_new_tensor_2d(ctx_step, GGML_TYPE_F32, 512, text_len);
            text_pe_data = compute_positional_embeddings(text_len, 512, text_alpha);
            text_rep = ggml_add(ctx_step, text_fused, text_pe);

            // Audio embeddings
            audio_ids_tensor = ggml_new_tensor_1d(ctx_step, GGML_TYPE_I32, audio_len);
            a_emb = ggml_get_rows(ctx_step, audio_embed, audio_ids_tensor);
            audio_pe = ggml_new_tensor_2d(ctx_step, GGML_TYPE_F32, 512, audio_len);
            audio_pe_data = compute_positional_embeddings(audio_len, 512, audio_alpha);
            audio_rep = ggml_add(ctx_step, a_emb, audio_pe);

            x_concat = ggml_concat(ctx_step, text_rep, audio_rep, 1);
            x = x_concat;
        } else {
            // Self-regressive: Feed only the latest token
            token_tensor = ggml_new_tensor_1d(ctx_step, GGML_TYPE_I32, 1);
            x = ggml_get_rows(ctx_step, audio_embed, token_tensor);

            int pos_idx = audio_len - 1;
            audio_pe = ggml_new_tensor_1d(ctx_step, GGML_TYPE_F32, 512);
            audio_pe_data = compute_positional_embeddings(1, 512, audio_alpha, pos_idx);
            x = ggml_add(ctx_step, x, audio_pe);
        }

        // Generate mask
        if (total_decoded == 0) {
            mask = ggml_new_tensor_3d(ctx_step, GGML_TYPE_F32, total_len, total_len, n_heads);
            mask_data = compute_prefix_causal_mask(total_len, text_len, n_heads);
        }

        int q_len = (total_decoded == 0) ? total_len : 1;
        const int hidden_dim = n_heads * head_dim;

        auto mul_f32 = [&](struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b) -> struct ggml_tensor* {
            struct ggml_tensor* r = ggml_mul_mat(ctx, a, (b->type == GGML_TYPE_F32) ? b : ggml_cast(ctx, b, GGML_TYPE_F32));
            ggml_mul_mat_set_prec(r, GGML_PREC_DEFAULT);
            return r;
        };

        // Execute attention layers
        for (int layer = 0; layer < n_layers; ++layer) {
            std::string prefix = "h.layers." + std::to_string(layer) + ".";
            struct ggml_tensor* qw = get_tensor(prefix + "self_attn.q.weight");
            struct ggml_tensor* qb = get_tensor(prefix + "self_attn.q.bias");
            struct ggml_tensor* kw = get_tensor(prefix + "self_attn.k.weight");
            struct ggml_tensor* kb = get_tensor(prefix + "self_attn.k.bias");
            struct ggml_tensor* vw = get_tensor(prefix + "self_attn.v.weight");
            struct ggml_tensor* vb = get_tensor(prefix + "self_attn.v.bias");
            struct ggml_tensor* out_w = get_tensor(prefix + "self_attn.out_proj.weight");
            struct ggml_tensor* out_b = get_tensor(prefix + "self_attn.out_proj.bias");
            struct ggml_tensor* ln1_w = get_tensor(prefix + "norm1.weight");
            struct ggml_tensor* ln1_b = get_tensor(prefix + "norm1.bias");
            struct ggml_tensor* ln2_w = get_tensor(prefix + "norm2.weight");
            struct ggml_tensor* ln2_b = get_tensor(prefix + "norm2.bias");
            struct ggml_tensor* ffn_w1 = get_tensor(prefix + "linear1.weight");
            struct ggml_tensor* ffn_b1 = get_tensor(prefix + "linear1.bias");
            struct ggml_tensor* ffn_w2 = get_tensor(prefix + "linear2.weight");
            struct ggml_tensor* ffn_b2 = get_tensor(prefix + "linear2.bias");

            if (!qw || !qb || !kw || !kb || !vw || !vb || !out_w || !out_b || !ln1_w || !ln1_b || !ln2_w || !ln2_b || !ffn_w1 || !ffn_b1 || !ffn_w2 || !ffn_b2) {
                std::cerr << "[T2S] Error: Missing layer " << layer << " weights in GGUF weight mapping!\n";
                ggml_free(ctx_step);
                ggml_gallocr_free(galloc);
                return {};
            }

            struct ggml_tensor* Q = ggml_add(ctx_step, mul_f32(ctx_step, qw, x), qb);
            struct ggml_tensor* K = ggml_add(ctx_step, mul_f32(ctx_step, kw, x), kb);
            struct ggml_tensor* V = ggml_add(ctx_step, mul_f32(ctx_step, vw, x), vb);

            if (layer == 0) {
                layer0_Q = Q;
                layer0_K = K;
                layer0_V = V;
            }

            Q = ggml_reshape_3d(ctx_step, Q, head_dim, n_heads, q_len);
            K = ggml_reshape_3d(ctx_step, K, head_dim, n_heads, q_len);
            V = ggml_reshape_3d(ctx_step, V, head_dim, n_heads, q_len);

            // Permute K and V to [head_dim, q_len, n_heads]
            struct ggml_tensor* K_perm = ggml_permute(ctx_step, K, 0, 2, 1, 3);
            struct ggml_tensor* K_cont = ggml_cont(ctx_step, K_perm);
            struct ggml_tensor* V_perm = ggml_permute(ctx_step, V, 0, 2, 1, 3);
            struct ggml_tensor* V_cont = ggml_cont(ctx_step, V_perm);

            struct ggml_tensor* K_dest = nullptr;
            struct ggml_tensor* V_dest = nullptr;

            if (total_decoded == 0) {
                int64_t offset_bytes = layer * kv_k->nb[3];
                K_dest = ggml_view_3d(ctx_step, kv_k, head_dim, total_len, n_heads,
                    kv_k->nb[1], kv_k->nb[2], offset_bytes);
                V_dest = ggml_view_3d(ctx_step, kv_v, head_dim, total_len, n_heads,
                    kv_v->nb[1], kv_v->nb[2], offset_bytes);
            } else {
                int pos_idx = total_len - 1;
                int64_t offset_bytes = layer * kv_k->nb[3] + pos_idx * kv_k->nb[1];
                K_dest = ggml_view_3d(ctx_step, kv_k, head_dim, 1, n_heads,
                    kv_k->nb[1], kv_k->nb[2], offset_bytes);
                V_dest = ggml_view_3d(ctx_step, kv_v, head_dim, 1, n_heads,
                    kv_v->nb[1], kv_v->nb[2], offset_bytes);
            }

            struct ggml_tensor* K_cpy = ggml_cpy(ctx_step, K_cont, K_dest);
            struct ggml_tensor* V_cpy = ggml_cpy(ctx_step, V_cont, V_dest);
            ggml_build_forward_expand(cgraph, K_cpy);
            ggml_build_forward_expand(cgraph, V_cpy);

            // Active views
            struct ggml_tensor* K_cached = ggml_view_3d(ctx_step, kv_k, head_dim, total_len, n_heads,
                kv_k->nb[1], kv_k->nb[2], layer * kv_k->nb[3]);
            struct ggml_tensor* V_cached = ggml_view_3d(ctx_step, kv_v, head_dim, total_len, n_heads,
                kv_v->nb[1], kv_v->nb[2], layer * kv_v->nb[3]);

            struct ggml_tensor* kqv = nullptr;

            // Standard Causal Attention Pathway with Swapped Q/K bypass optimization
            struct ggml_tensor* Q_perm = ggml_permute(ctx_step, Q, 0, 2, 1, 3);
            struct ggml_tensor* K_cached_perm = K_cached;
            struct ggml_tensor* V_cached_perm = ggml_permute(ctx_step, V_cached, 1, 0, 2, 3);

            struct ggml_tensor* Q_cont = ggml_cont(ctx_step, Q_perm);
            struct ggml_tensor* K_cont_cached = ggml_cont(ctx_step, K_cached_perm);
            struct ggml_tensor* kq = mul_f32(ctx_step, Q_cont, K_cont_cached); // [Query, Key, Head]
            kq = ggml_transpose(ctx_step, kq); // swap to [Key, Query, Head]
            kq = ggml_cont(ctx_step, kq);
            struct ggml_tensor* kq_scaled = ggml_scale(ctx_step, kq, 1.0f / std::sqrt((float)head_dim));
            struct ggml_tensor* kq_masked = mask ? ggml_add(ctx_step, kq_scaled, mask) : kq_scaled;
            struct ggml_tensor* kq_soft = ggml_ops_soft_max(ctx_step, kq_masked, backend);

            // Safe contiguous copy of V_cached_perm for complete backend compatibility
            struct ggml_tensor* V_cont_cached = ggml_cont(ctx_step, V_cached_perm);
            kqv = mul_f32(ctx_step, V_cont_cached, kq_soft);
            kqv = ggml_permute(ctx_step, kqv, 0, 2, 1, 3);
            kqv = ggml_cont(ctx_step, kqv);
            kqv = ggml_reshape_2d(ctx_step, kqv, hidden_dim, q_len);

            if (layer == 0) {
                layer0_kq = kq_soft;
                layer0_kqv = kqv;
            }

            // Attention dense output projection
            struct ggml_tensor* attn_out = ggml_add(ctx_step, mul_f32(ctx_step, out_w, kqv), out_b);

            // Residual + LN1
            struct ggml_tensor* x_attn = ggml_add(ctx_step, x, attn_out);
            x_attn = ggml_norm(ctx_step, x_attn, 1e-5f);
            x_attn = ggml_add(ctx_step, ggml_mul(ctx_step, x_attn, ln1_w), ln1_b);

            // MLP
            struct ggml_tensor* h = ggml_add(ctx_step, mul_f32(ctx_step, ffn_w1, x_attn), ffn_b1);
            h = ggml_double_swish(ctx_step, h, backend);
            struct ggml_tensor* mlp_out = ggml_add(ctx_step, mul_f32(ctx_step, ffn_w2, h), ffn_b2);

            if (layer == 0) {
                layer0_attn_out = attn_out;
                layer0_x_attn = x_attn;
                layer0_mlp_out = mlp_out;
            }

            // Residual + LN2
            x = ggml_add(ctx_step, x_attn, mlp_out);
            x = ggml_norm(ctx_step, x, 1e-5f);
            x = ggml_add(ctx_step, ggml_mul(ctx_step, x, ln2_w), ln2_b);

            if (layer == 0) {
                layer0_out = x;
            }
        }

        // Predict logits
        struct ggml_tensor* last_token_rep = ggml_view_2d(ctx_step, x, hidden_dim, 1, x->nb[1], (q_len - 1) * x->nb[1]);
        struct ggml_tensor* logits_tensor = mul_f32(ctx_step, predict_w, last_token_rep);
        ggml_build_forward_expand(cgraph, logits_tensor);

        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[T2S Debug] Step " << total_decoded << ": allocating graph..." << std::endl; std::fflush(stdout);
        // Allocate step buffers using persistent galloc
        if (!ggml_gallocr_alloc_graph(galloc, cgraph)) {
            std::cerr << "[T2S] Error: Failed to allocate graph using gallocr!\n";
            ggml_free(ctx_step);
            ggml_gallocr_free(galloc);
            return {};
        }

        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[T2S Debug] Step " << total_decoded << ": uploading inputs..." << std::endl; std::fflush(stdout);
        if (total_decoded == 0) {
            if (text_ids_tensor) ggml_backend_tensor_set(text_ids_tensor, text_ids.data(), 0, text_len * sizeof(int32_t));
            if (bert_features_local) ggml_backend_tensor_set(bert_features_local, bert_features->data, 0, ggml_nbytes(bert_features_local));
            if (audio_ids_tensor) ggml_backend_tensor_set(audio_ids_tensor, current_audio_ids.data(), 0, audio_len * sizeof(int32_t));
            if (text_pe) ggml_backend_tensor_set(text_pe, text_pe_data.data(), 0, text_pe_data.size() * sizeof(float));
            if (audio_pe) ggml_backend_tensor_set(audio_pe, audio_pe_data.data(), 0, audio_pe_data.size() * sizeof(float));
            if (mask) ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(float));
        } else {
            if (token_tensor) {
                int latest_token = current_audio_ids.back();
                ggml_backend_tensor_set(token_tensor, &latest_token, 0, sizeof(int32_t));
            }
            if (audio_pe) ggml_backend_tensor_set(audio_pe, audio_pe_data.data(), 0, audio_pe_data.size() * sizeof(float));
        }

        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[T2S Debug] Step " << total_decoded << ": computing graph..." << std::endl; std::fflush(stdout);
        ggml_backend_graph_compute(backend, cgraph);

        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[T2S Debug] Step " << total_decoded << ": fetching logits..." << std::endl; std::fflush(stdout);
        // Fetch logits back
        std::vector<float> logits(1025);
        ggml_backend_tensor_get(logits_tensor, logits.data(), 0, 1025 * sizeof(float));

        // Save debug logs if needed
        if (align_mode && total_decoded == 0) {
            save_tensor_binary("scratch/cpp_debug_text_emb.bin", t_emb);
            save_tensor_binary("scratch/cpp_debug_bert_proj_aligned.bin", bert_proj_aligned);
            save_tensor_binary("scratch/cpp_debug_text_fused.bin", text_fused);
            save_tensor_binary("scratch/cpp_debug_text_pe.bin", text_pe);
            save_tensor_binary("scratch/cpp_debug_text_rep.bin", text_rep);
            save_tensor_binary("scratch/cpp_debug_audio_emb.bin", a_emb);
            save_tensor_binary("scratch/cpp_debug_audio_pe.bin", audio_pe);
            save_tensor_binary("scratch/cpp_debug_audio_rep.bin", audio_rep);
            save_tensor_binary("scratch/cpp_debug_x_concat.bin", x_concat);
            save_tensor_binary("scratch/cpp_debug_layer0_Q.bin", layer0_Q);
            save_tensor_binary("scratch/cpp_debug_layer0_K.bin", layer0_K);
            save_tensor_binary("scratch/cpp_debug_layer0_V.bin", layer0_V);
            save_tensor_binary("scratch/cpp_debug_layer0_kq.bin", layer0_kq);
            save_tensor_binary("scratch/cpp_debug_layer0_kqv.bin", layer0_kqv);
            save_tensor_binary("scratch/cpp_debug_layer0_attn_out.bin", layer0_attn_out);
            save_tensor_binary("scratch/cpp_debug_layer0_x_attn.bin", layer0_x_attn);
            save_tensor_binary("scratch/cpp_debug_layer0_mlp_out.bin", layer0_mlp_out);
            save_tensor_binary("scratch/cpp_debug_layer0_out.bin", layer0_out);
        }

        if (align_mode && total_decoded < 5) {
            std::string out_path = "scratch/pipeline_alignment_cpp_logits_t" + std::to_string(total_decoded) + ".f32";
            std::ofstream file(out_path, std::ios::binary);
            if (file.is_open()) {
                file.write(reinterpret_cast<const char *>(logits.data()), (std::streamsize)(logits.size() * sizeof(float)));
            }
        }

        if (total_decoded < 11) {
            logits.resize(1024);
        }

        // Config sampling
        float temp = 0.6f;
        int top_k = 20;
        float top_p = 0.6f;
        float rep_penalty = 1.35f;
        const char* env_temp = std::getenv("T2S_TEMPERATURE");
        if (env_temp) temp = std::strtof(env_temp, nullptr);
        const char* env_top_k = std::getenv("T2S_TOP_K");
        if (env_top_k) top_k = std::strtol(env_top_k, nullptr, 10);
        const char* env_top_p = std::getenv("T2S_TOP_P");
        if (env_top_p) top_p = std::strtof(env_top_p, nullptr);
        const char* env_rep_penalty = std::getenv("T2S_REPETITION_PENALTY");
        if (env_rep_penalty) rep_penalty = std::strtof(env_rep_penalty, nullptr);

        bool is_greedy = align_mode || (temp <= 0.0f) || (top_k == 1);
        int32_t next_token = sample_logits(logits, current_audio_ids, temp, top_k, top_p, rep_penalty, is_greedy);

        if (next_token == 1024) break; // EOS

        generated_semantics.push_back(next_token);
        current_audio_ids.push_back(next_token);
        total_decoded++;
    }

    ggml_free(ctx_step);

    if (align_mode) {
        std::string tokens_path = "scratch/pipeline_alignment_cpp_tokens.txt";
        std::ofstream file(tokens_path);
        if (file.is_open()) {
            for (size_t i = 0; i < generated_semantics.size(); ++i) {
                file << generated_semantics[i] << (i == generated_semantics.size() - 1 ? "" : ",");
            }
        }
    }

    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[T2S] Autoregressively generated " << generated_semantics.size() << " semantic codes.\n";
    ggml_gallocr_free(galloc);
    return generated_semantics;
}

} // namespace gpt_sovits
