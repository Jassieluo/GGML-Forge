#include "gpt_t2s.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ops/ops.h"
#include "nn/nn.h"
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

    // Pre-convert FP16 weights to FP32 for non-CUDA (CPU/SYCL) backends
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
            /* .mem_size   = */ 16 * 1024 * 1024,
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
            if (!old_w) continue;

            if (old_w->type == GGML_TYPE_F16) {
                if (pair.first.find("embeddings") != std::string::npos) {
                    continue; // Skip embedding weights (keep FP16 embeddings)
                }

                int64_t w_elems = ggml_nelements(old_w);
                std::vector<uint8_t> w_bytes(ggml_nbytes(old_w));
                ggml_backend_tensor_get(old_w, w_bytes.data(), 0, w_bytes.size());

                std::vector<float> w_f32_data(w_elems);
                const ggml_fp16_t* ptr = (const ggml_fp16_t*)w_bytes.data();
                for (int64_t i = 0; i < w_elems; ++i) {
                    w_f32_data[i] = ggml_fp16_to_fp32(ptr[i]);
                }

                struct ggml_tensor* new_w = ggml_new_tensor(custom_ctx, GGML_TYPE_F32, ggml_n_dims(old_w), old_w->ne);
                ggml_set_name(new_w, old_w->name);

                fp32_tensors_list.push_back(new_w);
                fp32_upload_list.push_back({pair.first, w_f32_data});
            }
        }

        if (!fp32_tensors_list.empty()) {
            custom_buffer = ggml_backend_alloc_ctx_tensors(custom_ctx, backend);
            if (!custom_buffer) {
                std::cerr << "[T2S load] Error: Failed to allocate custom_buffer for weights!" << std::endl;
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
    }

    return true;
}

std::vector<int32_t> T2SModel::forward(
    const std::vector<int32_t>& text_ids,
    const std::vector<int32_t>& prompt_audio_ids,
    struct ggml_tensor* bert_features,
    float temp, int top_k, float top_p, float rep_penalty,
    ggml_backend_t backend
) {
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[T2S Debug] forward start. text_ids=" << text_ids.size() << ", prompt_audio_ids=" << prompt_audio_ids.size() << std::endl; std::fflush(stdout);
    
    int text_len = (int)text_ids.size();
    int prompt_len = (int)prompt_audio_ids.size();
    
    // Retrieve embedding tensors
    struct ggml_tensor* text_embed = get_tensor("bert_proj.weight");
    struct ggml_tensor* bert_proj_w = get_tensor("bert_proj.weight");
    struct ggml_tensor* bert_proj_b = get_tensor("bert_proj.bias");
    struct ggml_tensor* audio_embed = get_tensor("audio_embed.weight");
    struct ggml_tensor* predict_w = get_tensor("predict.weight");
    
    if (!text_embed || !bert_proj_w || !bert_proj_b || !audio_embed || !predict_w) {
        std::cerr << "[T2S] Error: Missing model weights in GGUF weight mapping!\n";
        return {};
    }

    // Wrap embedding layers with nn::Modules
    nn::Linear bert_proj(bert_proj_w, bert_proj_b);
    nn::Linear predict(predict_w, nullptr);
    
    // Setup persistent graph allocator (gallocr) for decoder steps
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!galloc) {
        std::cerr << "[T2S] Error: Failed to create graph allocator (gallocr)!\n";
        return {};
    }

    // Allocate KV Cache memory in backend buffers (head_dim=64, max_seq_len=1024, n_heads=16, layers=24, type=F32)
    int head_dim = 64;
    int max_seq_len = 1024;
    
    struct ggml_init_params cache_init_params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true
    };
    struct ggml_context* ctx_kv = ggml_init(cache_init_params);
    
    struct ggml_tensor* kv_k = ggml_new_tensor_4d(ctx_kv, GGML_TYPE_F32, head_dim, max_seq_len, n_heads, n_layers);
    struct ggml_tensor* kv_v = ggml_new_tensor_4d(ctx_kv, GGML_TYPE_F32, head_dim, max_seq_len, n_heads, n_layers);
    
    ggml_backend_buffer_t kv_buffer = ggml_backend_alloc_ctx_tensors(ctx_kv, backend);
    if (!kv_buffer) {
        std::cerr << "[T2S] Error: Failed to allocate KV cache GPU buffer!\n";
        ggml_free(ctx_kv);
        ggml_gallocr_free(galloc);
        return {};
    }
    
    // Fill KV cache with zeros initially
    std::vector<float> zero_kv(ggml_nelements(kv_k), 0.0f);
    ggml_backend_tensor_set(kv_k, zero_kv.data(), 0, zero_kv.size() * sizeof(float));
    ggml_backend_tensor_set(kv_v, zero_kv.data(), 0, zero_kv.size() * sizeof(float));

    std::vector<int32_t> current_audio_ids = prompt_audio_ids;
    std::vector<int32_t> generated_ids;
    
    float text_alpha = 1.0f;
    float audio_alpha = 1.0f;
    
    // Auto-regressive decoding loop
    for (int total_decoded = 0; total_decoded < 600; ++total_decoded) {
        int audio_len = (int)current_audio_ids.size();
        int total_len = text_len + audio_len;
        
        if (total_len >= max_seq_len) {
            if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[T2S Debug] Reached max sequence limit: " << total_len << std::endl;
            break;
        }

        struct ggml_init_params step_params = {
            /* .mem_size   = */ 256 * 1024 * 1024,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true
        };
        struct ggml_context* ctx_step = ggml_init(step_params);
        if (!ctx_step) {
            std::cerr << "[T2S] Error: Failed to initialize step context!\n";
            break;
        }

        struct ggml_cgraph* cgraph = ggml_new_graph_custom(ctx_step, 4096, false);
        
        struct ggml_tensor* text_ids_tensor = nullptr;
        struct ggml_tensor* bert_features_local = nullptr;
        struct ggml_tensor* audio_ids_tensor = nullptr;
        struct ggml_tensor* token_tensor = nullptr;
        struct ggml_tensor* text_pe = nullptr;
        struct ggml_tensor* audio_pe = nullptr;
        struct ggml_tensor* mask = nullptr;
        
        std::vector<float> text_pe_data;
        std::vector<float> audio_pe_data;
        std::vector<float> mask_data;

        struct ggml_tensor* x = nullptr;
        struct ggml_tensor* layer0_Q = nullptr;
        struct ggml_tensor* layer0_K = nullptr;
        struct ggml_tensor* layer0_V = nullptr;
        struct ggml_tensor* layer0_kq = nullptr;
        struct ggml_tensor* layer0_kqv = nullptr;
        struct ggml_tensor* layer0_attn_out = nullptr;
        struct ggml_tensor* layer0_x_attn = nullptr;
        struct ggml_tensor* layer0_mlp_out = nullptr;
        struct ggml_tensor* layer0_out = nullptr;

        auto mul_f32 = [&](struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b) -> struct ggml_tensor* {
            struct ggml_tensor* r = ggml_mul_mat(ctx, a, (b->type == GGML_TYPE_F32) ? b : ggml_cast(ctx, b, GGML_TYPE_F32));
            ggml_mul_mat_set_prec(r, GGML_PREC_DEFAULT);
            return r;
        };

        if (total_decoded == 0) {
            // First step: encode all prefix text tokens & prompt audio tokens
            text_ids_tensor = ggml_new_tensor_1d(ctx_step, GGML_TYPE_I32, text_len);
            
            // Text embeddings using direct get_rows
            struct ggml_tensor* t_emb = ggml_get_rows(ctx_step, text_embed, text_ids_tensor);
            bert_features_local = ggml_new_tensor_2d(ctx_step, GGML_TYPE_F32, 1024, text_len);
            struct ggml_tensor* bert_proj_aligned = bert_proj.forward(ctx_step, bert_features_local);
            
            struct ggml_tensor* text_fused = ggml_add(ctx_step, t_emb, bert_proj_aligned);
            text_pe = ggml_new_tensor_2d(ctx_step, GGML_TYPE_F32, 512, text_len);
            text_pe_data = compute_positional_embeddings(text_len, 512, text_alpha);
            text_rep = ggml_add(ctx_step, text_fused, text_pe);

            // Audio embeddings
            audio_ids_tensor = ggml_new_tensor_1d(ctx_step, GGML_TYPE_I32, audio_len);
            struct ggml_tensor* a_emb = ggml_get_rows(ctx_step, audio_embed, audio_ids_tensor);
            audio_pe = ggml_new_tensor_2d(ctx_step, GGML_TYPE_F32, 512, audio_len);
            audio_pe_data = compute_positional_embeddings(audio_len, 512, audio_alpha);
            struct ggml_tensor* audio_rep = ggml_add(ctx_step, a_emb, audio_pe);

            struct ggml_tensor* x_concat = ggml_concat(ctx_step, text_rep, audio_rep, 1);
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

            // Wrap Linear & LayerNorm & FeedForward layers on the stack
            nn::Linear q_proj(qw, qb);
            nn::Linear k_proj(kw, kb);
            nn::Linear v_proj(vw, vb);
            nn::Linear out_proj(out_w, out_b);
            nn::LayerNorm ln1(ln1_w, ln1_b, 1e-5f);
            nn::LayerNorm ln2(ln2_w, ln2_b, 1e-5f);
            nn::FeedForward ffn(ffn_w1, ffn_b1, ffn_w2, ffn_b2, nn::ActivationType::DOUBLE_SWISH);

            struct ggml_tensor* Q = q_proj.forward(ctx_step, x);
            struct ggml_tensor* K = k_proj.forward(ctx_step, x);
            struct ggml_tensor* V = v_proj.forward(ctx_step, x);

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
            struct ggml_tensor* kq_soft = ggml_soft_max(ctx_step, kq_masked);

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
            struct ggml_tensor* attn_out = out_proj.forward(ctx_step, kqv);

            // Residual + LN1
            struct ggml_tensor* x_attn = ggml_add(ctx_step, x, attn_out);
            x_attn = ln1.forward(ctx_step, x_attn, backend);

            // MLP using FeedForward module
            struct ggml_tensor* mlp_out = ffn.forward(ctx_step, x_attn, backend);

            if (layer == 0) {
                layer0_attn_out = attn_out;
                layer0_x_attn = x_attn;
                layer0_mlp_out = mlp_out;
            }

            // Residual + LN2
            x = ggml_add(ctx_step, x_attn, mlp_out);
            x = ln2.forward(ctx_step, x, backend);

            if (layer == 0) {
                layer0_out = x;
            }
        }

        // Predict logits using Linear module
        struct ggml_tensor* last_token_rep = ggml_view_2d(ctx_step, x, hidden_dim, 1, x->nb[1], (q_len - 1) * x->nb[1]);
        struct ggml_tensor* logits_tensor = predict.forward(ctx_step, last_token_rep);
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
                int32_t last_token = current_audio_ids.back();
                ggml_backend_tensor_set(token_tensor, &last_token, 0, sizeof(int32_t));
            }
            if (audio_pe) ggml_backend_tensor_set(audio_pe, audio_pe_data.data(), 0, audio_pe_data.size() * sizeof(float));
        }

        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[T2S Debug] Step " << total_decoded << ": computing..." << std::endl; std::fflush(stdout);
        ggml_backend_graph_compute(backend, cgraph);

        // Get logits back to CPU
        std::vector<float> host_logits(1025);
        ggml_backend_tensor_get(logits_tensor, host_logits.data(), 0, 1025 * sizeof(float));

        // Sample next token
        int32_t next_token = sample_logits(host_logits, current_audio_ids, temp, top_k, top_p, rep_penalty, (temp <= 0.0f));
        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[T2S Debug] Step " << total_decoded << ": sampled token " << next_token << std::endl; std::fflush(stdout);

        // Print intermediate debug info if requested
        if (GPT_SOVITS_DEBUG_ENABLED() && total_decoded == 0) {
            auto print_step_debug = [&](const std::string& name, struct ggml_tensor* t) {
                if (!t) return;
                int64_t total_elements = ggml_nelements(t);
                std::vector<float> data(total_elements);
                ggml_backend_tensor_get(t, data.data(), 0, total_elements * sizeof(float));
                float min_val = 1e30f;
                float max_val = -1e30f;
                for (float v : data) {
                    if (v < min_val) min_val = v;
                    if (v > max_val) max_val = v;
                }
                std::cout << "  C++ Step0 " << name << " - Shape: [" << t->ne[0] << ", " << t->ne[1] << ", " << t->ne[2] << "]\n";
                std::cout << "      Min/Max: " << min_val << " / " << max_val << "\n";
            };
            std::cout << "\n[T2S C++ Step 0 Intermediate Debug]:\n";
            print_step_debug("layer0_Q", layer0_Q);
            print_step_debug("layer0_K", layer0_K);
            print_step_debug("layer0_V", layer0_V);
            print_step_debug("layer0_kq", layer0_kq);
            print_step_debug("layer0_kqv", layer0_kqv);
            print_step_debug("layer0_attn_out", layer0_attn_out);
            print_step_debug("layer0_x_attn", layer0_x_attn);
            print_step_debug("layer0_mlp_out", layer0_mlp_out);
            print_step_debug("layer0_out", layer0_out);
            print_step_debug("logits", logits_tensor);
            std::cout << std::endl;

            dump_tensor_f32_if_requested("GPT_SOVITS_T2S_DUMP", "layer0_Q", layer0_Q);
            dump_tensor_f32_if_requested("GPT_SOVITS_T2S_DUMP", "layer0_K", layer0_K);
            dump_tensor_f32_if_requested("GPT_SOVITS_T2S_DUMP", "layer0_V", layer0_V);
            dump_tensor_f32_if_requested("GPT_SOVITS_T2S_DUMP", "layer0_kq", layer0_kq);
            dump_tensor_f32_if_requested("GPT_SOVITS_T2S_DUMP", "layer0_kqv", layer0_kqv);
            dump_tensor_f32_if_requested("GPT_SOVITS_T2S_DUMP", "layer0_attn_out", layer0_attn_out);
            dump_tensor_f32_if_requested("GPT_SOVITS_T2S_DUMP", "layer0_x_attn", layer0_x_attn);
            dump_tensor_f32_if_requested("GPT_SOVITS_T2S_DUMP", "layer0_mlp_out", layer0_mlp_out);
            dump_tensor_f32_if_requested("GPT_SOVITS_T2S_DUMP", "layer0_out", layer0_out);
            dump_tensor_f32_if_requested("GPT_SOVITS_T2S_DUMP", "logits", logits_tensor);
        }

        ggml_free(ctx_step);

        if (next_token == 1024) { // EOS token
            break;
        }
        current_audio_ids.push_back(next_token);
        generated_ids.push_back(next_token);
    }
    
    ggml_backend_buffer_free(kv_buffer);
    ggml_free(ctx_kv);
    ggml_gallocr_free(galloc);
    
    return generated_ids;
}

} // namespace gpt_sovits
