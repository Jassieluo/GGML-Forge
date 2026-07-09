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
#include <chrono>

namespace gpt_sovits {
namespace {

static std::vector<float> compute_positional_embeddings(int seq_len, int hidden_dim, float alpha, int pos_offset = 0) {
    std::vector<float> data(seq_len * hidden_dim, 0.0f);
    static int cached_dim = 0;
    static std::vector<float> inv_denoms;
    if (cached_dim != hidden_dim) {
        cached_dim = hidden_dim;
        inv_denoms.resize(hidden_dim / 2);
        for (int i = 0; i < hidden_dim / 2; ++i) {
            inv_denoms[i] = 1.0f / std::pow(10000.0f, (2.0f * i) / (float)hidden_dim);
        }
    }

    for (int i_pos = 0; i_pos < seq_len; ++i_pos) {
        int pos = i_pos + pos_offset;
        for (int i = 0; i < hidden_dim / 2; ++i) {
            float val = (float)pos * inv_denoms[i];
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

// Low-level sampler for top-k/top-p logits sampling
static int32_t sample_logits(
    std::vector<float>& logits, 
    const std::vector<int32_t>& current_audio_ids,
    float temp, int top_k, float top_p, float rep_penalty, bool is_greedy
) {
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

    if (is_greedy) {
        return (int32_t)(std::max_element(logits.begin(), logits.end()) - logits.begin());
    }

    std::vector<std::pair<float, int>> indexed_logits(logits.size());
    for (int i = 0; i < (int)logits.size(); ++i) {
        indexed_logits[i] = {logits[i], i};
    }
    std::sort(indexed_logits.begin(), indexed_logits.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

    if (top_k > 0 && top_k < (int)logits.size()) {
        indexed_logits.resize(top_k);
    }

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

    int kid_map = gguf_find_key(ctx_gguf, "gpt_sovits.t2s.name_map");
    if (kid_map >= 0) {
        std::string json_str = gguf_get_val_str(ctx_gguf, kid_map);
        std::unordered_map<std::string, std::string> loaded_map = nn::parse_flat_json(json_str);
        for (const auto& kv : loaded_map) {
            default_name_map[kv.first] = kv.second;
        }
    }
}

T2SModel::T2SModel() {
    register_module("word_embeddings", &word_embeddings);
    register_module("audio_embeddings", &audio_embeddings);
    register_module("bert_proj", &bert_proj);
    register_module("predict", &predict);
    
    layers.resize(24);
    for (int i = 0; i < 24; ++i) {
        register_module("layers." + std::to_string(i), &layers[i]);
        layers[i].self_attn.layer_idx = i;
        layers[i].ffn.act_type = nn::ActivationType::RELU;
    }
    
    init_default_name_map();
}

void T2SModel::init_default_name_map() {
    default_name_map["word_embeddings.weight"] = "ar_text_embedding.word_embeddings.weight";
    default_name_map["audio_embeddings.weight"] = "ar_audio_embedding.word_embeddings.weight";
    default_name_map["bert_proj.weight"] = "bert_proj.weight";
    default_name_map["bert_proj.bias"] = "bert_proj.bias";
    default_name_map["predict.weight"] = "ar_predict_layer.weight";
    
    for (int i = 0; i < 24; ++i) {
        std::string cpp = "layers." + std::to_string(i) + ".";
        std::string gguf = "h.layers." + std::to_string(i) + ".";
        
        default_name_map[cpp + "self_attn.q_proj.weight"] = gguf + "self_attn.q.weight";
        default_name_map[cpp + "self_attn.q_proj.bias"]   = gguf + "self_attn.q.bias";
        default_name_map[cpp + "self_attn.k_proj.weight"] = gguf + "self_attn.k.weight";
        default_name_map[cpp + "self_attn.k_proj.bias"]   = gguf + "self_attn.k.bias";
        default_name_map[cpp + "self_attn.v_proj.weight"] = gguf + "self_attn.v.weight";
        default_name_map[cpp + "self_attn.v_proj.bias"]   = gguf + "self_attn.v.bias";
        default_name_map[cpp + "self_attn.out_proj.weight"] = gguf + "self_attn.out_proj.weight";
        default_name_map[cpp + "self_attn.out_proj.bias"]   = gguf + "self_attn.out_proj.bias";
        
        default_name_map[cpp + "ln1.weight"] = gguf + "norm1.weight";
        default_name_map[cpp + "ln1.bias"]   = gguf + "norm1.bias";
        default_name_map[cpp + "ln2.weight"] = gguf + "norm2.weight";
        default_name_map[cpp + "ln2.bias"]   = gguf + "norm2.bias";
        
        default_name_map[cpp + "ffn.w1.weight"] = gguf + "linear1.weight";
        default_name_map[cpp + "ffn.w1.bias"]   = gguf + "linear1.bias";
        default_name_map[cpp + "ffn.w2.weight"] = gguf + "linear2.weight";
        default_name_map[cpp + "ffn.w2.bias"]   = gguf + "linear2.bias";
    }
}

bool T2SModel::load(const std::string& path, ggml_backend_t backend) {
    if (!load_gguf_model(path, *this, backend)) {
        return false;
    }

    // 🌟 One-click recursive bind using default name map!
    nn::bind(*this, *this, default_name_map);
    this->to(backend);

    // Dynamically retrieve head dim
    struct ggml_tensor* qw = layers[0].self_attn.q_proj.weight;
    if (qw) {
        int hidden_dim = (int)qw->ne[0];
        head_dim = hidden_dim / n_heads;
    }

    // Configure head counting parameters for all attention submodules
    for (int i = 0; i < 24; ++i) {
        layers[i].self_attn.n_heads = n_heads;
        layers[i].self_attn.head_dim = head_dim;
    }

    // Allocate GPU resident Keys and Values KV Cache (Native [head_dim, 512, n_heads, 24] shapes)
    // and pre-allocated static input placeholders.
    struct ggml_init_params kv_params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true
    };
    kv_ctx = ggml_init(kv_params);
    kv_k = ggml_new_tensor_4d(kv_ctx, GGML_TYPE_F32, head_dim, 512, n_heads, 24);
    kv_v = ggml_new_tensor_4d(kv_ctx, GGML_TYPE_F32, head_dim, 512, n_heads, 24);

    text_ids.tensor = ggml_new_tensor_1d(kv_ctx, GGML_TYPE_I32, 512);
    ggml_set_name(text_ids.tensor, "input_text_ids");
    audio_ids.tensor = ggml_new_tensor_1d(kv_ctx, GGML_TYPE_I32, 512);
    ggml_set_name(audio_ids.tensor, "input_audio_ids");
    token.tensor = ggml_new_tensor_1d(kv_ctx, GGML_TYPE_I32, 1);
    ggml_set_name(token.tensor, "input_token");
    bert_features.tensor = ggml_new_tensor_2d(kv_ctx, GGML_TYPE_F32, 1024, 512);
    ggml_set_name(bert_features.tensor, "input_bert_features");

    kv_buffer = ggml_backend_alloc_ctx_tensors(kv_ctx, backend);
    if (kv_buffer) {
        size_t total_elements = (size_t)head_dim * 512 * n_heads * 24;
        std::vector<float> zero_buf(total_elements, 0.0f);
        ggml_backend_tensor_set(kv_k, zero_buf.data(), 0, total_elements * sizeof(float));
        ggml_backend_tensor_set(kv_v, zero_buf.data(), 0, total_elements * sizeof(float));
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
    ggml_backend_t backend,
    ggml_gallocr_t galloc_in
) {
    bool align_mode = (std::getenv("T2S_ALIGNMENT") != nullptr);

    // Reset KV cache to zero to prevent residues from previous segment synthesis
    if (kv_k && kv_v) {
        size_t total_elements = (size_t)head_dim * 512 * n_heads * 24;
        std::vector<float> zero_buf(total_elements, 0.0f);
        ggml_backend_tensor_set(kv_k, zero_buf.data(), 0, total_elements * sizeof(float));
        ggml_backend_tensor_set(kv_v, zero_buf.data(), 0, total_elements * sizeof(float));
    }

    // Setup models weights
    struct ggml_tensor* ar_text_position_alpha = get_tensor("ar_text_position.alpha");
    struct ggml_tensor* ar_audio_position_alpha = get_tensor("ar_audio_position.alpha");
    
    if (!word_embeddings.weight || !audio_embeddings.weight || !bert_proj.weight || !bert_proj.bias || !ar_text_position_alpha || !ar_audio_position_alpha || !predict.weight) {
        std::cerr << "[T2S] Error: Missing model weights in GGUF weight mapping!\n";
        return {};
    }

    float text_alpha = 1.0f;
    float audio_alpha = 1.0f;
    ggml_backend_tensor_get(ar_text_position_alpha, &text_alpha, 0, sizeof(float));
    ggml_backend_tensor_get(ar_audio_position_alpha, &audio_alpha, 0, sizeof(float));



    int text_len = (int)(prompt_phones.size() + target_phones.size());
    std::vector<int32_t> text_ids_vec;
    text_ids_vec.reserve(text_len);
    text_ids_vec.insert(text_ids_vec.end(), prompt_phones.begin(), prompt_phones.end());
    text_ids_vec.insert(text_ids_vec.end(), target_phones.begin(), target_phones.end());

    std::vector<int32_t> current_audio_ids = prompt_semantics;
    std::vector<int32_t> generated_semantics;
    int total_decoded = 0;

    struct SchedGuard {
        ggml_backend_t blas_backend = nullptr;
        ggml_backend_sched_t sched = nullptr;
        ~SchedGuard() {
            if (sched) ggml_backend_sched_free(sched);
            if (blas_backend) ggml_backend_free(blas_backend);
        }
    } guard;

    const char* bname = ggml_backend_name(backend);
    if (bname && std::strcmp(bname, "CPU") == 0) {
        guard.blas_backend = ggml_backend_init_by_name("BLAS", nullptr);
        if (guard.blas_backend) {
            ggml_backend_t sched_backends[2] = { guard.blas_backend, backend };
            guard.sched = ggml_backend_sched_new(sched_backends, nullptr, 2, 4096, false, false);

        }
    }

    // Context size 4MB, allocated once and reset per step to stabilize graph keys
    struct ggml_init_params init_params = { 4 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_step = ggml_init(init_params);

    bool is_local_galloc = false;
    ggml_gallocr_t galloc = galloc_in;
    if (!galloc && !guard.sched) {
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        is_local_galloc = true;
        if (!galloc) {
            std::cerr << "[T2S] Error: Failed to create graph allocator (gallocr)!\n";
            ggml_free(ctx_step);
            return {};
        }
    }


    std::vector<float> temp_bert;

    double sum_build = 0;
    double sum_alloc = 0;
    double sum_upload = 0;
    double sum_compute = 0;
    double sum_sample = 0;
    int num_steps = 0;

    while (total_decoded < max_len) {
        int64_t t_start = ggml_time_us();
        int audio_len = (int)current_audio_ids.size();
        int total_len = text_len + audio_len;
        if (total_len >= 512) break;



        ggml_reset(ctx_step);
        struct ggml_cgraph* cgraph = ggml_new_graph(ctx_step);


        struct ggml_tensor* x = nullptr;
        struct ggml_tensor* bert_features_local = nullptr;
        struct ggml_tensor* text_ids_tensor_view = nullptr;
        struct ggml_tensor* audio_ids_tensor_view = nullptr;
        struct ggml_tensor* token_tensor_view = nullptr;
        struct ggml_tensor* text_pe = nullptr;
        struct ggml_tensor* audio_pe = nullptr;
        struct ggml_tensor* mask = nullptr;

        std::vector<float> text_pe_data;
        std::vector<float> audio_pe_data;
        std::vector<float> mask_data;

        struct ggml_tensor* t_emb = nullptr;
        struct ggml_tensor* text_fused = nullptr;
        struct ggml_tensor* text_rep = nullptr;
        struct ggml_tensor* audio_rep = nullptr;



        std::vector<nn::Input> step_inputs;
        int32_t last_token = 0;

        if (total_decoded == 0) {
            temp_bert.resize(1024 * text_len);
            if (bert_features->buffer) {
                ggml_backend_tensor_get(bert_features, temp_bert.data(), 0, temp_bert.size() * sizeof(float));
            } else if (bert_features->data) {
                std::memcpy(temp_bert.data(), bert_features->data, temp_bert.size() * sizeof(float));
            }

            if (GPT_SOVITS_DEBUG_ENABLED()) {
                std::cout << "[T2S Debug] bert_features first 10 values: ";
                for (int i = 0; i < 10; ++i) {
                    std::cout << temp_bert[i] << " ";
                }
                std::cout << "\n";
                std::fflush(stdout);
            }

            // First step: Process prompt phones and prompt semantics entirely
            nn::Input bert_in;
            bert_in.tensor = this->bert_features.tensor;
            bert_in.data_ptr = temp_bert.data();
            bert_in.size_bytes = temp_bert.size() * sizeof(float);
            step_inputs.push_back(bert_in);

            bert_features_local = ggml_view_2d(ctx_step, this->bert_features.tensor, 1024, text_len, this->bert_features.tensor->nb[1], 0);

            struct ggml_tensor* bert_proj_aligned = bert_proj(ctx_step, bert_features_local);

            // Text embeddings
            nn::Input text_in;
            text_in.tensor = this->text_ids.tensor;
            text_in.data_ptr = text_ids_vec.data();
            text_in.size_bytes = text_len * sizeof(int32_t);
            step_inputs.push_back(text_in);

            text_ids_tensor_view = ggml_view_1d(ctx_step, this->text_ids.tensor, text_len, 0);

            t_emb = word_embeddings(ctx_step, text_ids_tensor_view);
            text_fused = ggml_add(ctx_step, t_emb, bert_proj_aligned);

            text_pe_data = compute_positional_embeddings(text_len, 512, text_alpha);
            auto text_pe_in = nn::Input::tensor_2d(ctx_step, GGML_TYPE_F32, 512, text_len, text_pe_data.data(), text_pe_data.size() * sizeof(float));
            text_pe = text_pe_in.tensor;
            step_inputs.push_back(text_pe_in);

            text_rep = ggml_add(ctx_step, text_fused, text_pe);

            // Audio embeddings
            nn::Input audio_in;
            audio_in.tensor = this->audio_ids.tensor;
            audio_in.data_ptr = current_audio_ids.data();
            audio_in.size_bytes = audio_len * sizeof(int32_t);
            step_inputs.push_back(audio_in);

            audio_ids_tensor_view = ggml_view_1d(ctx_step, this->audio_ids.tensor, audio_len, 0);

            struct ggml_tensor* a_emb = audio_embeddings(ctx_step, audio_ids_tensor_view);

            audio_pe_data = compute_positional_embeddings(audio_len, 512, audio_alpha);
            auto audio_pe_in = nn::Input::tensor_2d(ctx_step, GGML_TYPE_F32, 512, audio_len, audio_pe_data.data(), audio_pe_data.size() * sizeof(float));
            audio_pe = audio_pe_in.tensor;
            step_inputs.push_back(audio_pe_in);

            audio_rep = ggml_add(ctx_step, a_emb, audio_pe);

            x = ggml_concat(ctx_step, text_rep, audio_rep, 1);


        } else {
            // Self-regressive: Feed only the latest token
            last_token = current_audio_ids.back();
            nn::Input token_in;
            token_in.tensor = this->token.tensor;
            token_in.data_ptr = &last_token;
            token_in.size_bytes = sizeof(int32_t);
            step_inputs.push_back(token_in);

            token_tensor_view = ggml_view_1d(ctx_step, this->token.tensor, 1, 0);

            x = audio_embeddings(ctx_step, token_tensor_view);

            int pos_idx = audio_len - 1;
            audio_pe_data = compute_positional_embeddings(1, 512, audio_alpha, pos_idx);
            auto audio_pe_in = nn::Input::tensor_1d(ctx_step, GGML_TYPE_F32, 512, audio_pe_data.data(), audio_pe_data.size() * sizeof(float));
            audio_pe = audio_pe_in.tensor;
            step_inputs.push_back(audio_pe_in);

            x = ggml_add(ctx_step, x, audio_pe);
        }

        // Generate mask
        mask = nullptr;
        if (total_decoded == 0) {
            mask_data = compute_prefix_causal_mask(total_len, text_len, n_heads);
            auto mask_in = nn::Input::tensor_3d(ctx_step, GGML_TYPE_F32, total_len, total_len, n_heads, mask_data.data(), mask_data.size() * sizeof(float));
            mask = mask_in.tensor;
            step_inputs.push_back(mask_in);
        } else {
            int max_cache_len = (int)this->kv_k->ne[1];
            mask_data.assign(max_cache_len, 0.0f);
            for (int i = total_len; i < max_cache_len; ++i) {
                mask_data[i] = -1e4f;
            }
            auto mask_in = nn::Input::tensor_1d(ctx_step, GGML_TYPE_F32, max_cache_len, mask_data.data(), max_cache_len * sizeof(float));
            mask = mask_in.tensor;
            step_inputs.push_back(mask_in);
        }

        int q_len = (total_decoded == 0) ? total_len : 1;
        const int hidden_dim = n_heads * head_dim;



        // Execute attention layers
        for (int layer = 0; layer < 24; ++layer) {
            auto& blk = layers[layer];

            struct ggml_tensor* attn_out = blk.self_attn(ctx_step, x, kv_k, kv_v, q_len, total_len, mask, cgraph);

            // Residual + LN1
            struct ggml_tensor* x_attn = ggml_add(ctx_step, x, attn_out);
            x_attn = blk.ln1(ctx_step, x_attn);

            // MLP using FeedForward module
            struct ggml_tensor* mlp_out = blk.ffn(ctx_step, x_attn);

            // Residual + LN2
            x = ggml_add(ctx_step, x_attn, mlp_out);
            x = blk.ln2(ctx_step, x);
        }

        // No final LayerNorm in GPT-SoVITS T2S model, proceed directly to prediction

        // Predict logits
        struct ggml_tensor* last_token_rep = ggml_view_2d(ctx_step, x, hidden_dim, 1, x->nb[1], (q_len - 1) * x->nb[1]);
        struct ggml_tensor* logits_tensor = predict(ctx_step, last_token_rep);
        ggml_build_forward_expand(cgraph, logits_tensor);

        int64_t t_after_build = ggml_time_us();
        sum_build += (t_after_build - t_start) / 1000.0;



        // Allocate step buffers using persistent galloc or sched
        if (guard.sched) {
            ggml_backend_sched_reset(guard.sched);
            if (!ggml_backend_sched_alloc_graph(guard.sched, cgraph)) {
                std::cerr << "[T2S] Error: Failed to allocate graph using sched!\n";
                ggml_free(ctx_step);
                if (galloc) ggml_gallocr_free(galloc);
                return {};
            }
        } else {
            if (!ggml_gallocr_alloc_graph(galloc, cgraph)) {
                std::cerr << "[T2S] Error: Failed to allocate graph using gallocr!\n";
                ggml_free(ctx_step);
                if (galloc) ggml_gallocr_free(galloc);
                return {};
            }
        }

        int64_t t_after_alloc = ggml_time_us();
        sum_alloc += (t_after_alloc - t_after_build) / 1000.0;



        // Upload input data using PyTorch-style Input wrappers
        for (const auto& input : step_inputs) {
            input.upload();
        }

        int64_t t_after_upload = ggml_time_us();
        sum_upload += (t_after_upload - t_after_alloc) / 1000.0;





        if (guard.sched) {
            ggml_backend_sched_graph_compute(guard.sched, cgraph);
        } else {
            ggml_backend_graph_compute(backend, cgraph);
        }

        int64_t t_after_compute = ggml_time_us();
        sum_compute += (t_after_compute - t_after_upload) / 1000.0;

        // Get logits back to CPU
        std::vector<float> host_logits(1025);
        ggml_backend_tensor_get(logits_tensor, host_logits.data(), 0, 1025 * sizeof(float));

        if (total_decoded == 0 && GPT_SOVITS_DEBUG_ENABLED()) {
            std::cout << "[T2S Debug] Step 0 logits first 20 values: ";
            for (int i = 0; i < 20; ++i) {
                std::cout << host_logits[i] << " ";
            }
            std::cout << "\n";
            
            // Find and print top 10 logits
            std::vector<std::pair<float, int>> sorted_logits;
            for (int i = 0; i < 1025; ++i) {
                sorted_logits.push_back({host_logits[i], i});
            }
            std::sort(sorted_logits.rbegin(), sorted_logits.rend());
            std::cout << "[T2S Debug] Step 0 Top 10 Logits:\n";
            for (int i = 0; i < 10; ++i) {
                std::cout << "  Rank " << i << ": index=" << sorted_logits[i].second 
                          << ", value=" << sorted_logits[i].first << "\n";
            }
            std::fflush(stdout);
        }

        if (total_decoded < 11) {
            host_logits.resize(1024);
        }

        // Config sampling
        float temp = 0.6f;
        int top_k = 20;
        float top_p = 0.6f;
        float rep_penalty = 1.35f;
        const char* env_temp = std::getenv("T2S_TEMPERATURE");
        if (env_temp) temp = std::strtof(env_temp, nullptr);
        const char* env_top_k = std::getenv("T2S_TOP_K");
        if (env_top_k) top_k = (int)std::strtol(env_top_k, nullptr, 10);
        const char* env_top_p = std::getenv("T2S_TOP_P");
        if (env_top_p) top_p = std::strtof(env_top_p, nullptr);
        const char* env_rep_penalty = std::getenv("T2S_REPETITION_PENALTY");
        if (env_rep_penalty) rep_penalty = std::strtof(env_rep_penalty, nullptr);

        bool is_greedy = align_mode || (temp <= 0.0f) || (top_k == 1);
        int32_t next_token = sample_logits(host_logits, current_audio_ids, temp, top_k, top_p, rep_penalty, is_greedy);

        if (next_token == 1024) { // EOS token
            break;
        }

        if (align_mode) {
            // Under alignment cross-testing, override sampling with reference tokens
            int target_idx = total_decoded;
            if (target_idx < (int)target_word2ph.size()) {
                // Here we dummy code or alignment tests override
            }
        }

        current_audio_ids.push_back(next_token);
        generated_semantics.push_back(next_token);
        total_decoded++;

        int64_t t_after_sample = ggml_time_us();
        sum_sample += (t_after_sample - t_after_compute) / 1000.0;
        num_steps++;
    }

    if (GPT_SOVITS_DEBUG_ENABLED() && num_steps > 0) {
        std::cout << "[T2S Loop Profile] Total steps: " << num_steps << "\n"
                  << "  Average Graph Build:   " << sum_build / num_steps << " ms/step\n"
                  << "  Average Graph Alloc:   " << sum_alloc / num_steps << " ms/step\n"
                  << "  Average Input Upload:  " << sum_upload / num_steps << " ms/step\n"
                  << "  Average GPU Compute:   " << sum_compute / num_steps << " ms/step\n"
                  << "  Average Sample/Fetch:  " << sum_sample / num_steps << " ms/step\n"
                  << "  Average Total Step:    " << (sum_build + sum_alloc + sum_upload + sum_compute + sum_sample) / num_steps << " ms/step\n";
    }

    if (galloc && is_local_galloc) {
        ggml_gallocr_free(galloc);
    }
    ggml_free(ctx_step);

    return generated_semantics;
}

} // namespace gpt_sovits
