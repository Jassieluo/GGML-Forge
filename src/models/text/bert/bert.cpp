#include "bert.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ops/ops.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cctype>

namespace gpt_sovits {

#pragma pack(push, 1)
struct ggml_bert_block_q4_0 {
    ggml_fp16_t d;       // delta
    uint8_t qs[16];      // 32 nibbles
};
#pragma pack(pop)

static void dequantize_q4_0_to_fp16(const uint8_t * src, ggml_fp16_t * dst, int64_t nelements) {
    int64_t qk = 32;
    int64_t nb = nelements / qk;
    const ggml_bert_block_q4_0 * blocks = (const ggml_bert_block_q4_0 *)src;
    for (int64_t i = 0; i < nb; ++i) {
        float d = ggml_fp16_to_fp32(blocks[i].d);
        for (int j = 0; j < qk / 2; ++j) {
            int x0 = (blocks[i].qs[j] & 0x0F) - 8;
            int x1 = (blocks[i].qs[j] >>   4) - 8;
            dst[i * qk + j + 0]  = ggml_fp32_to_fp16(x0 * d);
            dst[i * qk + j + 16] = ggml_fp32_to_fp16(x1 * d);
        }
    }
}

static void dequantize_q4_0_to_fp32(const uint8_t * src, float * dst, int64_t nelements) {
    int64_t qk = 32;
    int64_t nb = nelements / qk;
    const ggml_bert_block_q4_0 * blocks = (const ggml_bert_block_q4_0 *)src;
    for (int64_t i = 0; i < nb; ++i) {
        float d = ggml_fp16_to_fp32(blocks[i].d);
        for (int j = 0; j < qk / 2; ++j) {
            int x0 = (blocks[i].qs[j] & 0x0F) - 8;
            int x1 = (blocks[i].qs[j] >>   4) - 8;
            dst[i * qk + j + 0]  = x0 * d;
            dst[i * qk + j + 16] = x1 * d;
        }
    }
}

void BertModel::on_read_metadata(struct gguf_context* ctx_gguf) {
    int kid = gguf_find_key(ctx_gguf, "attention.head_count");
    if (kid >= 0) {
        n_heads = (int)gguf_get_val_u32(ctx_gguf, kid);
    } else {
        bool is_bert_large = false;
        int name_id = gguf_find_key(ctx_gguf, "general.name");
        if (name_id >= 0) {
            std::string gname = gguf_get_val_str(ctx_gguf, name_id);
            std::transform(gname.begin(), gname.end(), gname.begin(), [](unsigned char c){ return std::tolower(c); });
            if (gname.find("roberta") != std::string::npos || gname.find("large") != std::string::npos) {
                is_bert_large = true;
            }
        }
        n_heads = is_bert_large ? 16 : 8;
        if (GPT_SOVITS_DEBUG_ENABLED()) {
            std::cout << "[BERT] attention.head_count missing from GGUF. Detected general.name: " 
                      << (name_id >= 0 ? gguf_get_val_str(ctx_gguf, name_id) : "unknown") 
                      << " -> Configured n_heads = " << n_heads << std::endl;
        }
    }
}

bool BertModel::load(const std::string& path, ggml_backend_t backend) {
    if (!load_gguf_model(path, *this, backend)) {
        return false;
    }

    // Pre-convert FP16 weights to FP32 for non-CUDA (CPU/SYCL) backends to eliminate dynamic casts
    bool is_cuda = false;
    if (backend) {
        const char * bname = ggml_backend_name(backend);
        if (bname && strncmp(bname, "CUDA", 4) == 0) {
            is_cuda = true;
        }
    }

    if (!is_cuda) {
        bool is_sycl = false;
        if (backend) {
            const char * bname = ggml_backend_name(backend);
            if (bname && strncmp(bname, "SYCL", 4) == 0) {
                is_sycl = true;
            }
        }
        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[BERT load] Non-CUDA backend detected. Performing weights pre-processing..." << std::endl;
        struct ggml_init_params custom_params = {
            /* .mem_size   = */ 32 * 1024 * 1024, // 32MB metadata pool
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true
        };
        custom_ctx = ggml_init(custom_params);
        if (!custom_ctx) {
            std::cerr << "[BERT load] Error: Failed to initialize custom_ctx for FP32 weights!" << std::endl;
            return false;
        }

        struct UploadF32Entry {
            std::string name;
            std::vector<float> data;
        };
        struct UploadF16Entry {
            std::string name;
            std::vector<ggml_fp16_t> data;
        };
        std::vector<UploadF32Entry> fp32_upload_list;
        std::vector<struct ggml_tensor*> fp32_tensors_list;
        std::vector<UploadF16Entry> fp16_upload_list;
        std::vector<struct ggml_tensor*> fp16_tensors_list;

        for (const auto& pair : tensors) {
            struct ggml_tensor* old_w = pair.second;
            if (!old_w) continue;

            // 1. Convert non-embedding FP16 weights to FP32 (for CPU/SYCL optimization)
            if (old_w->type == GGML_TYPE_F16) {
                if (pair.first.find("embeddings") != std::string::npos) {
                    continue; // Skip embedding tensors (keep FP16 embeddings as FP16)
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
            // 2. Pre-dequantize Q4_0 embedding weights to FP16 (for SYCL get_rows support) or non-embedding weights to FP32 (for SYCL driver workaround)
            else if (old_w->type == GGML_TYPE_Q4_0) {
                if (pair.first.find("embeddings") != std::string::npos) {
                    int64_t w_elems = ggml_nelements(old_w);
                    std::vector<uint8_t> w_bytes(ggml_nbytes(old_w));
                    ggml_backend_tensor_get(old_w, w_bytes.data(), 0, w_bytes.size());

                    std::vector<ggml_fp16_t> w_fp16_data(w_elems);
                    dequantize_q4_0_to_fp16(w_bytes.data(), w_fp16_data.data(), w_elems);

                    struct ggml_tensor* new_w = ggml_new_tensor(custom_ctx, GGML_TYPE_F16, ggml_n_dims(old_w), old_w->ne);
                    ggml_set_name(new_w, old_w->name);

                    fp16_tensors_list.push_back(new_w);
                    fp16_upload_list.push_back({pair.first, w_fp16_data});
                } else if (is_sycl) {
                    int64_t w_elems = ggml_nelements(old_w);
                    std::vector<uint8_t> w_bytes(ggml_nbytes(old_w));
                    ggml_backend_tensor_get(old_w, w_bytes.data(), 0, w_bytes.size());

                    std::vector<float> w_f32_data(w_elems);
                    dequantize_q4_0_to_fp32(w_bytes.data(), w_f32_data.data(), w_elems);

                    struct ggml_tensor* new_w = ggml_new_tensor(custom_ctx, GGML_TYPE_F32, ggml_n_dims(old_w), old_w->ne);
                    ggml_set_name(new_w, old_w->name);

                    fp32_tensors_list.push_back(new_w);
                    fp32_upload_list.push_back({pair.first, w_f32_data});
                }
            }
        }

        if (!fp32_tensors_list.empty() || !fp16_tensors_list.empty()) {
            custom_buffer = ggml_backend_alloc_ctx_tensors(custom_ctx, backend);
            if (!custom_buffer) {
                std::cerr << "[BERT load] Error: Failed to allocate custom_buffer for weights!" << std::endl;
                return false;
            }

            for (size_t i = 0; i < fp32_tensors_list.size(); ++i) {
                struct ggml_tensor* nt = fp32_tensors_list[i];
                const auto& upload_entry = fp32_upload_list[i];
                ggml_backend_tensor_set(nt, upload_entry.data.data(), 0, upload_entry.data.size() * sizeof(float));
                tensors[upload_entry.name] = nt;
            }
            if (!fp32_tensors_list.empty()) {
                if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[BERT load] Pre-converted and uploaded " << fp32_tensors_list.size() << " weights to FP32 successfully." << std::endl;
            }

            for (size_t i = 0; i < fp16_tensors_list.size(); ++i) {
                struct ggml_tensor* nt = fp16_tensors_list[i];
                const auto& upload_entry = fp16_upload_list[i];
                ggml_backend_tensor_set(nt, upload_entry.data.data(), 0, upload_entry.data.size() * sizeof(ggml_fp16_t));
                tensors[upload_entry.name] = nt;
            }
            if (!fp16_tensors_list.empty()) {
                if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[BERT load] Pre-dequantized and uploaded " << fp16_tensors_list.size() << " embedding weights from Q4_0 to FP16 successfully." << std::endl;
            }
        } else {
            if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[BERT load] No weights found to pre-convert or pre-dequantize." << std::endl;
        }
    }

    return true;
}

struct ggml_tensor* BertModel::forward(struct ggml_context* ctx_graph, const std::vector<int32_t>& input_ids, ggml_backend_t backend) {
    int seq_len = (int)input_ids.size();
    if (seq_len == 0) {
        return ggml_new_tensor_2d(ctx_graph, GGML_TYPE_F32, 1024, 0);
    }
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[BERT Debug] forward start. seq_len=" << seq_len << std::endl; std::fflush(stdout);
    
    // Create self-contained context for BERT execution
    struct ggml_init_params init_params = {
        /* .mem_size   = */ 128 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true
    };
    struct ggml_context* ctx_bert = ggml_init(init_params);
    if (!ctx_bert) {
        std::cerr << "[BERT] Error: Failed to initialize private BERT context!\n";
        return nullptr;
    }
    
    // 1. Retrieve embedding tensors from loaded GGUFModel weights map
    struct ggml_tensor* word_embed = get_tensor("bert.embeddings.word_embeddings.weight");
    struct ggml_tensor* pos_embed = get_tensor("bert.embeddings.position_embeddings.weight");
    struct ggml_tensor* token_type_embed = get_tensor("bert.embeddings.token_type_embeddings.weight");
    
    struct ggml_tensor* ln_w = get_tensor("bert.embeddings.LayerNorm.weight");
    struct ggml_tensor* ln_b = get_tensor("bert.embeddings.LayerNorm.bias");
    
    if (!word_embed || !pos_embed || !token_type_embed || !ln_w || !ln_b) {
        std::cerr << "[GPT-SoVITS] Error: Missing BERT embedding tensors in GGUF file!\n";
        ggml_free(ctx_bert);
        return nullptr;
    }
    
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[BERT Debug] Embedding tensors retrieved." << std::endl; std::fflush(stdout);
    
    // 2. Setup input, position, and token type ID tensors inside a separate input context
    struct ggml_init_params input_params = {
        /* .mem_size   = */ 1 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true
    };
    struct ggml_context* ctx_inputs = ggml_init(input_params);
    if (!ctx_inputs) {
        std::cerr << "[BERT] Error: Failed to initialize inputs context!\n";
        ggml_free(ctx_bert);
        return nullptr;
    }

    struct ggml_tensor* input_ids_tensor = ggml_new_tensor_1d(ctx_inputs, GGML_TYPE_I32, seq_len);
    
    std::vector<int32_t> position_ids(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        position_ids[i] = i + 2;
    }
    struct ggml_tensor* position_ids_tensor = ggml_new_tensor_1d(ctx_inputs, GGML_TYPE_I32, seq_len);
    
    std::vector<int32_t> token_type_ids(seq_len, 0);
    struct ggml_tensor* token_type_ids_tensor = ggml_new_tensor_1d(ctx_inputs, GGML_TYPE_I32, seq_len);

    ggml_backend_buffer_t input_buffer = ggml_backend_alloc_ctx_tensors(ctx_inputs, backend);
    if (!input_buffer) {
        std::cerr << "[BERT] Error: Failed to allocate input buffer!\n";
        ggml_free(ctx_inputs);
        ggml_free(ctx_bert);
        return nullptr;
    }
    
    // 3. Extract embedding representations
    struct ggml_tensor* w_emb = ggml_get_rows(ctx_bert, word_embed, input_ids_tensor);
    struct ggml_tensor* p_emb = ggml_get_rows(ctx_bert, pos_embed, position_ids_tensor);
    struct ggml_tensor* t_emb = ggml_get_rows(ctx_bert, token_type_embed, token_type_ids_tensor);
    
    // Sum embeddings (Word + Position + Token Type)
    struct ggml_tensor* x = ggml_add(ctx_bert, ggml_add(ctx_bert, w_emb, p_emb), t_emb);
    // Cast to F32 for LayerNorm (CUDA norm ops require FP32)
    if (x->type != GGML_TYPE_F32) {
        x = ggml_cont(ctx_bert, ggml_cast(ctx_bert, x, GGML_TYPE_F32));
    }
    
    // Embeddings LayerNorm
    x = ggml_norm(ctx_bert, x, 1e-12f);
    x = ggml_add(ctx_bert, ggml_mul(ctx_bert, x, ln_w), ln_b);
    
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[BERT Debug] Embeddings graph built." << std::endl; std::fflush(stdout);
    
    // 4. Construct 22 Encoder Layer Blocks (aligned with PyTorch feature extraction at Layer 22)
    int num_layers = 22;
    int n_heads = 16;
    int head_dim = 64; // 1024 hidden / 16 heads
    
    for (int layer = 0; layer < num_layers; ++layer) {
        std::string layer_prefix = "bert.encoder.layer." + std::to_string(layer) + ".";
        
        // Retrieve self-attention weights & biases
        struct ggml_tensor* qw = get_tensor(layer_prefix + "attention.self.query.weight");
        struct ggml_tensor* qb = get_tensor(layer_prefix + "attention.self.query.bias");
        struct ggml_tensor* kw = get_tensor(layer_prefix + "attention.self.key.weight");
        struct ggml_tensor* kb = get_tensor(layer_prefix + "attention.self.key.bias");
        struct ggml_tensor* vw = get_tensor(layer_prefix + "attention.self.value.weight");
        struct ggml_tensor* vb = get_tensor(layer_prefix + "attention.self.value.bias");
        
        // Retrieve attention output projection & LayerNorm
        struct ggml_tensor* out_w = get_tensor(layer_prefix + "attention.output.dense.weight");
        struct ggml_tensor* out_b = get_tensor(layer_prefix + "attention.output.dense.bias");
        struct ggml_tensor* out_ln_w = get_tensor(layer_prefix + "attention.output.LayerNorm.weight");
        struct ggml_tensor* out_ln_b = get_tensor(layer_prefix + "attention.output.LayerNorm.bias");
        
        // Retrieve MLP intermediate & output dense layers + LayerNorm
        struct ggml_tensor* ffn_w1 = get_tensor(layer_prefix + "intermediate.dense.weight");
        struct ggml_tensor* ffn_b1 = get_tensor(layer_prefix + "intermediate.dense.bias");
        struct ggml_tensor* ffn_w2 = get_tensor(layer_prefix + "output.dense.weight");
        struct ggml_tensor* ffn_b2 = get_tensor(layer_prefix + "output.dense.bias");
        struct ggml_tensor* ffn_ln_w = get_tensor(layer_prefix + "output.LayerNorm.weight");
        struct ggml_tensor* ffn_ln_b = get_tensor(layer_prefix + "output.LayerNorm.bias");
        
        if (!qw || !qb || !kw || !kb || !vw || !vb || !out_w || !out_b || !out_ln_w || !out_ln_b ||
            !ffn_w1 || !ffn_b1 || !ffn_w2 || !ffn_b2 || !ffn_ln_w || !ffn_ln_b) {
            std::cerr << "[GPT-SoVITS] Error: Missing BERT Layer " << layer << " tensors in GGUF weight mapping!\n";
            ggml_free(ctx_bert);
            return nullptr;
        }
        
        // --- Multi-Head Self Attention (MHA) ---
        struct ggml_tensor* Q = ggml_add(ctx_bert, ggml_mul_mat(ctx_bert, qw, x), qb);
        struct ggml_tensor* K = ggml_add(ctx_bert, ggml_mul_mat(ctx_bert, kw, x), kb);
        struct ggml_tensor* V = ggml_add(ctx_bert, ggml_mul_mat(ctx_bert, vw, x), vb);
        
        // Reshape: [1024, seq_len] -> [64, 16, seq_len]
        Q = ggml_reshape_3d(ctx_bert, Q, head_dim, n_heads, seq_len);
        K = ggml_reshape_3d(ctx_bert, K, head_dim, n_heads, seq_len);
        V = ggml_reshape_3d(ctx_bert, V, head_dim, n_heads, seq_len);
        
        // Permute to batch self attention layout
        struct ggml_tensor* Q_perm = ggml_permute(ctx_bert, Q, 0, 2, 1, 3);
        struct ggml_tensor* K_perm = ggml_permute(ctx_bert, K, 0, 2, 1, 3);
        struct ggml_tensor* V_perm = ggml_permute(ctx_bert, V, 1, 2, 0, 3);
        
        // Make inputs contiguous for complete backend compatibility (avoiding permuted strides in GEMM)
        struct ggml_tensor* Q_cont = ggml_cont(ctx_bert, Q_perm);
        struct ggml_tensor* K_cont = ggml_cont(ctx_bert, K_perm);
        
        // Scaled dot-product
        struct ggml_tensor* kq = ggml_mul_mat(ctx_bert, K_cont, Q_cont);
        kq = ggml_scale(ctx_bert, kq, 1.0f / std::sqrt((float)head_dim));
        kq = ggml_ops_soft_max(ctx_bert, kq, backend);
        
        // Context multiplication
        struct ggml_tensor* V_cont = ggml_cont(ctx_bert, V_perm);
        struct ggml_tensor* kqv = ggml_mul_mat(ctx_bert, V_cont, kq);
        
        // Permute back & reshape to 2D
        kqv = ggml_permute(ctx_bert, kqv, 0, 2, 1, 3);
        kqv = ggml_cont(ctx_bert, kqv);
        kqv = ggml_reshape_2d(ctx_bert, kqv, 1024, seq_len);
        
        struct ggml_tensor* out_proj = ggml_mul_mat(ctx_bert, out_w, kqv);
        struct ggml_tensor* attn_out = ggml_add(ctx_bert, out_proj, out_b);
        
        // First Residual Addition & LayerNorm
        x = ggml_add(ctx_bert, x, attn_out);
        x = ggml_norm(ctx_bert, x, 1e-12f);
        
        struct ggml_tensor* x_mul = ggml_mul(ctx_bert, x, out_ln_w);
        x = ggml_add(ctx_bert, x_mul, out_ln_b);
        
        // --- MLP Feed-Forward Network (FFN) ---
        struct ggml_tensor* h = ggml_add(ctx_bert, ggml_mul_mat(ctx_bert, ffn_w1, x), ffn_b1);
        h = ggml_gelu(ctx_bert, h);
        struct ggml_tensor* mlp_out = ggml_add(ctx_bert, ggml_mul_mat(ctx_bert, ffn_w2, h), ffn_b2);
        
        // Second Residual Addition & LayerNorm
        x = ggml_add(ctx_bert, x, mlp_out);
        x = ggml_norm(ctx_bert, x, 1e-12f);
        x = ggml_add(ctx_bert, ggml_mul(ctx_bert, x, ffn_ln_w), ffn_ln_b);
    }
    
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[BERT Debug] Blocks graph built. Allocating buffer..." << std::endl; std::fflush(stdout);
    
    // Build the graph on the backend
    struct ggml_cgraph* gf = ggml_new_graph(ctx_bert);
    ggml_build_forward_expand(gf, x);

    // Create and use the graph allocator (ggml_gallocr) for memory planning and alignment
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!galloc) {
        std::cerr << "[BERT] Error: Failed to create graph allocator (gallocr)!\n";
        ggml_free(ctx_bert);
        return nullptr;
    }
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        std::cerr << "[BERT] Error: Failed to allocate graph using gallocr!\n";
        ggml_gallocr_free(galloc);
        ggml_free(ctx_bert);
        return nullptr;
    }
    
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[BERT Debug] Graph allocated. Uploading inputs..." << std::endl; std::fflush(stdout);
    
    // Copy the input CPU data to the GPU allocated backend tensors
    ggml_backend_tensor_set(input_ids_tensor, input_ids.data(), 0, seq_len * sizeof(int32_t));
    ggml_backend_tensor_set(position_ids_tensor, position_ids.data(), 0, seq_len * sizeof(int32_t));
    ggml_backend_tensor_set(token_type_ids_tensor, token_type_ids.data(), 0, seq_len * sizeof(int32_t));
    
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[BERT Debug] Inputs uploaded. Computing graph..." << std::endl; std::fflush(stdout);
    
    ggml_backend_graph_compute(backend, gf);
    
    // Retrieve computed output features back to CPU
    int out_elements = 1024 * seq_len;
    std::vector<float> host_out(out_elements);
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[BERT Debug] Completed compute. Retrieving output tensor (elements=" << out_elements << ")..." << std::endl; std::fflush(stdout);
    ggml_backend_tensor_get(x, host_out.data(), 0, out_elements * sizeof(float));
    
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[BERT Debug] Output tensor retrieved. Freeing gallocr..." << std::endl; std::fflush(stdout);
    ggml_gallocr_free(galloc);
    
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[BERT Debug] gallocr freed. Freeing inputs..." << std::endl; std::fflush(stdout);
    ggml_backend_buffer_free(input_buffer);
    ggml_free(ctx_inputs);

    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[BERT Debug] inputs freed. Freeing ctx_bert..." << std::endl; std::fflush(stdout);
    ggml_free(ctx_bert);
    
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[BERT Debug] ctx_bert freed. Creating output tensor in parent context..." << std::endl; std::fflush(stdout);
    // Create new tensor in the parent context containing the pre-computed features
    struct ggml_tensor* out_tensor = ggml_new_tensor_2d(ctx_graph, GGML_TYPE_F32, 1024, seq_len);
    std::memcpy(out_tensor->data, host_out.data(), out_elements * sizeof(float));
    
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[BERT Debug] Finished forward pass successfully. returning out_tensor=" << out_tensor << std::endl; std::fflush(stdout);
    return out_tensor;
}

} // namespace gpt_sovits
