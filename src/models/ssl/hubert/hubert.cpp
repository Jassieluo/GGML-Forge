#include "hubert.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ops/ops.h"
#include "nn/nn.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace gpt_sovits {

#pragma pack(push, 1)
struct ggml_hubert_block_q4_0 {
    ggml_fp16_t d;       // delta
    uint8_t qs[16];      // 32 nibbles
};
#pragma pack(pop)

static void dequantize_q4_0_to_fp16(const uint8_t * src, ggml_fp16_t * dst, int64_t nelements) {
    int64_t qk = 32;
    int64_t nb = nelements / qk;
    const ggml_hubert_block_q4_0 * blocks = (const ggml_hubert_block_q4_0 *)src;
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
    const ggml_hubert_block_q4_0 * blocks = (const ggml_hubert_block_q4_0 *)src;
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

static ggml_backend_t current_hubert_backend = nullptr;

static struct ggml_tensor* mul_f32(struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b) {
    return ggml_mul_mat(ctx, a, b);
}

static struct ggml_tensor* ggml_conv_1d_hubert(
    struct ggml_context* ctx,
    struct ggml_tensor* w,
    struct ggml_tensor* x,
    int stride,
    int padding,
    int dilation,
    ggml_backend_t backend
) {
    return ggml_ops_conv_1d(ctx, w, x, stride, padding, dilation, backend);
}

void HubertModel::on_read_metadata(struct gguf_context* ctx_gguf) {
    int kid = gguf_find_key(ctx_gguf, "attention.head_count");
    if (kid >= 0) {
        n_heads = (int)gguf_get_val_u32(ctx_gguf, kid);
    } else {
        n_heads = 12; // Default for wav2vec2-base
    }
}

bool HubertModel::load(const std::string& path, ggml_backend_t backend) {
    if (!load_gguf_model(path, *this, backend)) {
        return false;
    }

    // Pre-convert FP16 weights to FP32 for non-CUDA backends
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
        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[CNHuBERT load] Non-CUDA backend detected. Performing weights pre-processing..." << std::endl;
        struct ggml_init_params custom_params = {
            /* .mem_size   = */ 64 * 1024 * 1024,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true
        };
        custom_ctx = ggml_init(custom_params);
        if (!custom_ctx) {
            std::cerr << "[CNHuBERT load] Error: Failed to initialize custom_ctx for FP32 weights!" << std::endl;
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

            if (old_w->type == GGML_TYPE_F16) {
                if (pair.first.find("pos_conv_embed") != std::string::npos) {
                    continue; // Skip positional conv weight
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
            } else if (old_w->type == GGML_TYPE_Q4_0) {
                if (is_sycl) {
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

        if (!fp32_tensors_list.empty()) {
            custom_buffer = ggml_backend_alloc_ctx_tensors(custom_ctx, backend);
            if (!custom_buffer) {
                std::cerr << "[CNHuBERT load] Error: Failed to allocate custom_buffer!" << std::endl;
                return false;
            }

            for (size_t i = 0; i < fp32_tensors_list.size(); ++i) {
                struct ggml_tensor* nt = fp32_tensors_list[i];
                const auto& upload_entry = fp32_upload_list[i];
                ggml_backend_tensor_set(nt, upload_entry.data.data(), 0, upload_entry.data.size() * sizeof(float));
                tensors[upload_entry.name] = nt;
            }
        }
    }

    // Weight Normalization setup for positional conv
    struct ggml_tensor* pos_conv_g = get_tensor("encoder.pos_conv_embed.conv.weight_g");
    struct ggml_tensor* pos_conv_v = get_tensor("encoder.pos_conv_embed.conv.weight_v");

    if (pos_conv_g && pos_conv_v) {
        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[CNHuBERT] Pre-computing folded positional convolution weight normalization..." << std::endl;
        struct ggml_init_params init_params = {
            /* .mem_size   = */ 32 * 1024 * 1024,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ false
        };
        struct ggml_context* ctx_norm = ggml_init(init_params);

        std::vector<float> g_float(ggml_nelements(pos_conv_g));
        std::vector<float> v_float(ggml_nelements(pos_conv_v));

        if (pos_conv_g->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> g_fp16(g_float.size());
            ggml_backend_tensor_get(pos_conv_g, g_fp16.data(), 0, g_fp16.size() * sizeof(ggml_fp16_t));
            for (size_t i = 0; i < g_float.size(); ++i) {
                g_float[i] = ggml_fp16_to_fp32(g_fp16[i]);
            }
        } else {
            ggml_backend_tensor_get(pos_conv_g, g_float.data(), 0, g_float.size() * sizeof(float));
        }

        if (pos_conv_v->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> v_fp16(v_float.size());
            ggml_backend_tensor_get(pos_conv_v, v_fp16.data(), 0, v_fp16.size() * sizeof(ggml_fp16_t));
            for (size_t i = 0; i < v_float.size(); ++i) {
                v_float[i] = ggml_fp16_to_fp32(v_fp16[i]);
            }
        } else {
            ggml_backend_tensor_get(pos_conv_v, v_float.data(), 0, v_float.size() * sizeof(float));
        }

        int out_channels = pos_conv_g->ne[0];
        int in_channels = pos_conv_v->ne[1];
        int kernel_size = pos_conv_v->ne[0];

        std::vector<float> folded_weights(out_channels * in_channels * kernel_size);

        if (pos_conv_v->ne[2] == 1) {
            for (int oc = 0; oc < out_channels; ++oc) {
                double sum_sq = 0.0;
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int k = 0; k < kernel_size; ++k) {
                        float val = v_float[oc * (in_channels * kernel_size) + ic * kernel_size + k];
                        sum_sq += val * val;
                    }
                }
                const float norm = (float)std::sqrt(sum_sq);
                const float scale = g_float[oc] / (norm + 1e-12f);
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int k = 0; k < kernel_size; ++k) {
                        int idx = oc * (in_channels * kernel_size) + ic * kernel_size + k;
                        folded_weights[idx] = v_float[idx] * scale;
                    }
                }
            }
        } else {
            for (int k = 0; k < out_channels; ++k) {
                double sum_sq = 0.0;
                for (int j = 0; j < in_channels; ++j) {
                    for (int i = 0; i < kernel_size; ++i) {
                        float val = v_float[k * (in_channels * kernel_size) + j * kernel_size + i];
                        sum_sq += val * val;
                    }
                }
                const float norm = (float)std::sqrt(sum_sq);
                const float scale = g_float[k] / (norm + 1e-12f);
                for (int j = 0; j < in_channels; ++j) {
                    for (int i = 0; i < kernel_size; ++i) {
                        int idx = k * (in_channels * kernel_size) + j * kernel_size + i;
                        folded_weights[idx] = v_float[idx] * scale;
                    }
                }
            }
        }

        pos_conv_w = ggml_new_tensor_3d(ctx_norm, GGML_TYPE_F32, kernel_size, in_channels, out_channels);
        std::memcpy(pos_conv_w->data, folded_weights.data(), folded_weights.size() * sizeof(float));

        pos_conv_w_buffer = ggml_backend_alloc_ctx_tensors(ctx_norm, backend);
        ggml_backend_tensor_set(pos_conv_w, folded_weights.data(), 0, folded_weights.size() * sizeof(float));
        tensors["encoder.pos_conv_embed.conv.weight"] = pos_conv_w;

        ggml_free(ctx_norm);
    } else {
        std::cerr << "[CNHuBERT] Warning: Missing weight_g or weight_v positional weight norm tensors!\n";
    }

    return true;
}

struct ggml_tensor* HubertModel::forward(struct ggml_context* ctx_graph, struct ggml_tensor* input_audio, ggml_backend_t backend) {
    current_hubert_backend = backend;
    int audio_len = (int)input_audio->ne[0];
    if (audio_len == 0) {
        return ggml_new_tensor_2d(ctx_graph, GGML_TYPE_F32, 768, 0);
    }
    
    // Create self-contained context for HuBERT execution
    struct ggml_init_params init_params = {
        /* .mem_size   = */ 512 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true
    };
    struct ggml_context* ctx_hubert = ggml_init(init_params);
    if (!ctx_hubert) {
        std::cerr << "[CNHuBERT] Error: Failed to initialize private CNHuBERT context!\n";
        return nullptr;
    }

    // 1. Retrieve all required convolution and projection weights
    struct ggml_tensor* w0 = get_tensor("feature_extractor.conv_layers.0.conv.weight");
    struct ggml_tensor* ln0_w = get_tensor("feature_extractor.conv_layers.0.layer_norm.weight");
    struct ggml_tensor* ln0_b = get_tensor("feature_extractor.conv_layers.0.layer_norm.bias");
    
    struct ggml_tensor* w1 = get_tensor("feature_extractor.conv_layers.1.conv.weight");
    struct ggml_tensor* w2 = get_tensor("feature_extractor.conv_layers.2.conv.weight");
    struct ggml_tensor* w3 = get_tensor("feature_extractor.conv_layers.3.conv.weight");
    struct ggml_tensor* w4 = get_tensor("feature_extractor.conv_layers.4.conv.weight");
    struct ggml_tensor* w5 = get_tensor("feature_extractor.conv_layers.5.conv.weight");
    struct ggml_tensor* w6 = get_tensor("feature_extractor.conv_layers.6.conv.weight");
    
    struct ggml_tensor* proj_ln_w = get_tensor("feature_projection.layer_norm.weight");
    struct ggml_tensor* proj_ln_b = get_tensor("feature_projection.layer_norm.bias");
    struct ggml_tensor* proj_w = get_tensor("feature_projection.projection.weight");
    struct ggml_tensor* proj_b = get_tensor("feature_projection.projection.bias");
    
    struct ggml_tensor* pos_conv_b = get_tensor("encoder.pos_conv_embed.conv.bias");
    struct ggml_tensor* encoder_ln_w = get_tensor("encoder.layer_norm.weight");
    struct ggml_tensor* encoder_ln_b = get_tensor("encoder.layer_norm.bias");
    
    if (!w0 || !ln0_w || !ln0_b || !w1 || !w2 || !w3 || !w4 || !w5 || !w6 ||
        !proj_ln_w || !proj_ln_b || !proj_w || !proj_b || !pos_conv_b || !encoder_ln_w || !encoder_ln_b) {
        std::cerr << "[CNHuBERT] Error: Missing weights or normalization parameters in GGUF weight mapping!\n";
        ggml_free(ctx_hubert);
        return nullptr;
    }

    // Wrap GroupNorm / InstanceNorm layer
    nn::InstanceNorm ln0(ln0_w, ln0_b, 1e-5f);
    nn::LayerNorm proj_ln(proj_ln_w, proj_ln_b, 1e-5f);
    nn::Linear proj_dense(proj_w, proj_b);
    nn::LayerNorm encoder_ln(encoder_ln_w, encoder_ln_b, 1e-5f);
    
    // Create input audio tensor in ctx_hubert
    struct ggml_tensor* input_audio_tensor = ggml_new_tensor_1d(ctx_hubert, GGML_TYPE_F32, audio_len);
    struct ggml_tensor* cnn_conv0_dbg = nullptr;
    struct ggml_tensor* cnn_conv0_ln_dbg = nullptr;
    
    // 2. CNN Feature Extractor (7 Conv1D layers)
    struct ggml_tensor* x = input_audio_tensor;
    
    // Layer 0: Conv1D (kernel=10, stride=5, no-padding)
    x = ggml_conv_1d_hubert(ctx_hubert, w0, x, 5, 0, 1, backend);
    cnn_conv0_dbg = ggml_cont(ctx_hubert, x);
    
    // Layer 0 GroupNorm (groups=512, channels=512) -> represented as nn::InstanceNorm
    int seq_len_0 = (int)x->ne[0];
    x = ggml_reshape_2d(ctx_hubert, x, seq_len_0, 512);
    x = ln0.forward(ctx_hubert, x, backend);
    cnn_conv0_ln_dbg = ggml_cont(ctx_hubert, x);
    x = ggml_gelu_erf(ctx_hubert, x);
    
    // CUDA IM2COL layout setup
    x = ggml_reshape_3d(ctx_hubert, x, seq_len_0, 512, 1);
    x = ggml_cont(ctx_hubert, x);
    
    // Layer 1 to 6: Conv1D + GELU
    x = ggml_conv_1d_hubert(ctx_hubert, w1, x, 2, 0, 1, backend); x = ggml_gelu_erf(ctx_hubert, x);
    x = ggml_conv_1d_hubert(ctx_hubert, w2, x, 2, 0, 1, backend); x = ggml_gelu_erf(ctx_hubert, x);
    x = ggml_conv_1d_hubert(ctx_hubert, w3, x, 2, 0, 1, backend); x = ggml_gelu_erf(ctx_hubert, x);
    x = ggml_conv_1d_hubert(ctx_hubert, w4, x, 2, 0, 1, backend); x = ggml_gelu_erf(ctx_hubert, x);
    x = ggml_conv_1d_hubert(ctx_hubert, w5, x, 2, 0, 1, backend); x = ggml_gelu_erf(ctx_hubert, x);
    x = ggml_conv_1d_hubert(ctx_hubert, w6, x, 2, 0, 1, backend); x = ggml_gelu_erf(ctx_hubert, x);
    struct ggml_tensor* feature_extractor_dbg = ggml_cont(ctx_hubert, x);
    
    int seq_len = (int)x->ne[0];
    
    // 3. Feature Projection (512 -> 768)
    struct ggml_tensor* x_proj = ggml_permute(ctx_hubert, x, 1, 0, 2, 3);
    x_proj = ggml_cont(ctx_hubert, x_proj);
    x_proj = proj_ln.forward(ctx_hubert, x_proj, backend);
    x_proj = proj_dense.forward(ctx_hubert, x_proj);
    
    // 4. Positional Convolution Embedding (Grouped Conv1D with groups=16, kernel=128)
    struct ggml_tensor* x_pos_input_2d = ggml_cont(ctx_hubert, ggml_transpose(ctx_hubert, x_proj));
    struct ggml_tensor* x_pos_input = ggml_reshape_3d(ctx_hubert, x_pos_input_2d, seq_len, 768, 1);
    struct ggml_tensor* pos_conv_w_tensor = pos_conv_w;
    
    std::vector<struct ggml_tensor*> slices(16);
    for (int g = 0; g < 16; ++g) {
        struct ggml_tensor* x_g = ggml_view_3d(ctx_hubert, x_pos_input, seq_len, 48, 1, x_pos_input->nb[1], x_pos_input->nb[2], g * 48 * x_pos_input->nb[1]);
        struct ggml_tensor* w_g = ggml_view_3d(ctx_hubert, pos_conv_w_tensor, 128, 48, 48, pos_conv_w_tensor->nb[1], pos_conv_w_tensor->nb[2], g * 48 * pos_conv_w_tensor->nb[2]);
        struct ggml_tensor* conv_out = ggml_conv_1d_hubert(ctx_hubert, w_g, x_g, 1, 64, 1, backend);
        slices[g] = ggml_view_3d(ctx_hubert, conv_out, seq_len, 48, 1, conv_out->nb[1], conv_out->nb[2], 0);
    }
    
    struct ggml_tensor* pos_emb = slices[0];
    for (int g = 1; g < 16; ++g) {
        pos_emb = ggml_concat(ctx_hubert, pos_emb, slices[g], 1);
    }
    
    pos_emb = ggml_reshape_2d(ctx_hubert, pos_emb, seq_len, 768);
    pos_emb = ggml_permute(ctx_hubert, pos_emb, 1, 0, 2, 3);
    pos_emb = ggml_cont(ctx_hubert, pos_emb);
    struct ggml_tensor* pos_conv_b_reshaped = ggml_reshape_2d(ctx_hubert, pos_conv_b, 768, 1);
    pos_emb = ggml_add(ctx_hubert, pos_emb, pos_conv_b_reshaped);
    pos_emb = ggml_cont(ctx_hubert, pos_emb);
    pos_emb = ggml_gelu_erf(ctx_hubert, pos_emb);
    
    struct ggml_tensor* hidden_states = ggml_add(ctx_hubert, x_proj, pos_emb);
    hidden_states = ggml_cont(ctx_hubert, hidden_states);
    hidden_states = encoder_ln.forward(ctx_hubert, hidden_states, backend);
    struct ggml_tensor* x_normalized = hidden_states;
    
    // 5. Construct 12 Transformer Encoder layers using nn::TransformerEncoderLayer
    int num_layers = 12;
    int head_dim = 64; // 768 hidden / 12 heads
    struct ggml_tensor* layer0_output = nullptr;
    struct ggml_tensor* layer0_ln1 = nullptr;
    struct ggml_tensor* layer0_Q = nullptr;
    struct ggml_tensor* layer0_K = nullptr;
    struct ggml_tensor* layer0_V = nullptr;
    struct ggml_tensor* layer0_kq = nullptr;
    struct ggml_tensor* layer0_kqv = nullptr;
    struct ggml_tensor* layer0_attn_out = nullptr;
    struct ggml_tensor* layer0_x_attn = nullptr;
    struct ggml_tensor* layer0_pre_final_norm = nullptr;
    struct ggml_tensor* layer0_ln2 = nullptr;
    struct ggml_tensor* layer0_h = nullptr;
    struct ggml_tensor* layer0_mlp_out = nullptr;
    struct ggml_tensor* layer1_output = nullptr;
    struct ggml_tensor* layer5_output = nullptr;
    
    for (int layer = 0; layer < num_layers; ++layer) {
        std::string prefix = "encoder.layers." + std::to_string(layer) + ".";
        
        struct ggml_tensor* qw = get_tensor(prefix + "attention.q_proj.weight");
        struct ggml_tensor* qb = get_tensor(prefix + "attention.q_proj.bias");
        struct ggml_tensor* kw = get_tensor(prefix + "attention.k_proj.weight");
        struct ggml_tensor* kb = get_tensor(prefix + "attention.k_proj.bias");
        struct ggml_tensor* vw = get_tensor(prefix + "attention.v_proj.weight");
        struct ggml_tensor* vb = get_tensor(prefix + "attention.v_proj.bias");
        struct ggml_tensor* out_w = get_tensor(prefix + "attention.out_proj.weight");
        struct ggml_tensor* out_b = get_tensor(prefix + "attention.out_proj.bias");
        
        struct ggml_tensor* ln1_w = get_tensor(prefix + "layer_norm.weight");
        struct ggml_tensor* ln1_b = get_tensor(prefix + "layer_norm.bias");
        struct ggml_tensor* ln2_w = get_tensor(prefix + "final_layer_norm.weight");
        struct ggml_tensor* ln2_b = get_tensor(prefix + "final_layer_norm.bias");
        
        struct ggml_tensor* ffn_w1 = get_tensor(prefix + "feed_forward.intermediate_dense.weight");
        struct ggml_tensor* ffn_b1 = get_tensor(prefix + "feed_forward.intermediate_dense.bias");
        struct ggml_tensor* ffn_w2 = get_tensor(prefix + "feed_forward.output_dense.weight");
        struct ggml_tensor* ffn_b2 = get_tensor(prefix + "feed_forward.output_dense.bias");
        
        if (!qw || !qb || !kw || !kb || !vw || !vb || !out_w || !out_b || !ln1_w || !ln1_b || !ln2_w || !ln2_b ||
            !ffn_w1 || !ffn_b1 || !ffn_w2 || !ffn_b2) {
            std::cerr << "[CNHuBERT] Error: Missing layer " << layer << " weights in GGUF weight mapping!\n";
            ggml_free(ctx_hubert);
            return nullptr;
        }
        
        // Wrap layer in nn::TransformerEncoderLayer (Post-LN)
        nn::TransformerEncoderLayer encoder_layer(
            qw, qb, kw, kb, vw, vb, out_w, out_b, n_heads, head_dim,
            ffn_w1, ffn_b1, ffn_w2, ffn_b2, nn::ActivationType::GELU_ERF,
            ln1_w, ln1_b, ln2_w, ln2_b, 1e-5f, false // Post-LN
        );
        
        // Run forward
        hidden_states = encoder_layer.forward(ctx_hubert, hidden_states, nullptr, backend);
        
        // Keep track of debug outputs for comparison tests
        if (layer == 0) {
            layer0_output = hidden_states;
            layer0_ln1 = ggml_ops_layer_norm(ctx_hubert, hidden_states, ln1_w, ln1_b, 1e-5f, backend); // mock for test compatibility
            layer0_ln2 = hidden_states;
        } else if (layer == 1) {
            layer1_output = hidden_states;
        } else if (layer == 5) {
            layer5_output = hidden_states;
        }
    }
    
    // Allocate all tensors in ctx_hubert on the backend
    ggml_backend_buffer_t hubert_buffer = ggml_backend_alloc_ctx_tensors(ctx_hubert, backend);
    if (!hubert_buffer) {
        std::cerr << "[CNHuBERT] Error: Failed to allocate GPU backend buffer for CNHuBERT context!\n";
        ggml_free(ctx_hubert);
        return nullptr;
    }
    
    if (std::getenv("GPT_SOVITS_DEBUG") != nullptr) {
        std::cerr << "[CNHuBERT Debug] GPU Tensors Allocation check:\n"
                  << "  input_audio_tensor ptr=" << input_audio_tensor << ", data=" << (void*)input_audio_tensor->data << ", ne0=" << input_audio_tensor->ne[0] << "\n"
                  << "  w0 ptr=" << w0 << ", data=" << (void*)w0->data << ", ne0=" << w0->ne[0] << "\n" << std::endl;
    }
    ggml_backend_tensor_set(input_audio_tensor, input_audio->data, 0, audio_len * sizeof(float));
    
    // Build and compute the graph on the backend
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx_hubert, 32768, false);
    ggml_build_forward_expand(gf, hidden_states);
    if (cnn_conv0_dbg) {
        ggml_build_forward_expand(gf, cnn_conv0_dbg);
    }
    if (cnn_conv0_ln_dbg) {
        ggml_build_forward_expand(gf, cnn_conv0_ln_dbg);
    }
    if (feature_extractor_dbg) {
        ggml_build_forward_expand(gf, feature_extractor_dbg);
    }
    ggml_backend_graph_compute(backend, gf);
 
    // Debug intermediate CNHuBERT tensors
    if (std::getenv("GPT_SOVITS_DEBUG") != nullptr) {
        {
            auto print_tensor_info = [&](const std::string& name, struct ggml_tensor* t) {
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
                std::cout << "  C++ " << name << " - Shape: [" << t->ne[0] << ", " << t->ne[1] << ", " << t->ne[2] << "]\n";
                std::cout << "      Min/Max: " << min_val << " / " << max_val << "\n";
            };
            
            std::cout << "\n[CNHuBERT C++ Intermediate Debug]:\n";
            print_tensor_info("feature_extractor (CNN)", x);
            print_tensor_info("feature_projection", x_proj);
            print_tensor_info("pos_conv_embed", pos_emb);
            print_tensor_info("encoder_layer_norm", x_normalized);
            
            std::cout << "\n--- Layer Output Debug ---\n";
            print_tensor_info("layer_0 output", layer0_output);
            print_tensor_info("layer_1 output", layer1_output);
            print_tensor_info("layer_5 output", layer5_output);
            print_tensor_info("cnn_conv0", cnn_conv0_dbg);
            print_tensor_info("cnn_conv0_ln", cnn_conv0_ln_dbg);
            
            std::cout << "\n--- Final Model Output ---\n";
            print_tensor_info("hidden_states (layer 11)", hidden_states);
 
            dump_tensor_f32_if_requested("GPT_SOVITS_HUBERT_DUMP", "layer0_output", layer0_output);
            dump_tensor_f32_if_requested("GPT_SOVITS_HUBERT_DUMP", "layer1_output", layer1_output);
            dump_tensor_f32_if_requested("GPT_SOVITS_HUBERT_DUMP", "layer5_output", layer5_output);
            dump_tensor_f32_if_requested("GPT_SOVITS_HUBERT_DUMP", "layer11_output", hidden_states);
            dump_tensor_f32_if_requested("GPT_SOVITS_HUBERT_DUMP", "cnn_conv0", cnn_conv0_dbg);
            dump_tensor_f32_if_requested("GPT_SOVITS_HUBERT_DUMP", "cnn_conv0_ln", cnn_conv0_ln_dbg);
            dump_tensor_f32_if_requested("GPT_SOVITS_HUBERT_DUMP", "feature_extractor", feature_extractor_dbg);
        }
    }
    
    // Retrieve computed output features back to CPU
    int out_elements = 768 * seq_len;
    std::vector<float> host_out(out_elements);
    ggml_backend_tensor_get(hidden_states, host_out.data(), 0, out_elements * sizeof(float));
    
    // Clean up private context
    ggml_backend_buffer_free(hubert_buffer);
    ggml_free(ctx_hubert);
    
    // Create new tensor in the parent context containing the pre-computed features
    struct ggml_tensor* out_tensor = ggml_new_tensor_2d(ctx_graph, GGML_TYPE_F32, 768, seq_len);
    std::memcpy(out_tensor->data, host_out.data(), out_elements * sizeof(float));
    
    return out_tensor;
}

} // namespace gpt_sovits
