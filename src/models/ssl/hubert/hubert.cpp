#include "hubert.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <cstdlib>

namespace gpt_sovits {

// Thread-local backend state to decide between CPU and cuDNN kernels dynamically
thread_local ggml_backend_t current_hubert_backend = nullptr;

static struct ggml_tensor* force_w_f32(struct ggml_context* ctx, struct ggml_tensor* w);

static struct ggml_tensor* custom_conv_1d(
    struct ggml_context* ctx,
    struct ggml_tensor* w,
    struct ggml_tensor* x,
    int stride,
    int padding,
    int dilation
) {
    // 1. Force weights to float32
    struct ggml_tensor* w_f32 = force_w_f32(ctx, w);
    
    // 2. Reshape x if it's 1D
    struct ggml_tensor* x_reshaped = x;
    if (ggml_n_dims(x) == 1) {
        x_reshaped = ggml_reshape_2d(ctx, x, x->ne[0], 1);
    }
    
    // 3. Transpose x to [in_channels, seq_len]
    struct ggml_tensor* x_t = ggml_cont(ctx, ggml_transpose(ctx, x_reshaped));
    
    int64_t kernel_size = w_f32->ne[0];
    int64_t in_channels = w_f32->ne[1];
    int64_t out_channels = w_f32->ne[2];
    int64_t seq_len = x_t->ne[1];
    
    int64_t out_seq_len = (seq_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int64_t max_padded_idx = (out_seq_len - 1) * stride + (kernel_size - 1) * dilation;
    int64_t req_padded_len = max_padded_idx + 1;
    
    int64_t left_pad = padding;
    int64_t right_pad = req_padded_len - seq_len - left_pad;
    if (right_pad < 0) right_pad = 0;
    
    // 4. Pad input along the sequence dimension (dim 1)
    struct ggml_tensor* x_pad = ggml_pad_ext(ctx, x_t, 0, 0, left_pad, right_pad, 0, 0, 0, 0);
    
    // 5. Permute and contiguous weights to [in_channels, out_channels, kernel_size]
    struct ggml_tensor* w_perm = ggml_cont(ctx, ggml_permute(ctx, w_f32, 2, 0, 1, 3));
    
    // 6. Loop over kernel elements and accumulate
    struct ggml_tensor* sum = nullptr;
    for (int k = 0; k < kernel_size; ++k) {
        // Slice input sequence
        struct ggml_tensor* x_k_view = ggml_view_2d(ctx, x_pad, in_channels, out_seq_len, stride * x_pad->nb[1], k * dilation * x_pad->nb[1]);
        struct ggml_tensor* x_k = ggml_cont(ctx, x_k_view);
        
        // Slice weights
        struct ggml_tensor* w_k_view = ggml_view_2d(ctx, w_perm, in_channels, out_channels, w_perm->nb[1], k * w_perm->nb[2]);
        struct ggml_tensor* w_k = ggml_cont(ctx, w_k_view);
        
        // Matrix multiply: [in_channels, out_seq_len] * [in_channels, out_channels]^T -> [out_channels, out_seq_len]
        struct ggml_tensor* prod = ggml_mul_mat(ctx, x_k, w_k);
        
        if (sum == nullptr) {
            sum = prod;
        } else {
            sum = ggml_add(ctx, sum, prod);
        }
    }
    
    // 7. Return contiguous output of shape [out_seq_len, out_channels]
    return ggml_cont(ctx, sum);
}

static struct ggml_tensor* ggml_conv_1d_hubert(
    struct ggml_context* ctx,
    struct ggml_tensor* w,
    struct ggml_tensor* x,
    int stride,
    int padding,
    int dilation
) {
    bool is_sycl = false;
    if (current_hubert_backend) {
        const char * bname = ggml_backend_name(current_hubert_backend);
        if (bname && strncmp(bname, "SYCL", 4) == 0) {
            is_sycl = true;
        }
    }
    
    if (is_sycl) {
        return custom_conv_1d(ctx, w, x, stride, padding, dilation);
    }
    return ggml_conv_1d(ctx, w, x, stride, padding, dilation);
}

static struct ggml_tensor* force_w_f32(struct ggml_context* ctx, struct ggml_tensor* w) {
    if (!w) return nullptr;
    if (w->type == GGML_TYPE_F32) return w;
    struct ggml_tensor* casted = ggml_cast(ctx, w, GGML_TYPE_F32);
    return ggml_cont(ctx, casted);
}

static struct ggml_tensor* ggml_mul_mat_f32(struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b) {
    struct ggml_tensor* a_f32 = force_w_f32(ctx, a);
    struct ggml_tensor* b_f32 = (b->type == GGML_TYPE_F32) ? b : ggml_cast(ctx, b, GGML_TYPE_F32);
    
    if (a_f32->ne[0] != b_f32->ne[0]) {
        std::cerr << "\n[CRITICAL ERROR] Tensor dimension mismatch in CNHuBERT!" << std::endl;
        std::cerr << "  Tensor A (Weight): name=" << (a->name[0] ? a->name : "unnamed")
                  << ", type=" << a->type << " (F32 cast=" << a_f32->type << ")"
                  << ", shape=[" << a_f32->ne[0] << ", " << a_f32->ne[1] << ", " << a_f32->ne[2] << ", " << a_f32->ne[3] << "]"
                  << ", strides=[" << a_f32->nb[0] << ", " << a_f32->nb[1] << ", " << a_f32->nb[2] << ", " << a_f32->nb[3] << "]" << std::endl;
        std::cerr << "  Tensor B (Data):   name=" << (b->name[0] ? b->name : "unnamed")
                  << ", type=" << b->type << " (F32 cast=" << b_f32->type << ")"
                  << ", shape=[" << b_f32->ne[0] << ", " << b_f32->ne[1] << ", " << b_f32->ne[2] << ", " << b_f32->ne[3] << "]"
                  << ", strides=[" << b_f32->nb[0] << ", " << b_f32->nb[1] << ", " << b_f32->nb[2] << ", " << b_f32->nb[3] << "]" << std::endl;
        GGML_ASSERT(false && "Tensor dimension mismatch in CNHuBERT");
    }
    
    struct ggml_tensor* result = ggml_mul_mat(ctx, a_f32, b_f32);
    ggml_mul_mat_set_prec(result, GGML_PREC_F32);
    return result;
}

static void dump_tensor_f32_if_requested(
    const char * env_name,
    const char * suffix,
    struct ggml_tensor * t) {
    const char * base = std::getenv(env_name);
    if (!base || !t) {
        return;
    }

    const int64_t n = ggml_nelements(t);
    std::vector<float> data(n);
    ggml_backend_tensor_get(t, data.data(), 0, n * sizeof(float));

    std::string path = std::string(base) + "_" + suffix + ".f32";
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return;
    }
    file.write(reinterpret_cast<const char *>(data.data()), (std::streamsize)(n * sizeof(float)));
}

bool HubertModel::load(const std::string& path, ggml_backend_t backend) {
    if (!load_gguf_model(path, *this, backend)) {
        return false;
    }
    
    // Retrieve weight_g and weight_v to pre-compute the folded positional convolution weight
    struct ggml_tensor* g = get_tensor("encoder.pos_conv_embed.conv.weight_g");
    struct ggml_tensor* v = get_tensor("encoder.pos_conv_embed.conv.weight_v");
    
    if (g && v) {
        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[CNHuBERT] Pre-computing folded positional convolution weight normalization..." << std::endl;
        
        // Retrieve g data from backend
        std::vector<uint8_t> g_bytes(ggml_nbytes(g));
        ggml_backend_tensor_get(g, g_bytes.data(), 0, g_bytes.size());
        
        const int64_t g_size = ggml_nelements(g);
        std::vector<float> g_float(g_size);
        if (g->type == GGML_TYPE_F16) {
            const ggml_fp16_t* g_ptr = (const ggml_fp16_t*)g_bytes.data();
            for (int64_t i = 0; i < g_size; ++i) {
                g_float[i] = ggml_fp16_to_fp32(g_ptr[i]);
            }
        } else {
            const float* g_ptr = (const float*)g_bytes.data();
            std::copy(g_ptr, g_ptr + g_size, g_float.begin());
        }
        
        // Retrieve v data from backend
        std::vector<uint8_t> v_bytes(ggml_nbytes(v));
        ggml_backend_tensor_get(v, v_bytes.data(), 0, v_bytes.size());
        
        const int64_t kernel_width = v->ne[0];
        const int64_t in_channels_per_group = v->ne[1];
        const int64_t out_channels = v->ne[2];
        const int64_t v_size = ggml_nelements(v);
        std::vector<float> v_float(v_size);
        if (v->type == GGML_TYPE_F16) {
            const ggml_fp16_t* v_ptr = (const ggml_fp16_t*)v_bytes.data();
            for (int64_t i = 0; i < v_size; ++i) {
                v_float[i] = ggml_fp16_to_fp32(v_ptr[i]);
            }
        } else {
            const float* v_ptr = (const float*)v_bytes.data();
            std::copy(v_ptr, v_ptr + v_size, v_float.begin());
        }
        
        // Pre-compute folded weight in FP16 format
        pos_conv_w_data.resize(v_size * sizeof(ggml_fp16_t));
        ggml_fp16_t* dest_ptr = (ggml_fp16_t*)pos_conv_w_data.data();

        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[CNHuBERT] Pos conv shapes: g=[" << g->ne[0] << ", " << g->ne[1] << ", "
                  << g->ne[2] << "], v=[" << kernel_width << ", " << in_channels_per_group
                  << ", " << out_channels << "]" << std::endl;

        // Preferred path: PyTorch weight_norm on Conv1d uses one g scalar per output channel.
        if (g_size == out_channels) {
            for (int64_t oc = 0; oc < out_channels; ++oc) {
                double sum_sq = 0.0;
                for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
                    for (int64_t k = 0; k < kernel_width; ++k) {
                        const int64_t idx = oc * (in_channels_per_group * kernel_width) + ic * kernel_width + k;
                        const float val = v_float[idx];
                        sum_sq += val * val;
                    }
                }
                const float norm = (float)std::sqrt(sum_sq);
                const float scale = g_float[oc] / (norm + 1e-12f);
                for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
                    for (int64_t k = 0; k < kernel_width; ++k) {
                        const int64_t idx = oc * (in_channels_per_group * kernel_width) + ic * kernel_width + k;
                        dest_ptr[idx] = ggml_fp32_to_fp16(v_float[idx] * scale);
                    }
                }
            }
        } else if (g_size == kernel_width) {
            // Legacy fallback for unexpected exports that store g per kernel position.
            std::cerr << "[CNHuBERT] Warning: unexpected pos conv g size matches kernel width; using fallback folding." << std::endl;
            for (int64_t k = 0; k < kernel_width; ++k) {
                double sum_sq = 0.0;
                for (int64_t oc = 0; oc < out_channels; ++oc) {
                    for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
                        const int64_t idx = oc * (in_channels_per_group * kernel_width) + ic * kernel_width + k;
                        const float val = v_float[idx];
                        sum_sq += val * val;
                    }
                }
                const float norm = (float)std::sqrt(sum_sq);
                const float scale = g_float[k] / (norm + 1e-12f);
                for (int64_t oc = 0; oc < out_channels; ++oc) {
                    for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
                        const int64_t idx = oc * (in_channels_per_group * kernel_width) + ic * kernel_width + k;
                        dest_ptr[idx] = ggml_fp32_to_fp16(v_float[idx] * scale);
                    }
                }
            }
        } else {
            std::cerr << "[CNHuBERT] Warning: unsupported pos conv weight_g shape; copying unfused weight_v as fallback." << std::endl;
            for (int64_t i = 0; i < v_size; ++i) {
                dest_ptr[i] = ggml_fp32_to_fp16(v_float[i]);
            }
        }
        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[CNHuBERT] Pre-computation completed successfully. Folded positional weights cached.\n";
        
        // Allocate pos_conv_w on the backend
        struct ggml_init_params custom_params = {
            /* .mem_size   = */ 1 * 1024 * 1024,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true
        };
        custom_ctx = ggml_init(custom_params);
        pos_conv_w = ggml_new_tensor_3d(custom_ctx, GGML_TYPE_F16, kernel_width, in_channels_per_group, out_channels);
        custom_buffer = ggml_backend_alloc_ctx_tensors(custom_ctx, backend);
        if (custom_buffer) {
            ggml_backend_tensor_set(pos_conv_w, pos_conv_w_data.data(), 0, pos_conv_w_data.size());
            if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[CNHuBERT] Positional weights allocated on backend successfully.\n";
        } else {
            std::cerr << "[CNHuBERT] Failed to allocate positional weights on backend!\n";
        }
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

    // 1. Retrieve all required convolution and projection tensors
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

    {
        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[CNHuBERT Debug] w0 shape: [" << w0->ne[0] << ", " << w0->ne[1] << ", " << w0->ne[2] << ", " << w0->ne[3] << "]"
                  << " type=" << w0->type << std::endl;
        const int64_t sample_n = std::min<int64_t>(16, ggml_nelements(w0));
        if (sample_n > 0) {
            std::vector<uint8_t> w0_bytes(sample_n * sizeof(ggml_fp16_t));
            ggml_backend_tensor_get(w0, w0_bytes.data(), 0, w0_bytes.size());
            if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[CNHuBERT Debug] w0 sample:";
            if (w0->type == GGML_TYPE_F16) {
                const ggml_fp16_t* p = reinterpret_cast<const ggml_fp16_t*>(w0_bytes.data());
                for (int64_t i = 0; i < sample_n; ++i) {
                    std::cout << " " << ggml_fp16_to_fp32(p[i]);
                }
            } else {
                const float* p = reinterpret_cast<const float*>(w0_bytes.data());
                for (int64_t i = 0; i < sample_n; ++i) {
                    std::cout << " " << p[i];
                }
            }
            std::cout << std::endl;
        }
    }
    
    // Create input audio tensor in ctx_hubert
    struct ggml_tensor* input_audio_tensor = ggml_new_tensor_1d(ctx_hubert, GGML_TYPE_F32, audio_len);
    struct ggml_tensor* cnn_conv0_dbg = nullptr;
    struct ggml_tensor* cnn_conv0_ln_dbg = nullptr;
    
    // 2. CNN Feature Extractor (7 Conv1D layers)
    struct ggml_tensor* x = input_audio_tensor;
    
    // Layer 0: Conv1D (kernel=10, stride=5, no-padding)
    x = ggml_conv_1d_hubert(ctx_hubert, w0, x, 5, 0, 1);
    cnn_conv0_dbg = ggml_cont(ctx_hubert, x);
    
    // Layer 0 GroupNorm (groups=512, channels=512): normalize along the time dimension (ne0 = seq_len_0)
    int seq_len_0 = (int)x->ne[0];
    x = ggml_reshape_2d(ctx_hubert, x, seq_len_0, 512);
    x = ggml_cont(ctx_hubert, ggml_norm(ctx_hubert, x, 1e-5f));
    struct ggml_tensor* ln0_w_reshaped = ggml_reshape_2d(ctx_hubert, ln0_w, 1, 512);
    struct ggml_tensor* ln0_b_reshaped = ggml_reshape_2d(ctx_hubert, ln0_b, 1, 512);
    struct ggml_tensor* x_ln0 = ggml_mul(ctx_hubert, x, ln0_w_reshaped);
    x_ln0 = ggml_cont(ctx_hubert, x_ln0);
    x = ggml_add(ctx_hubert, x_ln0, ln0_b_reshaped);
    x = ggml_cont(ctx_hubert, x);
    cnn_conv0_ln_dbg = ggml_cont(ctx_hubert, x);
    x = ggml_gelu_erf(ctx_hubert, x);
    // CUDA IM2COL requires input to be a standard 3D tensor [seq_len, channels, 1] with correct multichan strides (nb[1] mapping)
    x = ggml_reshape_3d(ctx_hubert, x, seq_len_0, 512, 1);
    x = ggml_cont(ctx_hubert, x);
    
    // Layer 1 to 6: Conv1D + GELU
    x = ggml_conv_1d_hubert(ctx_hubert, w1, x, 2, 0, 1); x = ggml_gelu_erf(ctx_hubert, x);
    x = ggml_conv_1d_hubert(ctx_hubert, w2, x, 2, 0, 1); x = ggml_gelu_erf(ctx_hubert, x);
    x = ggml_conv_1d_hubert(ctx_hubert, w3, x, 2, 0, 1); x = ggml_gelu_erf(ctx_hubert, x);
    x = ggml_conv_1d_hubert(ctx_hubert, w4, x, 2, 0, 1); x = ggml_gelu_erf(ctx_hubert, x);
    x = ggml_conv_1d_hubert(ctx_hubert, w5, x, 2, 0, 1); x = ggml_gelu_erf(ctx_hubert, x);
    x = ggml_conv_1d_hubert(ctx_hubert, w6, x, 2, 0, 1); x = ggml_gelu_erf(ctx_hubert, x);
    struct ggml_tensor* feature_extractor_dbg = ggml_cont(ctx_hubert, x);
    
    int seq_len = (int)x->ne[0];
    
    // 3. Feature Projection (512 -> 768)
    struct ggml_tensor* x_proj = ggml_permute(ctx_hubert, x, 1, 0, 2, 3);
    x_proj = ggml_cont(ctx_hubert, x_proj);
    x_proj = ggml_cont(ctx_hubert, ggml_norm(ctx_hubert, x_proj, 1e-5f));
    struct ggml_tensor* x_proj_ln = ggml_mul(ctx_hubert, x_proj, proj_ln_w);
    x_proj_ln = ggml_cont(ctx_hubert, x_proj_ln);
    x_proj = ggml_add(ctx_hubert, x_proj_ln, proj_ln_b);
    x_proj = ggml_cont(ctx_hubert, x_proj);
    struct ggml_tensor* x_proj_linear = ggml_mul_mat_f32(ctx_hubert, proj_w, x_proj);
    x_proj_linear = ggml_cont(ctx_hubert, x_proj_linear);
    x_proj = ggml_add(ctx_hubert, x_proj_linear, proj_b);
    x_proj = ggml_cont(ctx_hubert, x_proj);
    
    // 4. Positional Convolution Embedding (Grouped Conv1D with groups=16, kernel=128)
    // x_proj is currently [768, seq_len]. Positional Conv1d expects [seq_len, 768].
    // A raw reshape would reinterpret the buffer with the wrong stride pattern, so transpose first.
    struct ggml_tensor* x_pos_input_2d = ggml_cont(ctx_hubert, ggml_transpose(ctx_hubert, x_proj));
    struct ggml_tensor* x_pos_input = ggml_reshape_3d(ctx_hubert, x_pos_input_2d, seq_len, 768, 1);
    struct ggml_tensor* pos_conv_w_tensor = pos_conv_w;
    
    std::vector<struct ggml_tensor*> slices(16);
    for (int g = 0; g < 16; ++g) {
        struct ggml_tensor* x_g = ggml_view_3d(ctx_hubert, x_pos_input, seq_len, 48, 1, x_pos_input->nb[1], x_pos_input->nb[2], g * 48 * x_pos_input->nb[1]);
        struct ggml_tensor* w_g = ggml_view_3d(ctx_hubert, pos_conv_w_tensor, 128, 48, 48, pos_conv_w_tensor->nb[1], pos_conv_w_tensor->nb[2], g * 48 * pos_conv_w_tensor->nb[2]);
        struct ggml_tensor* conv_out = ggml_conv_1d_hubert(ctx_hubert, w_g, x_g, 1, 64, 1);
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
    hidden_states = ggml_cont(ctx_hubert, ggml_norm(ctx_hubert, hidden_states, 1e-5f));
    struct ggml_tensor* hidden_states_ln = ggml_mul(ctx_hubert, hidden_states, encoder_ln_w);
    hidden_states_ln = ggml_cont(ctx_hubert, hidden_states_ln);
    hidden_states = ggml_add(ctx_hubert, hidden_states_ln, encoder_ln_b);
    hidden_states = ggml_cont(ctx_hubert, hidden_states);
    struct ggml_tensor* x_normalized = hidden_states;
    
    // 5. Construct 12 Transformer Encoder layers
    int num_layers = 12;
    int n_heads = 12;
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
        
        // This HuBERT config uses the standard post-LN encoder layer, not the stable pre-LN variant.
        struct ggml_tensor* Q_linear = ggml_mul_mat_f32(ctx_hubert, qw, hidden_states);
        Q_linear = ggml_cont(ctx_hubert, Q_linear);
        struct ggml_tensor* Q = ggml_add(ctx_hubert, Q_linear, qb);
        struct ggml_tensor* K_linear = ggml_mul_mat_f32(ctx_hubert, kw, hidden_states);
        K_linear = ggml_cont(ctx_hubert, K_linear);
        struct ggml_tensor* K = ggml_add(ctx_hubert, K_linear, kb);
        struct ggml_tensor* V_linear = ggml_mul_mat_f32(ctx_hubert, vw, hidden_states);
        V_linear = ggml_cont(ctx_hubert, V_linear);
        struct ggml_tensor* V = ggml_add(ctx_hubert, V_linear, vb);
        
        Q = ggml_reshape_3d(ctx_hubert, Q, head_dim, n_heads, seq_len);
        K = ggml_reshape_3d(ctx_hubert, K, head_dim, n_heads, seq_len);
        V = ggml_reshape_3d(ctx_hubert, V, head_dim, n_heads, seq_len);
        
        struct ggml_tensor* Q_perm = ggml_cont(ctx_hubert, ggml_permute(ctx_hubert, Q, 0, 2, 1, 3));
        struct ggml_tensor* K_perm = ggml_cont(ctx_hubert, ggml_permute(ctx_hubert, K, 0, 2, 1, 3));
        struct ggml_tensor* V_perm = ggml_permute(ctx_hubert, V, 1, 2, 0, 3);
        
        struct ggml_tensor* kq = ggml_mul_mat_f32(ctx_hubert, K_perm, Q_perm);
        kq = ggml_scale(ctx_hubert, kq, 1.0f / std::sqrt((float)head_dim));
        kq = ggml_soft_max(ctx_hubert, kq);
        
        struct ggml_tensor* V_cont = ggml_cont(ctx_hubert, V_perm);
        struct ggml_tensor* kqv = ggml_mul_mat_f32(ctx_hubert, V_cont, kq);
        
        if (layer == 0) {
            if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[CNHuBERT Debug] layer 0 tensor shapes:\n"
                      << "  Q: [" << Q->ne[0] << ", " << Q->ne[1] << ", " << Q->ne[2] << "]\n"
                      << "  K: [" << K->ne[0] << ", " << K->ne[1] << ", " << K->ne[2] << "]\n"
                      << "  V: [" << V->ne[0] << ", " << V->ne[1] << ", " << V->ne[2] << "]\n"
                      << "  Q_perm: [" << Q_perm->ne[0] << ", " << Q_perm->ne[1] << ", " << Q_perm->ne[2] << "]\n"
                      << "  K_perm: [" << K_perm->ne[0] << ", " << K_perm->ne[1] << ", " << K_perm->ne[2] << "]\n"
                      << "  V_perm: [" << V_perm->ne[0] << ", " << V_perm->ne[1] << ", " << V_perm->ne[2] << "]\n"
                      << "  kq: [" << kq->ne[0] << ", " << kq->ne[1] << ", " << kq->ne[2] << "]\n"
                      << "  V_cont: [" << V_cont->ne[0] << ", " << V_cont->ne[1] << ", " << V_cont->ne[2] << "]\n"
                      << "  kqv: [" << kqv->ne[0] << ", " << kqv->ne[1] << ", " << kqv->ne[2] << "]\n";
        }
        
        kqv = ggml_permute(ctx_hubert, kqv, 0, 2, 1, 3);
        kqv = ggml_cont(ctx_hubert, kqv);
        kqv = ggml_reshape_2d(ctx_hubert, kqv, 768, seq_len);
        
        struct ggml_tensor* attn_linear = ggml_mul_mat_f32(ctx_hubert, out_w, kqv);
        attn_linear = ggml_cont(ctx_hubert, attn_linear);
        struct ggml_tensor* attn_out = ggml_add(ctx_hubert, attn_linear, out_b);
        attn_out = ggml_cont(ctx_hubert, attn_out);
        struct ggml_tensor* x_attn = ggml_add(ctx_hubert, hidden_states, attn_out);
        x_attn = ggml_cont(ctx_hubert, x_attn);

        struct ggml_tensor* ln1 = ggml_cont(ctx_hubert, ggml_norm(ctx_hubert, x_attn, 1e-5f));
        struct ggml_tensor* ln1_affine = ggml_mul(ctx_hubert, ln1, ln1_w);
        ln1_affine = ggml_cont(ctx_hubert, ln1_affine);
        ln1 = ggml_add(ctx_hubert, ln1_affine, ln1_b);
        ln1 = ggml_cont(ctx_hubert, ln1);

        struct ggml_tensor* h_linear = ggml_mul_mat_f32(ctx_hubert, ffn_w1, ln1);
        h_linear = ggml_cont(ctx_hubert, h_linear);
        struct ggml_tensor* h = ggml_add(ctx_hubert, h_linear, ffn_b1);
        h = ggml_cont(ctx_hubert, h);
        h = ggml_gelu_erf(ctx_hubert, h);
        struct ggml_tensor* mlp_linear = ggml_mul_mat_f32(ctx_hubert, ffn_w2, h);
        mlp_linear = ggml_cont(ctx_hubert, mlp_linear);
        struct ggml_tensor* mlp_out = ggml_add(ctx_hubert, mlp_linear, ffn_b2);
        mlp_out = ggml_cont(ctx_hubert, mlp_out);

        struct ggml_tensor* pre_final_norm = ggml_add(ctx_hubert, ln1, mlp_out);
        pre_final_norm = ggml_cont(ctx_hubert, pre_final_norm);
        struct ggml_tensor* ln2 = ggml_cont(ctx_hubert, ggml_norm(ctx_hubert, pre_final_norm, 1e-5f));
        struct ggml_tensor* ln2_affine = ggml_mul(ctx_hubert, ln2, ln2_w);
        ln2_affine = ggml_cont(ctx_hubert, ln2_affine);
        hidden_states = ggml_add(ctx_hubert, ln2_affine, ln2_b);
        hidden_states = ggml_cont(ctx_hubert, hidden_states);
        if (layer == 0) {
            layer0_output = hidden_states;
            layer0_ln1 = ln1;
            layer0_Q = Q;
            layer0_K = K;
            layer0_V = V;
            layer0_kq = kq;
            layer0_kqv = kqv;
            layer0_attn_out = attn_out;
            layer0_x_attn = x_attn;
            layer0_pre_final_norm = pre_final_norm;
            layer0_ln2 = ln2;
            layer0_h = h;
            layer0_mlp_out = mlp_out;
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
    if (std::getenv("GPT_SOVITS_DEBUG") != nullptr) {
        std::cout << "[CNHuBERT Debug Type Checking]:\n"
                  << "  qw type: " << get_tensor("encoder.layers.0.attention.q_proj.weight")->type << "\n"
                  << "  x_normalized type: " << x_normalized->type << "\n"
                  << "  layer0_Q type: " << (layer0_Q ? layer0_Q->type : -1) << "\n"
                  << "  layer0_K type: " << (layer0_K ? layer0_K->type : -1) << "\n"
                  << "  layer0_kq type: " << (layer0_kq ? layer0_kq->type : -1) << "\n"
                  << "  layer0_kqv type: " << (layer0_kqv ? layer0_kqv->type : -1) << "\n"
                  << "  layer0_attn_out type: " << (layer0_attn_out ? layer0_attn_out->type : -1) << "\n"
                  << "  layer0_output type: " << (layer0_output ? layer0_output->type : -1) << "\n";
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
                if (t->ne[1] > 0) {
                    std::cout << "      First 10 values at channel 0: ";
                    for (int i = 0; i < std::min(10, (int)t->ne[1]); ++i) {
                        if (t->ne[0] == 768) {
                            std::cout << data[i * 768] << " ";
                        } else if (t->ne[0] == 512) {
                            std::cout << data[i * 512] << " ";
                        } else {
                            std::cout << data[i] << " ";
                        }
                    }
                    std::cout << "\n";
                }
            };
            
            std::cout << "\n[CNHuBERT C++ Intermediate Debug]:\n";
            print_tensor_info("feature_extractor (CNN)", x);
            print_tensor_info("feature_projection", x_proj);
            print_tensor_info("pos_conv_embed", pos_emb);
            print_tensor_info("encoder_layer_norm", x_normalized);
            
            std::cout << "\n--- Layer 0 Self-Attention & MLP Debug ---\n";
            print_tensor_info("layer0_ln1", layer0_ln1);
            print_tensor_info("layer0_Q", layer0_Q);
            print_tensor_info("layer0_K", layer0_K);
            print_tensor_info("layer0_V", layer0_V);
            print_tensor_info("layer0_kq", layer0_kq);
            print_tensor_info("layer0_kqv", layer0_kqv);
            print_tensor_info("layer0_attn_out", layer0_attn_out);
            print_tensor_info("layer0_ln2", layer0_ln2);
            print_tensor_info("layer0_h (GELU)", layer0_h);
            print_tensor_info("layer0_mlp_out", layer0_mlp_out);
            print_tensor_info("layer_0 output", layer0_output);
            print_tensor_info("layer_1 output", layer1_output);
            print_tensor_info("layer_5 output", layer5_output);
            print_tensor_info("cnn_conv0", cnn_conv0_dbg);
            print_tensor_info("cnn_conv0_ln", cnn_conv0_ln_dbg);
            
            std::cout << "\n--- Final Model Output ---\n";
            print_tensor_info("hidden_states (layer 11)", hidden_states);

            dump_tensor_f32_if_requested("GPT_SOVITS_HUBERT_DUMP", "layer0_attn_out", layer0_attn_out);
            dump_tensor_f32_if_requested("GPT_SOVITS_HUBERT_DUMP", "layer0_x_attn", layer0_x_attn);
            dump_tensor_f32_if_requested("GPT_SOVITS_HUBERT_DUMP", "layer0_mlp_out", layer0_mlp_out);
            dump_tensor_f32_if_requested("GPT_SOVITS_HUBERT_DUMP", "layer0_pre_final_norm", layer0_pre_final_norm);
            dump_tensor_f32_if_requested("GPT_SOVITS_HUBERT_DUMP", "layer0_ln2_pre_affine", layer0_ln2);
            dump_tensor_f32_if_requested("GPT_SOVITS_HUBERT_DUMP", "feature_projection", x_proj);
            dump_tensor_f32_if_requested("GPT_SOVITS_HUBERT_DUMP", "pos_conv_embed", pos_emb);
            dump_tensor_f32_if_requested("GPT_SOVITS_HUBERT_DUMP", "encoder_layer_norm", x_normalized);
            dump_tensor_f32_if_requested("GPT_SOVITS_HUBERT_DUMP", "layer0_ln1", layer0_ln1);
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
