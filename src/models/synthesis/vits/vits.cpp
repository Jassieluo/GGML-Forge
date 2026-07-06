#include "vits.h"
#include "ops/ops.h"
#include "nn/nn.h"
#include <cstdlib>
#include <random>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <unordered_map>
#if defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif

namespace gpt_sovits {

// Static members for flip permutation matrix
float VITSModel::flip_data[192 * 192];
bool  VITSModel::flip_data_ready = false;

void VITSModel::upload_pending_data(ggml_backend_t backend) {
    if (!flip_data_ready) {
        std::memset(flip_data, 0, sizeof(flip_data));
        for (int r = 0; r < 192; ++r) {
            flip_data[r * 192 + (191 - r)] = 1.0f;
        }
        flip_data_ready = true;
    }
    for (auto& entry : upload_entries) {
        if (GPT_SOVITS_DEBUG_ENABLED()) {
            size_t buf_size = entry.tensor->buffer ? ggml_backend_buffer_get_size(entry.tensor->buffer) : 0;
            void* base = entry.tensor->buffer ? ggml_backend_buffer_get_base(entry.tensor->buffer) : nullptr;
            std::cout << "[upload_pending_data] Setting tensor " << (entry.tensor->name ? entry.tensor->name : "NULL")
                      << " (" << entry.tensor << "), buffer: " << entry.tensor->buffer
                      << " (size: " << buf_size << ", base: " << base << ")"
                      << ", data: " << entry.tensor->data
                      << ", src_data: " << (void*)entry.data.data()
                      << ", size: " << entry.data.size() << std::endl;
        }
        if (entry.tensor->data == nullptr) {
            if (GPT_SOVITS_DEBUG_ENABLED()) {
                std::cout << "[upload_pending_data] WARNING: tensor data is NULL! Skipping upload to avoid crash." << std::endl;
            }
            continue;
        }
        ggml_backend_tensor_set(entry.tensor, entry.data.data(), 0, entry.data.size());
    }
    upload_entries.clear();
}

void VITSModel::on_prepare_tensor(struct ggml_tensor* tensor, const std::string& name) {
    // No-op: weights are expected to be in standard GGML layout [kernel, in, out] in GGUF
}

bool VITSModel::on_upload_tensor(
    struct ggml_tensor* t_backend,
    const void* raw_data,
    size_t size,
    enum ggml_type type,
    const std::string& name
) {
    return false; // No-op: standard copy by generic loader
}

// Helper: cast weights to FP32 for compute (CUDA FP16 gemm not supported on this GPU)
struct ggml_tensor* force_w_f32(struct ggml_context* ctx, struct ggml_tensor* w) {
    if (!w) return nullptr;
    if (w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_F16) {
        // It is a quantized type, return it as-is to use optimized CPU kernels!
        return w;
    }
    if (w->type == GGML_TYPE_F32) return w;
    struct ggml_tensor* casted = ggml_cast(ctx, w, GGML_TYPE_F32);
    return ggml_cont(ctx, casted);
}

static struct ggml_context* vits_custom_ctx = nullptr;
static ggml_backend_buffer_t vits_custom_buf = nullptr;

bool VITSModel::load(const std::string& path, ggml_backend_t backend) {
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS] Loading VITS GGUF model: " << path << std::endl;
    if (!load_gguf_model(path, *this, backend)) {
        std::cerr << "[VITS] Failed to load GGUF VITS model!" << std::endl;
        return false;
    }
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS] Loaded VITS successfully. Pre-computing Weight Normalization..." << std::endl;
    
    struct ggml_init_params custom_params = {
        /* .mem_size   = */ 16 * 1024 * 1024, // 16MB metadata pool, extremely safe!
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true
    };
    vits_custom_ctx = ggml_init(custom_params);
    if (!vits_custom_ctx) {
        std::cerr << "[VITS load] Error: Failed to initialize vits_custom_ctx!" << std::endl;
        return false;
    }

    // Allocate input placeholder tensors
    phone_ids.tensor = ggml_new_tensor_1d(vits_custom_ctx, GGML_TYPE_I32, 512);
    ggml_set_name(phone_ids.tensor, "input_phone_ids");
    phone_lengths.tensor = ggml_new_tensor_1d(vits_custom_ctx, GGML_TYPE_I32, 1);
    ggml_set_name(phone_lengths.tensor, "input_phone_lengths");
    word2ph.tensor = ggml_new_tensor_1d(vits_custom_ctx, GGML_TYPE_I32, 512);
    ggml_set_name(word2ph.tensor, "input_word2ph");
    bert_features.tensor = ggml_new_tensor_2d(vits_custom_ctx, GGML_TYPE_F32, 1024, 512);
    ggml_set_name(bert_features.tensor, "input_bert_features");
    prompt_semantics.tensor = ggml_new_tensor_1d(vits_custom_ctx, GGML_TYPE_I32, 512);
    ggml_set_name(prompt_semantics.tensor, "input_prompt_semantics");
    int64_t ge_dim = 512;
    struct ggml_tensor* prelu_w = get_tensor("prelu.weight");
    if (prelu_w) {
        ge_dim = prelu_w->ne[0]; // 1024 for V2Pro
    }
    refer_audio.tensor = ggml_new_tensor_2d(vits_custom_ctx, GGML_TYPE_F32, ge_dim, 1);
    ggml_set_name(refer_audio.tensor, "input_refer_audio");
    prompt_mel.tensor = ggml_new_tensor_2d(vits_custom_ctx, GGML_TYPE_F32, 100, 1024);
    ggml_set_name(prompt_mel.tensor, "input_prompt_mel");
    
    // Pre-compute dilated convolution weights (as FP16 for CUDA, FP32 for CPU/SYCL)
    std::vector<std::pair<std::string, std::vector<ggml_fp16_t>>> dilated_fp16_data_list;
    std::vector<struct ggml_tensor*> dilated_tensors_list;
    
    for (int b = 0; b < 15; ++b) {
        for (int l = 1; l <= 2; ++l) {
            int dilation = (l == 1) ? 3 : 5;
            
            std::string prefix = "dec.resblocks." + std::to_string(b) + ".convs1." + std::to_string(l);
            struct ggml_tensor* old_w = get_tensor(prefix + ".weight");
            if (!old_w) continue;
            
            int64_t kw = old_w->ne[0]; // 3, 7, or 11
            int64_t ic = old_w->ne[1];
            int64_t oc = old_w->ne[2];
            int64_t new_kw = (kw - 1) * dilation + 1; // Adaptively calculate correct dilated kernel size!
            int64_t w_elems = ggml_nelements(old_w);
            
            std::vector<float> w_host;
            if (!dequantize_tensor_to_f32(old_w, w_host, backend)) {
                std::cerr << "[VITS] Failed to read/dequantize weight data for " << prefix << std::endl;
                return false;
            }
            
            int64_t new_w_elems = new_kw * ic * oc;
            std::vector<float> w_dilated_host(new_w_elems, 0.0f);
            for (int64_t o = 0; o < oc; ++o) {
                for (int64_t i = 0; i < ic; ++i) {
                    for (int64_t k = 0; k < kw; ++k) {
                        int64_t old_idx = o * (ic * kw) + i * kw + k;
                        int64_t new_idx = o * (ic * new_kw) + i * new_kw + (k * dilation);
                        w_dilated_host[new_idx] = w_host[old_idx];
                    }
                }
            }
            
            std::vector<ggml_fp16_t> w_dilated_fp16(new_w_elems);
            for (int64_t i = 0; i < new_w_elems; ++i) {
                w_dilated_fp16[i] = ggml_fp32_to_fp16(w_dilated_host[i]);
            }
            struct ggml_tensor* new_w = ggml_new_tensor_3d(vits_custom_ctx, GGML_TYPE_F16, new_kw, ic, oc);
            dilated_tensors_list.push_back(new_w);
            dilated_fp16_data_list.push_back({prefix + ".weight_dilated", w_dilated_fp16});
        }
    }

    // Pre-compute repeated upsample/downsample filters for V3 alias-free activations
    std::vector<std::pair<std::string, std::vector<ggml_fp16_t>>> filter_fp16_data_list;
    std::vector<struct ggml_tensor*> filter_tensors_list;

    if (version == 3) {
        const std::vector<int> upsample_channels = {768, 384, 192, 96, 48, 24};
        for (int i = 0; i < 6; ++i) {
            int channels = upsample_channels[i];
            for (int j = 0; j < 3; ++j) {
                int block_idx = i * 3 + j;
                for (int act_idx = 0; act_idx < 6; ++act_idx) {
                    std::string act_prefix = "dec.resblocks." + std::to_string(block_idx) + ".activations." + std::to_string(act_idx);
                    
                    // Upsample filter
                    struct ggml_tensor* up_filter = get_tensor(act_prefix + ".upsample.filter");
                    if (up_filter) {
                        std::vector<float> up_host;
                        if (dequantize_tensor_to_f32(up_filter, up_host, backend)) {
                            std::vector<ggml_fp16_t> up_rep_fp16(12 * channels);
                            for (int c = 0; c < channels; ++c) {
                                for (int k = 0; k < 12; ++k) {
                                    up_rep_fp16[c * 12 + k] = ggml_fp32_to_fp16(up_host[k]);
                                }
                            }
                            struct ggml_tensor* new_up = ggml_new_tensor_3d(vits_custom_ctx, GGML_TYPE_F16, 12, 1, channels);
                            filter_tensors_list.push_back(new_up);
                            filter_fp16_data_list.push_back({act_prefix + ".upsample.filter_repeated", up_rep_fp16});
                        }
                    }

                    // Downsample filter
                    struct ggml_tensor* down_filter = get_tensor(act_prefix + ".downsample.lowpass.filter");
                    if (down_filter) {
                        std::vector<float> down_host;
                        if (dequantize_tensor_to_f32(down_filter, down_host, backend)) {
                            std::vector<ggml_fp16_t> down_rep_fp16(12 * channels);
                            for (int c = 0; c < channels; ++c) {
                                for (int k = 0; k < 12; ++k) {
                                    down_rep_fp16[c * 12 + k] = ggml_fp32_to_fp16(down_host[k]);
                                }
                            }
                            struct ggml_tensor* new_down = ggml_new_tensor_3d(vits_custom_ctx, GGML_TYPE_F16, 12, 1, channels);
                            filter_tensors_list.push_back(new_down);
                            filter_fp16_data_list.push_back({act_prefix + ".downsample.lowpass.filter_repeated", down_rep_fp16});
                        }
                    }
                }
            }
        }
    }
    
    // Allocate all custom tensors on the backend
    vits_custom_buf = ggml_backend_alloc_ctx_tensors(vits_custom_ctx, backend);
    if (!vits_custom_buf) {
        std::cerr << "[VITS load] Error: Failed to allocate vits_custom_buf!" << std::endl;
        return false;
    }
    
    // Upload dilated weights
    for (size_t i = 0; i < dilated_tensors_list.size(); ++i) {
        struct ggml_tensor* nt = dilated_tensors_list[i];
        const auto& name_and_data = dilated_fp16_data_list[i];
        ggml_backend_tensor_set(nt, name_and_data.second.data(), 0, name_and_data.second.size() * sizeof(ggml_fp16_t));
        tensors[name_and_data.first] = nt;
    }

    // Upload repeated filters
    for (size_t i = 0; i < filter_tensors_list.size(); ++i) {
        struct ggml_tensor* nt = filter_tensors_list[i];
        const auto& name_and_data = filter_fp16_data_list[i];
        ggml_backend_tensor_set(nt, name_and_data.second.data(), 0, name_and_data.second.size() * sizeof(ggml_fp16_t));
        tensors[name_and_data.first] = nt;
    }
 
    ggml_backend_synchronize(backend);
    return true;
}

std::unique_ptr<VITSModel> VITSModel::create(const std::string& path) {
    struct ggml_context* ggml_ctx_backend = nullptr;
    struct gguf_init_params params_backend = {
        /* .no_alloc = */ true,
        /* .ctx      = */ &ggml_ctx_backend
    };
    struct gguf_context* ctx_gguf = gguf_init_from_file(path.c_str(), params_backend);
    if (!ctx_gguf) {
        return nullptr;
    }

    int ver = 0;
    int kid_ver = gguf_find_key(ctx_gguf, "gpt_sovits.version");
    if (kid_ver != -1) {
        enum gguf_type type = gguf_get_kv_type(ctx_gguf, kid_ver);
        if (type == GGUF_TYPE_STRING) {
            std::string ver_str = gguf_get_val_str(ctx_gguf, kid_ver);
            if (ver_str.find("v3") != std::string::npos || ver_str == "3") {
                ver = 3;
            } else if (ver_str.find("v4") != std::string::npos || ver_str == "4") {
                ver = 4;
            } else if (ver_str.find("v1") != std::string::npos || ver_str == "1") {
                ver = 1;
            } else if (ver_str.find("v2") != std::string::npos || ver_str == "2") {
                ver = 2;
            }
        } else if (type == GGUF_TYPE_UINT32) {
            ver = (int)gguf_get_val_u32(ctx_gguf, kid_ver);
        } else if (type == GGUF_TYPE_INT32) {
            ver = (int)gguf_get_val_i32(ctx_gguf, kid_ver);
        }
    }

    if (ver == 0) {
        int n_tensors = gguf_get_n_tensors(ctx_gguf);
        for (int i = 0; i < n_tensors; ++i) {
            std::string name = gguf_get_tensor_name(ctx_gguf, i);
            if (name == "enc_p.text_embedding.weight") {
                struct ggml_tensor* text_emb_w = ggml_get_tensor(ggml_ctx_backend, "enc_p.text_embedding.weight");
                if (text_emb_w) {
                    int vocab_size = (int)text_emb_w->ne[1];
                    if (vocab_size == 732) {
                        bool has_flows = false;
                        for (int j = 0; j < n_tensors; ++j) {
                            std::string tname = gguf_get_tensor_name(ctx_gguf, j);
                            if (tname.find("flow.flows") != std::string::npos) {
                                has_flows = true;
                                break;
                            }
                        }
                        if (has_flows) {
                            ver = 2; // V2Pro (Classic VITS)
                        } else {
                            ver = 3; // V3 (CFM)
                        }
                    } else {
                        ver = 2;
                    }
                }
                break;
            }
        }
    }

    gguf_free(ctx_gguf);
    if (ggml_ctx_backend) {
        ggml_free(ggml_ctx_backend);
    }

    if (ver == 3 || ver == 4) {
        return std::make_unique<VITSModelCFM>();
    } else {
        return std::make_unique<VITSModelClassic>();
    }
}

// Thread-local backend state to decide between CPU and cuDNN kernels dynamically
thread_local ggml_backend_t current_vits_backend = nullptr;

struct conv_1d_direct_params {
    int stride;
    int padding;
    int dilation;
    std::vector<float> w_f32_buf;
};
static thread_local std::deque<conv_1d_direct_params> g_conv_1d_direct_params_pool;

void clear_conv_1d_params_pool() {
    g_conv_1d_direct_params_pool.clear();
}

static void ggml_conv_1d_direct_cpu_callback(
    struct ggml_tensor* dst,
    const struct ggml_tensor* x_transposed,
    const struct ggml_tensor* w,
    int ith,
    int nth,
    void* userdata
) {
    conv_1d_direct_params* params = (conv_1d_direct_params*)userdata;
    int stride = params->stride;
    int padding = params->padding;
    int dilation = params->dilation;

    int64_t seq_len = x_transposed->ne[0];
    int64_t in_channels = x_transposed->ne[1];

    int64_t kernel_size = w->ne[0];
    int64_t out_channels = w->ne[2];

    int64_t out_seq_len = dst->ne[0];

    int64_t oc_start = (out_channels * ith) / nth;
    int64_t oc_end = (out_channels * (ith + 1)) / nth;

    const float* x_data = (const float*)x_transposed->data;
    float* dst_data = (float*)dst->data;

    bool w_is_f16 = (w->type == GGML_TYPE_F16);
    const float* w_ptr = w_is_f16 ? params->w_f32_buf.data() : (const float*)w->data;

    for (int64_t oc = oc_start; oc < oc_end; ++oc) {
        float* __restrict dst_oc = dst_data + oc * out_seq_len;
        std::memset(dst_oc, 0, out_seq_len * sizeof(float));

        for (int64_t ic = 0; ic < in_channels; ++ic) {
            const float* __restrict x_ic = x_data + ic * seq_len;
            int64_t w_base = oc * (in_channels * kernel_size) + ic * kernel_size;

            for (int64_t k = 0; k < kernel_size; ++k) {
                float w_val = w_ptr[w_base + k];
                if (w_val == 0.0f) continue;

                int64_t k_offset = k * dilation - padding;

                if (stride == 1) {
                    int64_t t_start = std::max((int64_t)0, -k_offset);
                    int64_t t_end = std::min(out_seq_len, seq_len - k_offset);

                    if (t_start < t_end) {
                        const float* __restrict x_ic_start = x_ic + t_start + k_offset;
                        float* __restrict dst_oc_start = dst_oc + t_start;
                        int64_t len = t_end - t_start;

#if defined(__AVX2__)
                        __m256 w_vec = _mm256_set1_ps(w_val);
                        int64_t t_inner = 0;
                        
                        for (; t_inner <= len - 32; t_inner += 32) {
                            __m256 x0 = _mm256_loadu_ps(x_ic_start + t_inner);
                            __m256 x1 = _mm256_loadu_ps(x_ic_start + t_inner + 8);
                            __m256 x2 = _mm256_loadu_ps(x_ic_start + t_inner + 16);
                            __m256 x3 = _mm256_loadu_ps(x_ic_start + t_inner + 24);

                            __m256 d0 = _mm256_loadu_ps(dst_oc_start + t_inner);
                            __m256 d1 = _mm256_loadu_ps(dst_oc_start + t_inner + 8);
                            __m256 d2 = _mm256_loadu_ps(dst_oc_start + t_inner + 16);
                            __m256 d3 = _mm256_loadu_ps(dst_oc_start + t_inner + 24);

                            d0 = _mm256_fmadd_ps(w_vec, x0, d0);
                            d1 = _mm256_fmadd_ps(w_vec, x1, d1);
                            d2 = _mm256_fmadd_ps(w_vec, x2, d2);
                            d3 = _mm256_fmadd_ps(w_vec, x3, d3);

                            _mm256_storeu_ps(dst_oc_start + t_inner, d0);
                            _mm256_storeu_ps(dst_oc_start + t_inner + 8, d1);
                            _mm256_storeu_ps(dst_oc_start + t_inner + 16, d2);
                            _mm256_storeu_ps(dst_oc_start + t_inner + 24, d3);
                        }

                        for (; t_inner <= len - 8; t_inner += 8) {
                            __m256 x_vec = _mm256_loadu_ps(x_ic_start + t_inner);
                            __m256 d_vec = _mm256_loadu_ps(dst_oc_start + t_inner);
                            d_vec = _mm256_fmadd_ps(w_vec, x_vec, d_vec);
                            _mm256_storeu_ps(dst_oc_start + t_inner, d_vec);
                        }

                        for (; t_inner < len; ++t_inner) {
                            dst_oc_start[t_inner] += w_val * x_ic_start[t_inner];
                        }
#else
                        for (int64_t t_inner = 0; t_inner < len; ++t_inner) {
                            dst_oc_start[t_inner] += w_val * x_ic_start[t_inner];
                        }
#endif
                    }
                } else {
                    for (int64_t t = 0; t < out_seq_len; ++t) {
                        int64_t in_t = t * stride + k_offset;
                        if (in_t >= 0 && in_t < seq_len) {
                            dst_oc[t] += w_val * x_ic[in_t];
                        }
                    }
                }
            }
        }
    }
}

static struct ggml_tensor* ggml_conv_1d_direct_cpu(
    struct ggml_context* ctx,
    struct ggml_tensor* w,
    struct ggml_tensor* x_transposed,
    int stride,
    int padding,
    int dilation
) {
    int64_t seq_len = x_transposed->ne[0];
    int64_t in_channels = x_transposed->ne[1];
    int64_t kernel_size = w->ne[0];
    int64_t out_channels = w->ne[2];

    int64_t out_seq_len = (seq_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    g_conv_1d_direct_params_pool.push_back({stride, padding, dilation, {}});
    conv_1d_direct_params& params = g_conv_1d_direct_params_pool.back();

    int64_t w_elems = ggml_nelements(w);
    bool w_is_f16 = (w->type == GGML_TYPE_F16);
    if (w_is_f16) {
        params.w_f32_buf.resize(w_elems);
        const ggml_fp16_t* w_f16 = (const ggml_fp16_t*)w->data;
        for (int64_t i = 0; i < w_elems; ++i) {
            params.w_f32_buf[i] = ggml_fp16_to_fp32(w_f16[i]);
        }
    }

    void* userdata = &params;

    struct ggml_tensor* dst = ggml_map_custom2(ctx, x_transposed, w, ggml_conv_1d_direct_cpu_callback, GGML_N_TASKS_MAX, userdata);
    
    dst->ne[0] = out_seq_len;
    dst->ne[1] = out_channels;
    dst->ne[2] = 1;
    dst->ne[3] = 1;

    dst->nb[0] = ggml_type_size(dst->type);
    dst->nb[1] = dst->nb[0] * dst->ne[0];
    dst->nb[2] = dst->nb[1] * dst->ne[1];
    dst->nb[3] = dst->nb[2] * dst->ne[2];

    return dst;
}

static struct ggml_tensor* ggml_conv_1d_im2col_f32(
    struct ggml_context* ctx,
    struct ggml_tensor* w,
    struct ggml_tensor* x,
    int stride,
    int padding,
    int dilation
) {
    struct ggml_tensor* w_f32 = force_w_f32(ctx, w);
    
    if (w->ne[0] == 1 && stride == 1 && dilation == 1 && padding == 0) {
        struct ggml_tensor* w_reshaped = ggml_reshape_2d(ctx, w_f32, w_f32->ne[1], w_f32->ne[2]);
        struct ggml_tensor* x_t = ggml_cont(ctx, ggml_transpose(ctx, x));
        struct ggml_tensor* result = ggml_mul_mat(ctx, x_t, w_reshaped);
        return ggml_cont(ctx, result);
    }

    struct ggml_tensor* im2col = ggml_im2col(ctx, w_f32, x, stride, 0, padding, 0, dilation, 0, false, GGML_TYPE_F32);
    struct ggml_tensor* im2col_reshaped = ggml_reshape_2d(ctx, im2col, im2col->ne[0], (im2col->ne[2] * im2col->ne[1]));
    struct ggml_tensor* w_reshaped = ggml_reshape_2d(ctx, w_f32, (w_f32->ne[0] * w_f32->ne[1]), w_f32->ne[2]);

    bool is_sycl = false;
    if (current_vits_backend) {
        const char * bname = ggml_backend_name(current_vits_backend);
        if (bname && strncmp(bname, "SYCL", 4) == 0) {
            is_sycl = true;
        }
    }

    if (is_sycl) {
        if (!ggml_is_contiguous(im2col_reshaped)) {
            im2col_reshaped = ggml_cont(ctx, im2col_reshaped);
        }
        if (!ggml_is_contiguous(w_reshaped)) {
            w_reshaped = ggml_cont(ctx, w_reshaped);
        }
    }
    
    struct ggml_tensor* result = ggml_mul_mat(ctx, im2col_reshaped, w_reshaped);
    return ggml_reshape_2d(ctx, result, im2col->ne[1], w_f32->ne[2]);
}

struct ggml_tensor* ggml_conv_1d_vits(
    struct ggml_context* ctx,
    struct ggml_tensor* w,
    struct ggml_tensor* x,
    int stride,
    int padding,
    int dilation,
    ggml_backend_t backend
) {
    return ggml_ops_conv_1d(ctx, w, x, stride, padding, dilation, 1, backend);
}

struct ggml_tensor* ggml_conv_1d_with_bias(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* w,
    struct ggml_tensor* b,
    int stride,
    int dilation,
    int padding,
    ggml_backend_t backend
) {
    struct ggml_tensor* x_transposed = ggml_cont(ctx, ggml_transpose(ctx, x));
    struct ggml_tensor* conv = ggml_ops_conv_1d(ctx, w, x_transposed, stride, padding, dilation, 1, backend, b);
    return ggml_cont(ctx, ggml_transpose(ctx, conv));
}

struct ggml_tensor* ggml_conv_transpose_1d_with_bias(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* w,
    struct ggml_tensor* b,
    int stride,
    int padding,
    ggml_backend_t backend
) {
    struct ggml_tensor* x_transposed = ggml_cont(ctx, ggml_transpose(ctx, x));
    struct ggml_tensor* conv_t = ggml_ops_conv_transpose_1d(ctx, w, x_transposed, stride, padding, 1, 1, backend, b);
    return ggml_cont(ctx, ggml_transpose(ctx, conv_t));
}

struct ggml_tensor* ggml_conv_1d_with_bias_no_transpose(
    struct ggml_context* ctx,
    struct ggml_tensor* x,      // [seq_len, in_channels]
    struct ggml_tensor* w,      // [kernel_size, in_channels, out_channels]
    struct ggml_tensor* b,      // [out_channels]
    int stride,
    int dilation,
    int padding,
    ggml_backend_t backend
) {
    return ggml_ops_conv_1d(ctx, w, x, stride, padding, dilation, 1, backend, b);
}

struct ggml_tensor* ggml_conv_transpose_1d_with_bias_no_transpose(
    struct ggml_context* ctx,
    struct ggml_tensor* x,      // [seq_len, in_channels]
    struct ggml_tensor* w,      // [kernel_size, out_channels, in_channels]
    struct ggml_tensor* b,      // [out_channels]
    int stride,
    int padding,
    ggml_backend_t backend
) {
    return ggml_ops_conv_transpose_1d(ctx, w, x, stride, padding, 1, 1, backend, b);
}

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

        std::string prefix1 = "dec.resblocks." + std::to_string(block_idx) + ".convs1." + std::to_string(l);
        std::string prefix2 = "dec.resblocks." + std::to_string(block_idx) + ".convs2." + std::to_string(l);

        struct ggml_tensor* c1_w = nullptr;
        int dilation_effective = dilation;
        if (dilation > 1) {
            c1_w = model.get_tensor(prefix1 + ".weight_dilated");
            dilation_effective = 1;
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

struct ggml_tensor* build_vits_generator(
    struct ggml_context* ctx_graph,
    struct ggml_tensor* latent,
    struct ggml_tensor* speaker_embedding,
    VITSModel& model,
    ggml_backend_t backend
) {
    struct ggml_tensor* dec_conv_pre_w = model.get_tensor("dec.conv_pre.weight");
    struct ggml_tensor* dec_conv_pre_b = model.get_tensor("dec.conv_pre.bias");
    if (!dec_conv_pre_w || !dec_conv_pre_b) {
        std::cerr << "[VITS] Error: Missing dec.conv_pre weights!" << std::endl;
        return nullptr;
    }

    struct ggml_tensor* latent_transposed = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, latent));
    struct ggml_tensor* h = ggml_conv_1d_with_bias_no_transpose(ctx_graph, latent_transposed, dec_conv_pre_w, dec_conv_pre_b, 1, 1, 3, backend);
    model.debug_conv_pre = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, h));

    if (speaker_embedding != nullptr) {
        struct ggml_tensor* cond_w = model.get_tensor("dec.cond.weight");
        struct ggml_tensor* cond_b = model.get_tensor("dec.cond.bias");
        if (cond_w && cond_b) {
            struct ggml_tensor* g_proj_t = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, speaker_embedding));
            struct ggml_tensor* g_proj = ggml_conv_1d_with_bias_no_transpose(ctx_graph, g_proj_t, cond_w, cond_b, 1, 1, 0, backend); // [1, channels]
            model.debug_cond = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, g_proj));
            h = ggml_add(ctx_graph, h, g_proj);
        }
    }

    const std::vector<int> dilations = {1, 3, 5};

    struct ggml_tensor* ups0_w = model.get_tensor("dec.ups.0.weight");
    struct ggml_tensor* ups0_b = model.get_tensor("dec.ups.0.bias");
    struct ggml_tensor* ups1_w = model.get_tensor("dec.ups.1.weight");
    struct ggml_tensor* ups1_b = model.get_tensor("dec.ups.1.bias");
    struct ggml_tensor* ups2_w = model.get_tensor("dec.ups.2.weight");
    struct ggml_tensor* ups2_b = model.get_tensor("dec.ups.2.bias");
    struct ggml_tensor* ups3_w = model.get_tensor("dec.ups.3.weight");
    struct ggml_tensor* ups3_b = model.get_tensor("dec.ups.3.bias");
    struct ggml_tensor* ups4_w = model.get_tensor("dec.ups.4.weight");
    struct ggml_tensor* ups4_b = model.get_tensor("dec.ups.4.bias");

    if (ups0_w && ups0_b) {
        h = ggml_leaky_relu(ctx_graph, h, 0.1f, false);
        h = ggml_conv_transpose_1d_with_bias_no_transpose(ctx_graph, h, ups0_w, ups0_b, 10, 3, backend);
        if (GPT_SOVITS_DEBUG_ENABLED()) {
            model.debug_ups[0] = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, h));
        }
        
        struct ggml_tensor* r0 = mrf_resblock_no_transpose(ctx_graph, h, model, 0, 256, 3, dilations, backend);
        struct ggml_tensor* r1 = mrf_resblock_no_transpose(ctx_graph, h, model, 1, 256, 7, dilations, backend);
        struct ggml_tensor* r2 = mrf_resblock_no_transpose(ctx_graph, h, model, 2, 256, 11, dilations, backend);
        
        if (GPT_SOVITS_DEBUG_ENABLED()) {
            model.debug_resblocks[0] = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, r0));
            model.debug_resblocks[1] = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, r1));
            model.debug_resblocks[2] = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, r2));
        }
        
        struct ggml_tensor* xs = ggml_add(ctx_graph, r0, r1);
        xs = ggml_add(ctx_graph, xs, r2);
        h = ggml_scale(ctx_graph, xs, 1.0f / 3.0f);
    }

    if (ups1_w && ups1_b) {
        h = ggml_leaky_relu(ctx_graph, h, 0.1f, false);
        h = ggml_conv_transpose_1d_with_bias_no_transpose(ctx_graph, h, ups1_w, ups1_b, 8, 4, backend);
        if (GPT_SOVITS_DEBUG_ENABLED()) {
            model.debug_ups[1] = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, h));
        }
        
        struct ggml_tensor* r3 = mrf_resblock_no_transpose(ctx_graph, h, model, 3, 128, 3, dilations, backend);
        struct ggml_tensor* r4 = mrf_resblock_no_transpose(ctx_graph, h, model, 4, 128, 7, dilations, backend);
        struct ggml_tensor* r5 = mrf_resblock_no_transpose(ctx_graph, h, model, 5, 128, 11, dilations, backend);
        
        if (GPT_SOVITS_DEBUG_ENABLED()) {
            model.debug_resblocks[3] = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, r3));
            model.debug_resblocks[4] = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, r4));
            model.debug_resblocks[5] = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, r5));
        }
        
        struct ggml_tensor* xs = ggml_add(ctx_graph, r3, r4);
        xs = ggml_add(ctx_graph, xs, r5);
        h = ggml_scale(ctx_graph, xs, 1.0f / 3.0f);
    }

    if (ups2_w && ups2_b) {
        h = ggml_leaky_relu(ctx_graph, h, 0.1f, false);
        h = ggml_conv_transpose_1d_with_bias_no_transpose(ctx_graph, h, ups2_w, ups2_b, 2, 3, backend);
        if (GPT_SOVITS_DEBUG_ENABLED()) {
            model.debug_ups[2] = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, h));
        }
        
        struct ggml_tensor* r6 = mrf_resblock_no_transpose(ctx_graph, h, model, 6, 64, 3, dilations, backend);
        struct ggml_tensor* r7 = mrf_resblock_no_transpose(ctx_graph, h, model, 7, 64, 7, dilations, backend);
        struct ggml_tensor* r8 = mrf_resblock_no_transpose(ctx_graph, h, model, 8, 64, 11, dilations, backend);
        
        if (GPT_SOVITS_DEBUG_ENABLED()) {
            model.debug_resblocks[6] = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, r6));
            model.debug_resblocks[7] = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, r7));
            model.debug_resblocks[8] = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, r8));
        }
        
        struct ggml_tensor* xs = ggml_add(ctx_graph, r6, r7);
        xs = ggml_add(ctx_graph, xs, r8);
        h = ggml_scale(ctx_graph, xs, 1.0f / 3.0f);
    }

    if (ups3_w && ups3_b) {
        h = ggml_leaky_relu(ctx_graph, h, 0.1f, false);
        h = ggml_conv_transpose_1d_with_bias_no_transpose(ctx_graph, h, ups3_w, ups3_b, 2, 0, backend);
        if (GPT_SOVITS_DEBUG_ENABLED()) {
            model.debug_ups[3] = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, h));
        }
        
        struct ggml_tensor* r9 = mrf_resblock_no_transpose(ctx_graph, h, model, 9, 32, 3, dilations, backend);
        struct ggml_tensor* r10 = mrf_resblock_no_transpose(ctx_graph, h, model, 10, 32, 7, dilations, backend);
        struct ggml_tensor* r11 = mrf_resblock_no_transpose(ctx_graph, h, model, 11, 32, 11, dilations, backend);
        
        if (GPT_SOVITS_DEBUG_ENABLED()) {
            model.debug_resblocks[9] = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, r9));
            model.debug_resblocks[10] = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, r10));
            model.debug_resblocks[11] = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, r11));
        }
        
        struct ggml_tensor* xs = ggml_add(ctx_graph, r9, r10);
        xs = ggml_add(ctx_graph, xs, r11);
        h = ggml_scale(ctx_graph, xs, 1.0f / 3.0f);
    }

    if (ups4_w && ups4_b) {
        h = ggml_leaky_relu(ctx_graph, h, 0.1f, false);
        h = ggml_conv_transpose_1d_with_bias_no_transpose(ctx_graph, h, ups4_w, ups4_b, 2, 0, backend);
        if (GPT_SOVITS_DEBUG_ENABLED()) {
            model.debug_ups[4] = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, h));
        }
        
        struct ggml_tensor* r12 = mrf_resblock_no_transpose(ctx_graph, h, model, 12, 16, 3, dilations, backend);
        struct ggml_tensor* r13 = mrf_resblock_no_transpose(ctx_graph, h, model, 13, 16, 7, dilations, backend);
        struct ggml_tensor* r14 = mrf_resblock_no_transpose(ctx_graph, h, model, 14, 16, 11, dilations, backend);
        
        if (GPT_SOVITS_DEBUG_ENABLED()) {
            model.debug_resblocks[12] = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, r12));
            model.debug_resblocks[13] = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, r13));
            model.debug_resblocks[14] = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, r14));
        }
        
        struct ggml_tensor* xs = ggml_add(ctx_graph, r12, r13);
        xs = ggml_add(ctx_graph, xs, r14);
        h = ggml_scale(ctx_graph, xs, 1.0f / 3.0f);
    }

    struct ggml_tensor* conv_post_w = model.get_tensor("dec.conv_post.weight");
    if (!conv_post_w) {
        std::cerr << "[VITS] Error: Missing dec.conv_post.weight!" << std::endl;
        return nullptr;
    }

    h = ggml_leaky_relu(ctx_graph, h, 0.01f, false);
    struct ggml_tensor* conv = ggml_conv_1d_vits(ctx_graph, conv_post_w, h, 1, 3, 1, backend); // [out_seq_len, 1]
    struct ggml_tensor* audio = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, conv));
    model.debug_conv_post = audio;
    return ggml_tanh(ctx_graph, audio);
}

static struct ggml_tensor* ggml_mish(
    struct ggml_context* ctx,
    struct ggml_tensor* x
) {
    struct ggml_tensor* x_f32 = (x->type == GGML_TYPE_F32) ? x : ggml_cont(ctx, ggml_cast(ctx, x, GGML_TYPE_F32));
    struct ggml_tensor* exp_x = ggml_exp(ctx, x_f32);
    struct ggml_tensor* ones = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, x_f32->ne[0], x_f32->ne[1]);
    ones = ggml_fill(ctx, ones, 1.0f);
    struct ggml_tensor* sp = ggml_log(ctx, ggml_add(ctx, exp_x, ones));
    return ggml_mul(ctx, x_f32, ggml_tanh(ctx, sp));
}

static bool our_ggml_can_mul_mat(const struct ggml_tensor* a, const struct ggml_tensor* b) {
    return a->ne[0] == b->ne[0] && a->ne[2] == b->ne[2] && a->ne[3] == b->ne[3];
}

struct ggml_tensor* ggml_linear(
    struct ggml_context* ctx,
    struct ggml_tensor* x,     // ne0=in_features, ne1=T
    struct ggml_tensor* w,     // ne0=in_features, ne1=out_features (or 3D: [k=1, in, out])
    struct ggml_tensor* b      // [out_features] or nullptr
) {
    if (w->ne[2] > 1 && w->ne[0] == 1) {
        w = ggml_cont(ctx, ggml_reshape_2d(ctx, w, w->ne[1], w->ne[2]));  // [in, out]
    }
    struct ggml_tensor* w_f32 = force_w_f32(ctx, w);
    if (!our_ggml_can_mul_mat(w_f32, x)) {
        std::cerr << "[ggml_linear ERROR] w name: " << (w->name ? w->name : "NULL")
                  << " shape: [" << w_f32->ne[0] << ", " << w_f32->ne[1] << ", " << w_f32->ne[2] << ", " << w_f32->ne[3] << "]"
                  << " | x name: " << (x->name ? x->name : "NULL")
                  << " shape: [" << x->ne[0] << ", " << x->ne[1] << ", " << x->ne[2] << ", " << x->ne[3] << "]"
                  << std::endl;
    }
    struct ggml_tensor* out = ggml_mul_mat(ctx, w_f32, x);
    ggml_mul_mat_set_prec(out, GGML_PREC_F32);
    if (b) {
        struct ggml_tensor* b2d = ggml_reshape_2d(ctx, b, b->ne[0], 1);
        out = ggml_add(ctx, out, b2d);
    }
    return out;
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
    int pad = (int)(w->ne[0] - 1) / 2;
    struct ggml_tensor* conv = ggml_conv_1d_vits(ctx, w, x_t, 1, pad, 1, backend);
    struct ggml_tensor* conv_t = ggml_cont(ctx, ggml_transpose(ctx, conv));  // [2*C, T]

    struct ggml_tensor* b2d = ggml_reshape_2d(ctx, b, b->ne[0], 1);
    conv_t = ggml_add(ctx, conv_t, b2d);

    struct ggml_tensor* x1 = ggml_view_2d(ctx, conv_t, in_ch, conv_t->ne[1], conv_t->nb[1], 0);
    struct ggml_tensor* x2 = ggml_view_2d(ctx, conv_t, in_ch, conv_t->ne[1], conv_t->nb[1], in_ch * sizeof(float));

    struct ggml_tensor* glu = ggml_mul(ctx, x1, ggml_sigmoid(ctx, x2));
    return ggml_add(ctx, x, glu);
}

struct ggml_tensor* ggml_layer_norm(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* gamma,  // [channels]
    struct ggml_tensor* beta,   // [channels]
    float eps,
    ggml_backend_t backend
) {
    return ggml_ops_layer_norm(ctx, x, gamma, beta, eps, backend);
}

static struct ggml_tensor* build_encoder_layer(
    struct ggml_context* ctx,
    struct ggml_tensor* x,        // [C, T], ne0=C=192, ne1=T
    struct ggml_tensor* x_mask,   // [1, T] mask tensor
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
    int C = (int)x->ne[0];  // hidden_channels = 192

    struct ggml_tensor* q_w = model.get_tensor(prefix + "conv_q.weight"); // [1, C, C]
    struct ggml_tensor* q_b = model.get_tensor(prefix + "conv_q.bias");
    struct ggml_tensor* k_w = model.get_tensor(prefix + "conv_k.weight");
    struct ggml_tensor* k_b = model.get_tensor(prefix + "conv_k.bias");
    struct ggml_tensor* v_w = model.get_tensor(prefix + "conv_v.weight");
    struct ggml_tensor* v_b = model.get_tensor(prefix + "conv_v.bias");
    struct ggml_tensor* o_w = model.get_tensor(prefix + "conv_o.weight");
    struct ggml_tensor* o_b = model.get_tensor(prefix + "conv_o.bias");

    struct ggml_tensor* q = ggml_conv_1d_with_bias(ctx, x, q_w, q_b, 1, 1, 0, backend);
    struct ggml_tensor* k = ggml_conv_1d_with_bias(ctx, x, k_w, k_b, 1, 1, 0, backend);
    struct ggml_tensor* v = ggml_conv_1d_with_bias(ctx, x, v_w, v_b, 1, 1, 0, backend);

    if (prefix.find("enc_p.encoder_ssl.attn_layers.0.") != std::string::npos) {
        model.debug_enc_q = q;
    }

    q = ggml_cont(ctx, ggml_reshape_4d(ctx, q, d_k, n_head, T, 1));
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3)); // [d_k, T, n_head, 1]
    k = ggml_cont(ctx, ggml_reshape_4d(ctx, k, d_k, n_head, T, 1));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_reshape_4d(ctx, v, d_k, n_head, T, 1));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);
    struct ggml_tensor* k_f32 = (k->type == GGML_TYPE_F32) ? k : ggml_cast(ctx, k, GGML_TYPE_F32);
    struct ggml_tensor* q_f32 = (q->type == GGML_TYPE_F32) ? q : ggml_cast(ctx, q, GGML_TYPE_F32);
    struct ggml_tensor* scores = ggml_mul_mat(ctx, k_f32, q_f32);
    scores = ggml_scale(ctx, scores, inv_sqrt_dk);
    struct ggml_tensor* scores_before_rel = scores;

    if (emb_rel_k != nullptr) {
        int pad_length = std::max(T - 5, 0);
        struct ggml_tensor* padded_emb_k = emb_rel_k;
        if (pad_length > 0) {
            struct ggml_tensor* zeros_pad = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_k, pad_length, emb_rel_k->ne[2]);
            zeros_pad = ggml_fill(ctx, zeros_pad, 0.0f);
            struct ggml_tensor* temp = ggml_concat(ctx, zeros_pad, emb_rel_k, 1);
            padded_emb_k = ggml_concat(ctx, temp, zeros_pad, 1);
        }
        int slice_start = std::max(5 - T, 0);
        int slice_len = 2 * T - 1;
        size_t offset_k = slice_start * padded_emb_k->nb[1];
        struct ggml_tensor* rel_emb_k = ggml_view_3d(ctx, padded_emb_k, d_k, slice_len, emb_rel_k->ne[2], padded_emb_k->nb[1], padded_emb_k->nb[2], offset_k);

        struct ggml_tensor* q_scaled = ggml_scale(ctx, q, inv_sqrt_dk);
        struct ggml_tensor* rel_emb_k_f32 = (rel_emb_k->type == GGML_TYPE_F32) ? rel_emb_k : ggml_cast(ctx, rel_emb_k, GGML_TYPE_F32);
        struct ggml_tensor* q_scaled_f32 = (q_scaled->type == GGML_TYPE_F32) ? q_scaled : ggml_cast(ctx, q_scaled, GGML_TYPE_F32);
        struct ggml_tensor* rel_logits = ggml_mul_mat(ctx, rel_emb_k_f32, q_scaled_f32);

        struct ggml_tensor* zeros_col = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, T, n_head);
        zeros_col = ggml_fill(ctx, zeros_col, 0.0f);
        struct ggml_tensor* x_padded = ggml_concat(ctx, rel_logits, zeros_col, 0); // [2*T, T, n_head]

        struct ggml_tensor* x_flat = ggml_reshape_2d(ctx, x_padded, T * 2 * T, n_head);

        struct ggml_tensor* zeros_flat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T - 1, n_head);
        zeros_flat = ggml_fill(ctx, zeros_flat, 0.0f);
        struct ggml_tensor* x_flat_padded = ggml_concat(ctx, x_flat, zeros_flat, 0);

        struct ggml_tensor* x_final = ggml_reshape_3d(ctx, x_flat_padded, 2 * T - 1, T + 1, n_head);

        size_t view_offset = (T - 1) * sizeof(float);
        struct ggml_tensor* scores_local = ggml_view_3d(ctx, x_final, T, T, n_head, x_final->nb[1], x_final->nb[2], view_offset);

        struct ggml_tensor* scores_local_cont = ggml_cont(ctx, scores_local);
        scores = ggml_add(ctx, scores, scores_local_cont);
    }

    struct ggml_tensor* attn_w = ggml_soft_max(ctx, scores);  // [T_k, T_q, n_head, 1], ne0=T_k

    struct ggml_tensor* v_t = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
    struct ggml_tensor* v_t_f32 = (v_t->type == GGML_TYPE_F32) ? v_t : ggml_cast(ctx, v_t, GGML_TYPE_F32);
    struct ggml_tensor* attn_w_f32 = (attn_w->type == GGML_TYPE_F32) ? attn_w : ggml_cast(ctx, attn_w, GGML_TYPE_F32);
    struct ggml_tensor* out = ggml_mul_mat(ctx, v_t_f32, attn_w_f32);  // v_t^T * attn_w → [d_k, T_q, n_head, 1]

    out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
    struct ggml_tensor* attn_raw = ggml_reshape_2d(ctx, out, d_k * n_head, T);

    if (emb_rel_v != nullptr) {
        int pad_length = std::max(T - 5, 0);
        struct ggml_tensor* padded_emb_v = emb_rel_v;
        if (pad_length > 0) {
            struct ggml_tensor* zeros_pad = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_k, pad_length, emb_rel_v->ne[2]);
            zeros_pad = ggml_fill(ctx, zeros_pad, 0.0f);
            struct ggml_tensor* temp = ggml_concat(ctx, zeros_pad, emb_rel_v, 1);
            padded_emb_v = ggml_concat(ctx, temp, zeros_pad, 1);
        }
        int slice_start = std::max(5 - T, 0);
        int slice_len = 2 * T - 1;
        size_t offset_v = slice_start * padded_emb_v->nb[1];
        struct ggml_tensor* rel_emb_v = ggml_view_3d(ctx, padded_emb_v, d_k, slice_len, emb_rel_v->ne[2], padded_emb_v->nb[1], padded_emb_v->nb[2], offset_v);

        struct ggml_tensor* zeros_cols = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T - 1, T, n_head);
        zeros_cols = ggml_fill(ctx, zeros_cols, 0.0f);
        struct ggml_tensor* x_padded = ggml_concat(ctx, attn_w, zeros_cols, 0); // [2*T-1, T, n_head]

        struct ggml_tensor* x_flat = ggml_reshape_2d(ctx, x_padded, T * (2 * T - 1), n_head);

        struct ggml_tensor* zeros_beg = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, n_head);
        zeros_beg = ggml_fill(ctx, zeros_beg, 0.0f);
        struct ggml_tensor* x_flat_padded = ggml_concat(ctx, zeros_beg, x_flat, 0);

        struct ggml_tensor* x_final = ggml_reshape_3d(ctx, x_flat_padded, 2 * T, T, n_head);

        size_t view_offset = 1 * sizeof(float);
        struct ggml_tensor* rel_weights = ggml_cont(ctx, ggml_view_3d(ctx, x_final, 2 * T - 1, T, n_head, x_final->nb[1], x_final->nb[2], view_offset));

        struct ggml_tensor* rel_emb_v_t = ggml_cont(ctx, ggml_transpose(ctx, rel_emb_v));

        struct ggml_tensor* dummy = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, slice_len, d_k, n_head);
        struct ggml_tensor* rel_emb_v_t_repeated = ggml_cont(ctx, ggml_repeat(ctx, rel_emb_v_t, dummy));
        struct ggml_tensor* rel_weights_f32 = (rel_weights->type == GGML_TYPE_F32) ? rel_weights : ggml_cast(ctx, rel_weights, GGML_TYPE_F32);
        struct ggml_tensor* rel_emb_v_t_repeated_f32 = (rel_emb_v_t_repeated->type == GGML_TYPE_F32) ? rel_emb_v_t_repeated : ggml_cast(ctx, rel_emb_v_t_repeated, GGML_TYPE_F32);
        struct ggml_tensor* rel_out_bias = ggml_mul_mat(ctx, rel_weights_f32, rel_emb_v_t_repeated_f32);

        rel_out_bias = ggml_cont(ctx, ggml_permute(ctx, rel_out_bias, 2, 0, 1, 3));
        struct ggml_tensor* rel_out_bias_flat = ggml_reshape_2d(ctx, rel_out_bias, d_k * n_head, T);

        attn_raw = ggml_add(ctx, attn_raw, rel_out_bias_flat);
    }

    if (prefix.find("enc_p.encoder_ssl.attn_layers.0.") != std::string::npos) {
        model.debug_enc_fa_raw = scores_before_rel;
        model.debug_enc_scores = scores;
        model.debug_enc_attn_w = attn_w;
        model.debug_enc_out_raw = attn_raw;
    }

    struct ggml_tensor* attn_proj = ggml_conv_1d_with_bias(ctx, attn_raw, o_w, o_b, 1, 1, 0, backend);
    struct ggml_tensor* x_attn = ggml_add(ctx, x, attn_proj);

    struct ggml_tensor* ln1_g = model.get_tensor(norm1_prefix + ".gamma");
    struct ggml_tensor* ln1_b = model.get_tensor(norm1_prefix + ".beta");
    x_attn = ggml_layer_norm(ctx, x_attn, ln1_g, ln1_b, 1e-5f, backend);

    struct ggml_tensor* ffn_w1 = model.get_tensor(ffn1_prefix + ".weight");
    struct ggml_tensor* ffn_b1 = model.get_tensor(ffn1_prefix + ".bias");
    struct ggml_tensor* ffn_w2 = model.get_tensor(ffn2_prefix + ".weight");
    struct ggml_tensor* ffn_b2 = model.get_tensor(ffn2_prefix + ".bias");

    struct ggml_tensor* ffn_out = ggml_conv_1d_with_bias(ctx, x_attn, ffn_w1, ffn_b1, 1, 1, 1, backend);  // kernel=3, pad=1
    ffn_out = ggml_relu(ctx, ffn_out);
    ffn_out = ggml_conv_1d_with_bias(ctx, ffn_out, ffn_w2, ffn_b2, 1, 1, 1, backend);

    struct ggml_tensor* x_out = ggml_add(ctx, x_attn, ffn_out);
    struct ggml_tensor* ln2_g = model.get_tensor(norm2_prefix + ".gamma");
    struct ggml_tensor* ln2_b = model.get_tensor(norm2_prefix + ".beta");
    struct ggml_tensor* result = ggml_layer_norm(ctx, x_out, ln2_g, ln2_b, 1e-5f, backend);

    if (prefix.find("enc_p.encoder_ssl.attn_layers.0.") != std::string::npos) {
        model.debug_enc_attn = result;
    }

    return result;
}

struct ggml_tensor* build_encoder(
    struct ggml_context* ctx,
    struct ggml_tensor* x,        // [C, T]
    struct ggml_tensor* x_mask,   // mask
    VITSModel& model,
    const std::string& base_prefix,  // "enc_p.encoder_ssl"
    int n_layers,
    int n_head,
    int d_k,
    int T,
    ggml_backend_t backend
) {
    std::string enc_short = base_prefix;
    size_t dot_pos = enc_short.rfind('.');
    if (dot_pos != std::string::npos) enc_short = enc_short.substr(dot_pos + 1);

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

        static std::unordered_map<std::string, std::vector<float>> rel_k_bufs, rel_v_bufs;
        if (!emb_rel_k) {
            std::string fk = "scratch/enc_relk_" + enc_short + "_layer" + std::to_string(l) + ".f32";
            std::ifstream f(fk, std::ios::binary);
            if (f) {
                auto& buf = rel_k_bufs[enc_short + std::to_string(l)];
                buf.resize(96 * 9);
                f.read((char*)buf.data(), 96 * 9 * sizeof(float));
                emb_rel_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 96, 9, 1);
                emb_rel_k->data = buf.data();
            }
        }
        if (!emb_rel_v) {
            std::string fv = "scratch/enc_relv_" + enc_short + "_layer" + std::to_string(l) + ".f32";
            std::ifstream f(fv, std::ios::binary);
            if (f) {
                auto& buf = rel_v_bufs[enc_short + std::to_string(l)];
                buf.resize(96 * 9);
                f.read((char*)buf.data(), 96 * 9 * sizeof(float));
                emb_rel_v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 96, 9, 1);
                emb_rel_v->data = buf.data();
            }
        }

        x = build_encoder_layer(ctx, x, x_mask, model, lp, n1p, n2p, f1p, f2p, n_head, d_k, T, emb_rel_k, emb_rel_v, backend);
    }
    return x;
}

struct ggml_tensor* build_mrte(
    struct ggml_context* ctx,
    struct ggml_tensor* y,        // SSL features [192, T_y]
    struct ggml_tensor* y_mask,   // [1, T_y]
    struct ggml_tensor* text,     // Text features [192, T_x]
    struct ggml_tensor* text_mask,// [1, T_x]
    struct ggml_tensor* ge,       // Speaker embedding [512, 1]
    VITSModel& model,
    ggml_backend_t backend
) {
    (void)text_mask;
    int T_y = (int)y->ne[1];
    int T_x = (int)text->ne[1];

    struct ggml_tensor* c_pre_w = model.get_tensor("enc_p.mrte.c_pre.weight");
    struct ggml_tensor* c_pre_b = model.get_tensor("enc_p.mrte.c_pre.bias");
    struct ggml_tensor* y_proj = ggml_linear(ctx, y, c_pre_w, c_pre_b);  // [512, T_y]

    struct ggml_tensor* text_pre_w = model.get_tensor("enc_p.mrte.text_pre.weight");
    struct ggml_tensor* text_pre_b = model.get_tensor("enc_p.mrte.text_pre.bias");
    struct ggml_tensor* text_proj = ggml_linear(ctx, text, text_pre_w, text_pre_b);  // [512, T_x]

    if (std::getenv("ENC_ALIGNMENT")) {
        model.debug_ref_enc_spectral_0 = y_proj;
        model.debug_ref_enc_spectral_3 = text_proj;
    }

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

    struct ggml_tensor* q_mrte = ggml_linear(ctx, y_proj, q_w, q_b);  // [512, T_y]
    struct ggml_tensor* k_mrte = ggml_linear(ctx, text_proj, k_w, k_b);  // [512, T_x]
    struct ggml_tensor* v_mrte = ggml_linear(ctx, text_proj, v_w, v_b);  // [512, T_x]

    if (std::getenv("ENC_ALIGNMENT")) {
        model.debug_enc_q_cont = q_mrte;
        model.debug_enc_fa_raw = k_mrte;
    }

    q_mrte = ggml_cont(ctx, ggml_reshape_3d(ctx, q_mrte, mrte_d_k, mrte_n_head, T_y));
    q_mrte = ggml_cont(ctx, ggml_permute(ctx, q_mrte, 0, 2, 1, 3));  // [128, T_y, 4]

    k_mrte = ggml_cont(ctx, ggml_reshape_3d(ctx, k_mrte, mrte_d_k, mrte_n_head, T_x));
    k_mrte = ggml_cont(ctx, ggml_permute(ctx, k_mrte, 0, 2, 1, 3));  // [128, T_x, 4]

    v_mrte = ggml_cont(ctx, ggml_reshape_3d(ctx, v_mrte, mrte_d_k, mrte_n_head, T_x));
    v_mrte = ggml_cont(ctx, ggml_permute(ctx, v_mrte, 0, 2, 1, 3));  // [128, T_x, 4]

    float mrte_scale = 1.0f / sqrtf((float)mrte_d_k);
    struct ggml_tensor* k_mrte_f32 = (k_mrte->type == GGML_TYPE_F32) ? k_mrte : ggml_cast(ctx, k_mrte, GGML_TYPE_F32);
    struct ggml_tensor* q_mrte_f32 = (q_mrte->type == GGML_TYPE_F32) ? q_mrte : ggml_cast(ctx, q_mrte, GGML_TYPE_F32);
    struct ggml_tensor* scores_mrte = ggml_mul_mat(ctx, k_mrte_f32, q_mrte_f32);
    scores_mrte = ggml_scale(ctx, scores_mrte, mrte_scale);

    struct ggml_tensor* attn_w_mrte = ggml_soft_max(ctx, scores_mrte);

    struct ggml_tensor* v_t_mrte = ggml_cont(ctx, ggml_permute(ctx, v_mrte, 1, 0, 2, 3));
    struct ggml_tensor* v_t_mrte_f32 = (v_t_mrte->type == GGML_TYPE_F32) ? v_t_mrte : ggml_cast(ctx, v_t_mrte, GGML_TYPE_F32);
    struct ggml_tensor* attn_w_mrte_f32 = (attn_w_mrte->type == GGML_TYPE_F32) ? attn_w_mrte : ggml_cast(ctx, attn_w_mrte, GGML_TYPE_F32);
    struct ggml_tensor* out_mrte = ggml_mul_mat(ctx, v_t_mrte_f32, attn_w_mrte_f32);  // [d_k, T_y, n_head]

    out_mrte = ggml_cont(ctx, ggml_permute(ctx, out_mrte, 0, 2, 1, 3));
    struct ggml_tensor* cross_out = ggml_reshape_2d(ctx, out_mrte, mrte_d_k * mrte_n_head, T_y);

    struct ggml_tensor* cross_proj = ggml_linear(ctx, cross_out, o_w, o_b);  // [512, T_y]

    if (std::getenv("ENC_ALIGNMENT")) {
        model.debug_enc_vt = cross_out;
        model.debug_ref_enc_temporal_0 = cross_proj;
    }

    struct ggml_tensor* ge_2d = ggml_reshape_2d(ctx, ge, 512, 1);
    struct ggml_tensor* mrte_res = ggml_add(ctx, y_proj, cross_proj);
    mrte_res = ggml_add(ctx, mrte_res, ge_2d);

    if (std::getenv("ENC_ALIGNMENT")) {
        model.debug_enc_vt = cross_out;
        model.debug_ref_enc_pre_attn = mrte_res;
    }

    struct ggml_tensor* c_post_w = model.get_tensor("enc_p.mrte.c_post.weight");
    struct ggml_tensor* c_post_b = model.get_tensor("enc_p.mrte.c_post.bias");
    struct ggml_tensor* result = ggml_linear(ctx, mrte_res, c_post_w, c_post_b);  // [192, T_y]

    if (std::getenv("ENC_ALIGNMENT")) {
        model.debug_enc_mrte_out = result;
    }

    return result;
}

struct ggml_tensor* build_wn(
    struct ggml_context* ctx,
    struct ggml_tensor* x,        // [C, T]
    struct ggml_tensor* x_mask,   // [1, T] or nullptr
    struct ggml_tensor* g,        // conditioning
    VITSModel& model,
    const std::string& prefix,    // "flow.flows.0.enc."
    int hidden_channels,          // 192
    int kernel_size,              // 5
    int dilation_rate,            // 1
    int n_layers,                 // 4
    ggml_backend_t backend
) {
    struct ggml_tensor* output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_channels, x->ne[1]);
    output = ggml_fill(ctx, output, 0.0f);

    for (int i = 0; i < n_layers; ++i) {
        int dilation = (int)std::pow((float)dilation_rate, i);
        int padding = (int)((kernel_size * dilation - dilation) / 2);

        std::string ilp = prefix + "in_layers." + std::to_string(i);
        struct ggml_tensor* in_w = model.get_tensor(ilp + ".weight");
        struct ggml_tensor* in_b = model.get_tensor(ilp + ".bias");
        struct ggml_tensor* x_in = ggml_conv_1d_with_bias(ctx, x, in_w, in_b, 1, dilation, padding, backend);

        if (g) {
            int cond_offset = i * 2 * hidden_channels;
            struct ggml_tensor* g_l = ggml_view_2d(ctx, g, 2 * hidden_channels, g->ne[1],
                g->nb[1], cond_offset * sizeof(float));
            x_in = ggml_add(ctx, x_in, ggml_cont(ctx, g_l));
        }

        struct ggml_tensor* acts = ggml_ops_gated_tanh_sigmoid(ctx, x_in, hidden_channels, backend);

        if (prefix.find("flow.flows.6.enc.") != std::string::npos && i == 0) {
            model.debug_ref_enc_post_fc = acts;
        }

        std::string rsp = prefix + "res_skip_layers." + std::to_string(i);
        struct ggml_tensor* rs_w = model.get_tensor(rsp + ".weight");
        struct ggml_tensor* rs_b = model.get_tensor(rsp + ".bias");
        struct ggml_tensor* res_skip = ggml_conv_1d_with_bias(ctx, acts, rs_w, rs_b, 1, 1, 0, backend);

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
            if (prefix.find("flow.flows.6.enc.") != std::string::npos && i == 0) {
                model.debug_ref_enc_pre_pool = skip_acts;
            }
        } else {
            output = ggml_add(ctx, output, res_skip);
        }
    }
    if (x_mask) {
        output = ggml_mul(ctx, output, ggml_repeat(ctx, x_mask, output));
    }
    return output;
}

struct ggml_tensor* build_coupling_layer(
    struct ggml_context* ctx,
    struct ggml_tensor* x,        // [channels, T]
    struct ggml_tensor* x_mask,   // [1, T]
    struct ggml_tensor* g,        // [512, 1]
    VITSModel& model,
    const std::string& prefix,    // "flow.flows.0."
    int channels,                 // 192
    int hidden_channels,          // 192
    int kernel_size,              // 5
    int dilation_rate,            // 1
    int n_layers,                 // 4
    bool reverse,                 // true for inference
    ggml_backend_t backend
) {
    int half_c = channels / 2;  // 96

    size_t x_stride = x->ne[0] * sizeof(float);
    struct ggml_tensor* x0 = ggml_view_2d(ctx, x, half_c, x->ne[1], x_stride, 0);
    struct ggml_tensor* x1 = ggml_view_2d(ctx, x, half_c, x->ne[1], x_stride, half_c * sizeof(float));

    if (prefix.find("flow.flows.0.") != std::string::npos) {
        model.debug_ref_enc_temporal_0 = x0;
    }
    struct ggml_tensor* pre_w = model.get_tensor(prefix + "pre.weight");
    struct ggml_tensor* pre_b = model.get_tensor(prefix + "pre.bias");
    struct ggml_tensor* h = ggml_conv_1d_with_bias(ctx, x0, pre_w, pre_b, 1, 1, 0, backend);
    if (x_mask) {
        h = ggml_mul(ctx, h, ggml_repeat(ctx, x_mask, h));
    }

    if (prefix.find("flow.flows.0.") != std::string::npos) {
        model.debug_ref_enc_pre_attn = h;
    }

    struct ggml_tensor* g_proj = nullptr;
    struct ggml_tensor* cond_w = model.get_tensor(prefix + "enc.cond_layer.weight");
    struct ggml_tensor* cond_b = model.get_tensor(prefix + "enc.cond_layer.bias");
    if (cond_w && cond_b && g) {
        g_proj = ggml_conv_1d_with_bias(ctx, g, cond_w, cond_b, 1, 1, 0, backend);
    }

    h = build_wn(ctx, h, x_mask, g_proj, model, prefix + "enc.",
        hidden_channels, kernel_size, dilation_rate, n_layers, backend);

    if (prefix.find("flow.flows.0.") != std::string::npos) {
        model.debug_ref_enc_post_attn = h;
    }

    struct ggml_tensor* post_w = model.get_tensor(prefix + "post.weight");
    struct ggml_tensor* post_b = model.get_tensor(prefix + "post.bias");
    struct ggml_tensor* mean = ggml_conv_1d_with_bias(ctx, h, post_w, post_b, 1, 1, 0, backend);
    if (x_mask) {
        mean = ggml_mul(ctx, mean, ggml_repeat(ctx, x_mask, mean));
    }
    if (prefix.find("flow.flows.0.") != std::string::npos) {
        model.debug_enc_vt = mean;
    }

    if (!reverse) {
        struct ggml_tensor* new_x1 = ggml_add(ctx, mean, ggml_cont(ctx, x1));
        return ggml_cont(ctx, ggml_concat(ctx, x0, new_x1, 0));
    } else {
        struct ggml_tensor* new_x1 = ggml_sub(ctx, ggml_cont(ctx, x1), mean);
        return ggml_cont(ctx, ggml_concat(ctx, x0, new_x1, 0));
    }
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
    x = ggml_linear(ctx, x, s0_w, s0_b);
    model.debug_ref_enc_spectral_0 = x;
    x = ggml_ops_mish(ctx, x, backend);

    struct ggml_tensor* s3_w = model.get_tensor("ref_enc.spectral.3.fc.weight");
    struct ggml_tensor* s3_b = model.get_tensor("ref_enc.spectral.3.fc.bias");
    x = ggml_linear(ctx, x, s3_w, s3_b);
    model.debug_ref_enc_spectral_3 = x;
    x = ggml_ops_mish(ctx, x, backend);

    struct ggml_tensor* t0_w = model.get_tensor("ref_enc.temporal.0.conv1.conv.weight");
    struct ggml_tensor* t0_b = model.get_tensor("ref_enc.temporal.0.conv1.conv.bias");
    x = conv1d_glu(ctx, x, t0_w, t0_b, backend);
    model.debug_ref_enc_temporal_0 = x;

    struct ggml_tensor* t1_w = model.get_tensor("ref_enc.temporal.1.conv1.conv.weight");
    struct ggml_tensor* t1_b = model.get_tensor("ref_enc.temporal.1.conv1.conv.bias");
    x = conv1d_glu(ctx, x, t1_w, t1_b, backend);
    model.debug_ref_enc_temporal_1 = x;

    {
        int n_head = 2;
        int d_k = 64;
        int d_v = 64;

        struct ggml_tensor* residual = x;
        model.debug_ref_enc_pre_attn = x;

        struct ggml_tensor* w_qs = model.get_tensor("ref_enc.slf_attn.w_qs.weight");
        struct ggml_tensor* b_qs = model.get_tensor("ref_enc.slf_attn.w_qs.bias");
        struct ggml_tensor* w_ks = model.get_tensor("ref_enc.slf_attn.w_ks.weight");
        struct ggml_tensor* b_ks = model.get_tensor("ref_enc.slf_attn.w_ks.bias");
        struct ggml_tensor* w_vs = model.get_tensor("ref_enc.slf_attn.w_vs.weight");
        struct ggml_tensor* b_vs = model.get_tensor("ref_enc.slf_attn.w_vs.bias");
        struct ggml_tensor* attn_fc_w = model.get_tensor("ref_enc.slf_attn.fc.weight");
        struct ggml_tensor* attn_fc_b = model.get_tensor("ref_enc.slf_attn.fc.bias");

        struct ggml_tensor* q = ggml_linear(ctx, x, w_qs, b_qs);
        struct ggml_tensor* k = ggml_linear(ctx, x, w_ks, b_ks);
        struct ggml_tensor* v = ggml_linear(ctx, x, w_vs, b_vs);

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
        struct ggml_tensor* kq = ggml_mul_mat(ctx, k_f32, q_f32);
        kq = ggml_scale(ctx, kq, scale);
        kq = ggml_soft_max(ctx, kq);

        struct ggml_tensor* v_perm = ggml_permute(ctx, v, 1, 0, 2, 3);
        struct ggml_tensor* v_cont = ggml_cont(ctx, v_perm);
        struct ggml_tensor* v_cont_f32 = (v_cont->type == GGML_TYPE_F32) ? v_cont : ggml_cast(ctx, v_cont, GGML_TYPE_F32);
        struct ggml_tensor* kq_f32 = (kq->type == GGML_TYPE_F32) ? kq : ggml_cast(ctx, kq, GGML_TYPE_F32);
        struct ggml_tensor* attn_out = ggml_mul_mat(ctx, v_cont_f32, kq_f32);

        attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 1, 0, 2, 3));
        attn_out = ggml_cont(ctx, ggml_reshape_2d(ctx, attn_out, d_v * n_head, T));
        struct ggml_tensor* output = ggml_linear(ctx, attn_out, attn_fc_w, attn_fc_b);
        x = ggml_add(ctx, output, residual);
        model.debug_ref_enc_post_attn = x;
    }
    struct ggml_tensor* fc_w = model.get_tensor("ref_enc.fc.fc.weight");
    struct ggml_tensor* fc_b = model.get_tensor("ref_enc.fc.fc.bias");
    x = ggml_linear(ctx, x, fc_w, fc_b);
    model.debug_ref_enc_post_fc = x;

    x = ggml_cont(ctx, ggml_transpose(ctx, x));
    model.debug_ref_enc_pre_pool = x;
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
    current_vits_backend = backend;
    struct ggml_tensor* ge = build_ref_enc(ctx_graph, mel_spec, *this, backend);
    
    struct ggml_tensor* sv_emb_b = get_tensor("sv_emb.bias");
    struct ggml_tensor* sv_emb_w = get_tensor("sv_emb.weight");
    struct ggml_tensor* prelu_w = get_tensor("prelu.weight");
    if (sv_emb_b && prelu_w) {
        int64_t dim = sv_emb_b->ne[0];
        
        struct ggml_tensor* sv_proj = nullptr;
        if (sv_emb && sv_emb_w) {
            nn::Linear sv_emb_layer(sv_emb_w, sv_emb_b);
            sv_proj = sv_emb_layer.forward(ctx_graph, sv_emb);
            sv_proj = ggml_reshape_2d(ctx_graph, sv_proj, 1, dim);
        } else {
            sv_proj = ggml_reshape_2d(ctx_graph, sv_emb_b, 1, dim);
        }
        
        ge = ggml_add(ctx_graph, ge, sv_proj);
        
        nn::PReLU prelu_layer(prelu_w);
        ge = prelu_layer.forward(ctx_graph, ge, backend);
    }
    
    return ge;
}

struct ggml_tensor* vq_decode(
    struct ggml_context* ctx,
    struct ggml_tensor* token_ids,
    VITSModel& model
) {
    struct ggml_tensor* codebook = model.get_tensor("quantizer.vq.layers.0._codebook.embed");
    if (!codebook) {
        std::cerr << "[VITS] Error: Missing quantizer codebook!" << std::endl;
        return nullptr;
    }
    return ggml_get_rows(ctx, codebook, token_ids);
}

struct ggml_tensor* interp_nearest_2x(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    VITSModel& model
) {
    int64_t C = x->ne[0];
    int64_t T = x->ne[1];
    struct ggml_tensor* x_3d = ggml_reshape_3d(ctx, x, C, 1, T);
    struct ggml_tensor* target = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, 2, T);
    struct ggml_tensor* repeated = ggml_repeat(ctx, x_3d, target);
    return ggml_cont(ctx, ggml_reshape_2d(ctx, repeated, C, T * 2));
}

} // namespace gpt_sovits
