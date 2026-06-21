#include "models.h"
#include <cstdlib>
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
        ggml_backend_tensor_set(entry.tensor, entry.data.data(), 0, entry.data.size());
    }
    upload_entries.clear();
}

// Helper: cast weights to FP32 for compute (CUDA FP16 gemm not supported on this GPU)
static struct ggml_tensor* force_w_f32(struct ggml_context* ctx, struct ggml_tensor* w) {
    if (!w) return nullptr;
    if (w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_F16) {
        // It is a quantized type, return it as-is to use optimized CPU kernels!
        return w;
    }
    if (w->type == GGML_TYPE_F32) return w;
    struct ggml_tensor* casted = ggml_cast(ctx, w, GGML_TYPE_F32);
    return ggml_cont(ctx, casted);
}

// Helper: Linear layer (matmul + bias) — weight cast to FP32 if needed
static struct ggml_tensor* ggml_linear(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* w, struct ggml_tensor* b);

static struct ggml_context* vits_custom_ctx = nullptr;
static ggml_backend_buffer_t vits_custom_buf = nullptr;

bool VITSModel::load(const std::string& path, ggml_backend_t backend) {
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS] Loading VITS GGUF model: " << path << std::endl;
    if (!load_gguf_model(path, *this, backend)) {
        std::cerr << "[VITS] Failed to load GGUF VITS model!" << std::endl;
        return false;
    }
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS] Loaded VITS successfully. Pre-computing Weight Normalization..." << std::endl;
    
    bool is_cuda = false;
    if (backend) {
        const char * bname = ggml_backend_name(backend);
        if (bname && strncmp(bname, "CUDA", 4) == 0) {
            is_cuda = true;
        }
    }

    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS] Pre-computing weights (FP32 conversion + dilated convolutions)..." << std::endl;
    
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
    
    // 1. If non-CUDA (CPU/SYCL), pre-convert loaded GGUF FP16 weights to FP32
    struct UploadF32Entry {
        std::string name;
        std::vector<float> data;
    };
    std::vector<UploadF32Entry> fp32_upload_list;
    std::vector<struct ggml_tensor*> fp32_tensors_list;

    if (!is_cuda) {
        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS load] Non-CUDA backend detected. Pre-converting FP16 weights to FP32..." << std::endl;
        for (const auto& pair : tensors) {
            struct ggml_tensor* old_w = pair.second;
            if (!old_w || old_w->type != GGML_TYPE_F16) continue;

            // Skip embedding and codebook tensors to let get_rows run on FP16 (as SYCL get_rows crashes on FP32)
            if (pair.first.find("embedding") != std::string::npos || pair.first.find("embed") != std::string::npos) {
                continue;
            }

            int64_t w_elems = ggml_nelements(old_w);
            std::vector<uint8_t> w_bytes(ggml_nbytes(old_w));
            ggml_backend_tensor_get(old_w, w_bytes.data(), 0, w_bytes.size());

            std::vector<float> w_f32_data(w_elems);
            const ggml_fp16_t* ptr = (const ggml_fp16_t*)w_bytes.data();
            for (int64_t i = 0; i < w_elems; ++i) {
                w_f32_data[i] = ggml_fp16_to_fp32(ptr[i]);
            }

            struct ggml_tensor* new_w = ggml_new_tensor(vits_custom_ctx, GGML_TYPE_F32, ggml_n_dims(old_w), old_w->ne);
            ggml_set_name(new_w, old_w->name);

            fp32_tensors_list.push_back(new_w);
            fp32_upload_list.push_back({pair.first, w_f32_data});
        }
    }

    // 2. Pre-compute dilated convolution weights (as FP16 for CUDA, FP32 for CPU/SYCL)
    std::vector<std::pair<std::string, std::vector<ggml_fp16_t>>> dilated_fp16_data_list;
    std::vector<std::pair<std::string, std::vector<float>>> dilated_fp32_data_list;
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
            
            std::vector<uint8_t> w_bytes(ggml_nbytes(old_w));
            ggml_backend_tensor_get(old_w, w_bytes.data(), 0, w_bytes.size());
            
            std::vector<float> w_host(w_elems);
            if (old_w->type == GGML_TYPE_F16) {
                const ggml_fp16_t* ptr = (const ggml_fp16_t*)w_bytes.data();
                for (int64_t i = 0; i < w_elems; ++i) {
                    w_host[i] = ggml_fp16_to_fp32(ptr[i]);
                }
            } else {
                const float* ptr = (const float*)w_bytes.data();
                std::copy(ptr, ptr + w_elems, w_host.begin());
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
            
            if (is_cuda) {
                std::vector<ggml_fp16_t> w_dilated_fp16(new_w_elems);
                for (int64_t i = 0; i < new_w_elems; ++i) {
                    w_dilated_fp16[i] = ggml_fp32_to_fp16(w_dilated_host[i]);
                }
                struct ggml_tensor* new_w = ggml_new_tensor_3d(vits_custom_ctx, GGML_TYPE_F16, new_kw, ic, oc);
                dilated_tensors_list.push_back(new_w);
                dilated_fp16_data_list.push_back({prefix + ".weight_dilated", w_dilated_fp16});
            } else {
                struct ggml_tensor* new_w = ggml_new_tensor_3d(vits_custom_ctx, GGML_TYPE_F32, new_kw, ic, oc);
                dilated_tensors_list.push_back(new_w);
                dilated_fp32_data_list.push_back({prefix + ".weight_dilated", w_dilated_host});
            }
        }
    }
    
    // Allocate all custom tensors on the backend
    vits_custom_buf = ggml_backend_alloc_ctx_tensors(vits_custom_ctx, backend);
    if (!vits_custom_buf) {
        std::cerr << "[VITS load] Error: Failed to allocate vits_custom_buf!" << std::endl;
        return false;
    }
    
    // Upload F32 converted weights (non-CUDA only)
    if (!is_cuda) {
        for (size_t i = 0; i < fp32_tensors_list.size(); ++i) {
            struct ggml_tensor* nt = fp32_tensors_list[i];
            const auto& upload_entry = fp32_upload_list[i];
            ggml_backend_tensor_set(nt, upload_entry.data.data(), 0, upload_entry.data.size() * sizeof(float));
            tensors[upload_entry.name] = nt;
        }
    }

    // Upload dilated weights
    if (is_cuda) {
        for (size_t i = 0; i < dilated_tensors_list.size(); ++i) {
            struct ggml_tensor* nt = dilated_tensors_list[i];
            const auto& name_and_data = dilated_fp16_data_list[i];
            ggml_backend_tensor_set(nt, name_and_data.second.data(), 0, name_and_data.second.size() * sizeof(ggml_fp16_t));
            tensors[name_and_data.first] = nt;
        }
    } else {
        for (size_t i = 0; i < dilated_tensors_list.size(); ++i) {
            struct ggml_tensor* nt = dilated_tensors_list[i];
            const auto& name_and_data = dilated_fp32_data_list[i];
            ggml_backend_tensor_set(nt, name_and_data.second.data(), 0, name_and_data.second.size() * sizeof(float));
            tensors[name_and_data.first] = nt;
        }
    }

    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS load] Synchronizing backend to verify upload..." << std::endl;
    ggml_backend_synchronize(backend);
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS] Static FP32 conversion + dilated weights uploaded successfully!" << std::endl;

    return true;
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

    // Direct heap-free optimized convolution loop
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
                    // Contiguous SIMD-friendly AXPY loop!
                    // 0 <= t + k_offset < seq_len => -k_offset <= t < seq_len - k_offset
                    int64_t t_start = std::max((int64_t)0, -k_offset);
                    int64_t t_end = std::min(out_seq_len, seq_len - k_offset);

                    if (t_start < t_end) {
                        const float* __restrict x_ic_start = x_ic + t_start + k_offset;
                        float* __restrict dst_oc_start = dst_oc + t_start;
                        int64_t len = t_end - t_start;

#if defined(__AVX2__)
                        // AVX2 optimized AXPY path
                        __m256 w_vec = _mm256_set1_ps(w_val);
                        int64_t t_inner = 0;
                        
                        // Process 32 elements at a time (loop unrolling for maximum instruction throughput)
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

                        // Process remaining 8 elements at a time
                        for (; t_inner <= len - 8; t_inner += 8) {
                            __m256 x_vec = _mm256_loadu_ps(x_ic_start + t_inner);
                            __m256 d_vec = _mm256_loadu_ps(dst_oc_start + t_inner);
                            d_vec = _mm256_fmadd_ps(w_vec, x_vec, d_vec);
                            _mm256_storeu_ps(dst_oc_start + t_inner, d_vec);
                        }

                        // Remainder
                        for (; t_inner < len; ++t_inner) {
                            dst_oc_start[t_inner] += w_val * x_ic_start[t_inner];
                        }
#else
                        // Non-AVX2 fallback
                        for (int64_t t_inner = 0; t_inner < len; ++t_inner) {
                            dst_oc_start[t_inner] += w_val * x_ic_start[t_inner];
                        }
#endif
                    }
                } else {
                    // Fallback for stride > 1 (highly robust)
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
    if (std::getenv("GPT_SOVITS_DEBUG") != nullptr) {
        std::cout << "[im2col_f32 Start] x shape: [" << x->ne[0] << ", " << x->ne[1] << ", " << x->ne[2] << ", " << x->ne[3] << "]"
                  << " | w shape: [" << w->ne[0] << ", " << w->ne[1] << ", " << w->ne[2] << ", " << w->ne[3] << "]" << std::endl;
    }

    // Cast weights to F32 once dynamically in graph to bypass on-the-fly FP16 dequantization in mul_mat
    struct ggml_tensor* w_f32 = force_w_f32(ctx, w);
    
    // Optimized path for 1x1 convolutions: bypass im2col completely!
    if (w->ne[0] == 1 && stride == 1 && dilation == 1 && padding == 0) {
        // Reshape weights: [1, in_channels, out_channels] -> [in_channels, out_channels]
        struct ggml_tensor* w_reshaped = ggml_reshape_2d(ctx, w_f32, w_f32->ne[1], w_f32->ne[2]);
        
        // Transpose and make contiguous: [seq_len, in_channels] -> [in_channels, seq_len]
        struct ggml_tensor* x_t = ggml_cont(ctx, ggml_transpose(ctx, x));
        
        // Matrix multiply: [in_channels, seq_len] x [in_channels, out_channels] -> [seq_len, out_channels]
        struct ggml_tensor* result = ggml_mul_mat(ctx, x_t, w_reshaped);
        
        // The output of ggml_mul_mat already has shape [seq_len, out_channels] (e.g. [474, 192]).
        // No transpose is needed. Make contiguous directly.
        struct ggml_tensor* final_res = ggml_cont(ctx, result);

        if (std::getenv("GPT_SOVITS_DEBUG") != nullptr) {
            std::cout << "[im2col_f32 1x1 Bypass] w_reshaped: [" << w_reshaped->ne[0] << ", " << w_reshaped->ne[1] << "]"
                      << " | x_t: [" << x_t->ne[0] << ", " << x_t->ne[1] << "]"
                      << " | result: [" << result->ne[0] << ", " << result->ne[1] << "]"
                      << " | final_res: [" << final_res->ne[0] << ", " << final_res->ne[1] << "]" << std::endl;
        }
        return final_res;
    }

    // Perform im2col into a pure F32 representation
    struct ggml_tensor* im2col = ggml_im2col(ctx, w_f32, x, stride, 0, padding, 0, dilation, 0, false, GGML_TYPE_F32);
    
    if (std::getenv("GPT_SOVITS_DEBUG") != nullptr) {
        if (w_f32) std::cout << "[im2col Debug] w_f32: [" << w_f32->ne[0] << ", " << w_f32->ne[1] << ", " << w_f32->ne[2] << ", " << w_f32->ne[3] << "]" << std::endl;
        if (x) std::cout << "[im2col Debug] x: [" << x->ne[0] << ", " << x->ne[1] << ", " << x->ne[2] << ", " << x->ne[3] << "]" << std::endl;
        if (im2col) std::cout << "[im2col Debug] im2col: [" << im2col->ne[0] << ", " << im2col->ne[1] << ", " << im2col->ne[2] << ", " << im2col->ne[3] << "], nelements=" << ggml_nelements(im2col) << std::endl;
    }

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
        // Under Intel SYCL with oneDNN enabled, oneDNN's matmul descriptor requires standard contiguous strides.
        // Reshaping changes strides without copy, which crashes oneDNN. We force contiguity on GPU memory.
        if (!ggml_is_contiguous(im2col_reshaped)) {
            im2col_reshaped = ggml_cont(ctx, im2col_reshaped);
        }
        if (!ggml_is_contiguous(w_reshaped)) {
            w_reshaped = ggml_cont(ctx, w_reshaped);
        }
    }
    
    // Multiply F32 im2col with F32 weights to leverage super-optimized AVX2 FP32 matrix-multiplication assembly
    struct ggml_tensor* result = ggml_mul_mat(ctx, im2col_reshaped, w_reshaped);
    
    struct ggml_tensor* final_res = ggml_reshape_2d(ctx, result, im2col->ne[1], w_f32->ne[2]);

    if (std::getenv("GPT_SOVITS_DEBUG") != nullptr) {
        std::cout << "[im2col_f32 Standard] im2col_reshaped: [" << im2col_reshaped->ne[0] << ", " << im2col_reshaped->ne[1] << "]"
                  << " | w_reshaped: [" << w_reshaped->ne[0] << ", " << w_reshaped->ne[1] << "]"
                  << " | result: [" << result->ne[0] << ", " << result->ne[1] << "]"
                  << " | final_res: [" << final_res->ne[0] << ", " << final_res->ne[1] << "]" << std::endl;
    }

    return final_res;
}

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

static struct ggml_tensor* custom_conv_transpose_1d(
    struct ggml_context* ctx,
    struct ggml_tensor* w,
    struct ggml_tensor* x,
    int stride,
    int padding,
    int dilation
) {
    GGML_ASSERT(dilation == 1);
    
    // 1. Force weights to float32
    struct ggml_tensor* w_f32 = force_w_f32(ctx, w);
    
    int64_t kernel_size = w_f32->ne[0];
    int64_t out_channels = w_f32->ne[1];
    int64_t in_channels = w_f32->ne[2];
    int64_t seq_len = x->ne[0];
    
    // 2. Permute weights to [out_channels, in_channels, kernel_size]
    struct ggml_tensor* w_perm = ggml_cont(ctx, ggml_permute(ctx, w_f32, 2, 0, 1, 3));
    
    // 3. For each remainder r in [0, stride - 1], compute the sub-sequence
    int64_t L_max = seq_len + (kernel_size - 1) / stride;
    
    std::vector<struct ggml_tensor*> sub_seqs;
    for (int r = 0; r < stride; ++r) {
        int64_t M_r = (kernel_size - 1 - r) / stride + 1;
        if (M_r <= 0) {
            // No elements, fall back to zeros
            struct ggml_tensor* zero_seq = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, out_channels, 1, L_max);
            sub_seqs.push_back(zero_seq);
            continue;
        }
        
        // Extract sliced weights for remainder r: [M_r, in_channels, out_channels]
        std::vector<struct ggml_tensor*> slices;
        for (int m = M_r - 1; m >= 0; --m) {
            int k = m * stride + r;
            struct ggml_tensor* w_k = ggml_view_2d(ctx, w_perm, out_channels, in_channels, w_perm->nb[1], k * w_perm->nb[2]);
            struct ggml_tensor* w_k_t = ggml_cont(ctx, ggml_transpose(ctx, w_k));
            struct ggml_tensor* w_k_3d = ggml_reshape_3d(ctx, w_k_t, 1, in_channels, out_channels);
            slices.push_back(w_k_3d);
        }
        
        struct ggml_tensor* W_r_reversed = slices[0];
        for (size_t idx = 1; idx < slices.size(); ++idx) {
            W_r_reversed = ggml_concat(ctx, W_r_reversed, slices[idx], 0);
        }
        
        // Run convolution
        struct ggml_tensor* y_r = custom_conv_1d(ctx, W_r_reversed, x, 1, M_r - 1, 1);
        struct ggml_tensor* y_r_t = ggml_cont(ctx, ggml_transpose(ctx, y_r));
        
        // Pad to L_max on sequence dimension
        int64_t L_r = seq_len + M_r - 1;
        int64_t pad_right = L_max - L_r;
        struct ggml_tensor* y_r_padded = y_r_t;
        if (pad_right > 0) {
            y_r_padded = ggml_pad_ext(ctx, y_r_t, 0, 0, 0, pad_right, 0, 0, 0, 0);
        }
        
        struct ggml_tensor* y_r_3d = ggml_reshape_3d(ctx, y_r_padded, out_channels, 1, L_max);
        sub_seqs.push_back(y_r_3d);
    }
    
    // 4. Concatenate along dimension 1 (the interleaved dimension)
    struct ggml_tensor* y_stacked = sub_seqs[0];
    for (size_t r = 1; r < sub_seqs.size(); ++r) {
        y_stacked = ggml_concat(ctx, y_stacked, sub_seqs[r], 1);
    }
    
    struct ggml_tensor* y_interleaved = ggml_reshape_2d(ctx, y_stacked, out_channels, stride * L_max);
    
    // 5. Crop to out_seq_len
    int64_t out_seq_len = (seq_len - 1) * stride + kernel_size;
    struct ggml_tensor* y_cropped = ggml_view_2d(ctx, y_interleaved, out_channels, out_seq_len, y_interleaved->nb[1], 0);
    struct ggml_tensor* y_cropped_cont = ggml_cont(ctx, y_cropped);
    
    struct ggml_tensor* output = ggml_cont(ctx, ggml_transpose(ctx, y_cropped_cont));
    return output;
}

static struct ggml_tensor* ggml_conv_1d_vits(
    struct ggml_context* ctx,
    struct ggml_tensor* w,
    struct ggml_tensor* x,
    int stride,
    int padding,
    int dilation
) {
    bool is_cuda = false;
    bool is_sycl = false;
    
    if (current_vits_backend) {
        const char * bname = ggml_backend_name(current_vits_backend);
        if (bname) {
            if (strncmp(bname, "CUDA", 4) == 0) {
                is_cuda = true;
            } else if (strncmp(bname, "SYCL", 4) == 0) {
                is_sycl = true;
            }
        }
    }

    if (std::getenv("GPT_SOVITS_DEBUG") != nullptr) {
        const char* bname = current_vits_backend ? ggml_backend_name(current_vits_backend) : "nullptr";
        std::cout << "[ggml_conv_1d_vits] current_vits_backend: " << (void*)current_vits_backend 
                  << " | name: " << bname << " | is_cuda: " << is_cuda << " | is_sycl: " << is_sycl << std::endl;
    }

    if (is_cuda) {
        return ggml_conv_1d_cudnn(ctx, w, x, stride, padding, dilation);
    } else if (is_sycl) {
        return custom_conv_1d(ctx, w, x, stride, padding, dilation);
    } else {
        return ggml_conv_1d_im2col_f32(ctx, w, x, stride, padding, dilation);
    }
}

static struct ggml_tensor* ggml_conv_transpose_1d_vits(
    struct ggml_context* ctx,
    struct ggml_tensor* w,
    struct ggml_tensor* x,
    int stride,
    int padding,
    int dilation
) {
    bool is_cuda = false;
    bool is_sycl = false;
    if (current_vits_backend) {
        const char * bname = ggml_backend_name(current_vits_backend);
        if (bname) {
            if (strncmp(bname, "CUDA", 4) == 0) {
                is_cuda = true;
            } else if (strncmp(bname, "SYCL", 4) == 0) {
                is_sycl = true;
            }
        }
    }
    if (is_cuda) {
        return ggml_conv_transpose_1d_cudnn(ctx, w, x, stride, padding, dilation);
    }
    if (is_sycl) {
        return custom_conv_transpose_1d(ctx, w, x, stride, padding, dilation);
    }
    // Cast weights to F32 as the native conv_transpose_1d operator only supports GGML_TYPE_F32 weights.
    struct ggml_tensor* w_f32 = force_w_f32(ctx, w);
    return ggml_conv_transpose_1d(ctx, w_f32, x, stride, padding, dilation);
}

// Helper to construct 1D convolution with bias in GGML
static struct ggml_tensor* ggml_conv_1d_with_bias(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* w,
    struct ggml_tensor* b,
    int stride,
    int dilation,
    int padding
) {
    if (std::getenv("GPT_SOVITS_DEBUG") != nullptr) {
        std::cout << "[Conv1d Debug] Original x shape: [" << x->ne[0] << ", " << x->ne[1] << ", " << x->ne[2] << ", " << x->ne[3] << "]"
                  << " | w shape: [" << w->ne[0] << ", " << w->ne[1] << ", " << w->ne[2] << ", " << w->ne[3] << "]" << std::endl;
    }
              
    // x shape: [in_channels, seq_len] -> transpose to [seq_len, in_channels] for ggml_conv_1d
    struct ggml_tensor* x_transposed = ggml_cont(ctx, ggml_transpose(ctx, x));

    // w shape: [kernel_size, in_channels, out_channels]
    // conv output shape: [out_frames, out_channels]
    // Uses cuDNN-accelerated or CPU native 1D convolution
    struct ggml_tensor* conv = ggml_conv_1d_vits(ctx, w, x_transposed, stride, padding, dilation);

    // Transpose conv back to [out_channels, out_frames]
    struct ggml_tensor* conv_transposed = ggml_cont(ctx, ggml_transpose(ctx, conv));

    // Reshape bias to be broadcastable along the sequence dimension: [out_channels, 1]
    struct ggml_tensor* b_reshaped = ggml_reshape_2d(ctx, b, b->ne[0], 1);

    if (std::getenv("GPT_SOVITS_DEBUG") != nullptr) {
        std::cout << "[Conv1d Debug] conv shape: [" << conv->ne[0] << ", " << conv->ne[1] << ", " << conv->ne[2] << ", " << conv->ne[3] << "]"
                  << " | conv_transposed shape: [" << conv_transposed->ne[0] << ", " << conv_transposed->ne[1] << ", " << conv_transposed->ne[2] << ", " << conv_transposed->ne[3] << "]"
                  << " | b shape: [" << b->ne[0] << ", " << b->ne[1] << ", " << b->ne[2] << ", " << b->ne[3] << "]"
                  << " | b_reshaped shape: [" << b_reshaped->ne[0] << ", " << b_reshaped->ne[1] << ", " << b_reshaped->ne[2] << ", " << b_reshaped->ne[3] << "]" << std::endl;
    }

    return ggml_add(ctx, conv_transposed, b_reshaped);
}

// Helper to construct 1D Transposed Convolution with bias in GGML
static struct ggml_tensor* ggml_conv_transpose_1d_with_bias(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* w,
    struct ggml_tensor* b,
    int stride,
    int padding
) {
    if (std::getenv("GPT_SOVITS_DEBUG") != nullptr) {
        std::cout << "[ConvTranspose1d Debug] Original x shape: [" << x->ne[0] << ", " << x->ne[1] << ", " << x->ne[2] << ", " << x->ne[3] << "]"
                  << " | w shape: [" << w->ne[0] << ", " << w->ne[1] << ", " << w->ne[2] << ", " << w->ne[3] << "]" << std::endl;
    }
    // x shape: [in_channels, seq_len] -> transpose to [seq_len, in_channels]
    struct ggml_tensor* x_transposed = ggml_cont(ctx, ggml_transpose(ctx, x));

    // cuDNN-accelerated transposed conv handles padding/dilation natively.
    // However, other GGML backends (CPU, SYCL, etc.) do not support padding != 0 for 1D transposed convolution natively.
    // So on those backends, if padding > 0, we run transposed convolution with padding = 0, and then crop the padding from both ends of the output.
    bool is_cuda = false;
    if (current_vits_backend) {
        const char * bname = ggml_backend_name(current_vits_backend);
        if (bname && strncmp(bname, "CUDA", 4) == 0) {
            is_cuda = true;
        }
    }
    struct ggml_tensor* conv_t;
    if (!is_cuda && padding > 0) {
        // Run with padding = 0 on non-CUDA backends
        struct ggml_tensor* conv_t_raw = ggml_conv_transpose_1d_vits(ctx, w, x_transposed, stride, 0, 1);
        
        // Crop the sequence dimension (ne[0]) by 'padding' from both ends
        int64_t cropped_seq_len = conv_t_raw->ne[0] - 2 * padding;
        size_t offset_bytes = padding * conv_t_raw->nb[0];
        
        struct ggml_tensor* cropped_view = ggml_view_2d(
            ctx,
            conv_t_raw,
            cropped_seq_len,
            conv_t_raw->ne[1],
            conv_t_raw->nb[1],
            offset_bytes
        );
        
        // Make the view contiguous
        conv_t = ggml_cont(ctx, cropped_view);
    } else {
        conv_t = ggml_conv_transpose_1d_vits(ctx, w, x_transposed, stride, padding, 1);
    }

    // Transpose conv_t back to [out_channels, out_seq_len]
    struct ggml_tensor* conv_t_transposed = ggml_cont(ctx, ggml_transpose(ctx, conv_t));

    struct ggml_tensor* b_reshaped = ggml_reshape_2d(ctx, b, b->ne[0], 1);
    return ggml_add(ctx, conv_t_transposed, b_reshaped);
}

// Helper to construct 1D convolution with bias in GGML without transposes (expects x to be [seq_len, in_channels])
static struct ggml_tensor* ggml_conv_1d_with_bias_no_transpose(
    struct ggml_context* ctx,
    struct ggml_tensor* x,      // [seq_len, in_channels]
    struct ggml_tensor* w,      // [kernel_size, in_channels, out_channels]
    struct ggml_tensor* b,      // [out_channels]
    int stride,
    int dilation,
    int padding
) {
    if (std::getenv("GPT_SOVITS_DEBUG") != nullptr) {
        std::cout << "[Conv1d NoTranspose Debug] x shape: [" << x->ne[0] << ", " << x->ne[1] << ", " << x->ne[2] << ", " << x->ne[3] << "]"
                  << " | w shape: [" << w->ne[0] << ", " << w->ne[1] << ", " << w->ne[2] << ", " << w->ne[3] << "]" << std::endl;
    }

    struct ggml_tensor* conv = ggml_conv_1d_vits(ctx, w, x, stride, padding, dilation);

    // Reshape bias to be broadcastable along the sequence dimension: [1, out_channels]
    struct ggml_tensor* b_reshaped = ggml_reshape_2d(ctx, b, 1, b->ne[0]);
    return ggml_add(ctx, conv, b_reshaped);
}

// Helper to construct 1D Transposed Convolution with bias in GGML without transposes (expects x to be [seq_len, in_channels])
static struct ggml_tensor* ggml_conv_transpose_1d_with_bias_no_transpose(
    struct ggml_context* ctx,
    struct ggml_tensor* x,      // [seq_len, in_channels]
    struct ggml_tensor* w,      // [kernel_size, out_channels, in_channels]
    struct ggml_tensor* b,      // [out_channels]
    int stride,
    int padding
) {
    if (std::getenv("GPT_SOVITS_DEBUG") != nullptr) {
        std::cout << "[ConvTranspose1d NoTranspose Debug] x shape: [" << x->ne[0] << ", " << x->ne[1] << ", " << x->ne[2] << ", " << x->ne[3] << "]"
                  << " | w shape: [" << w->ne[0] << ", " << w->ne[1] << ", " << w->ne[2] << ", " << w->ne[3] << "]" << std::endl;
    }

    bool is_cuda = false;
    if (current_vits_backend) {
        const char * bname = ggml_backend_name(current_vits_backend);
        if (bname && strncmp(bname, "CUDA", 4) == 0) {
            is_cuda = true;
        }
    }
    struct ggml_tensor* conv_t;
    if (!is_cuda && padding > 0) {
        // Run with padding = 0 on non-CUDA backends
        struct ggml_tensor* conv_t_raw = ggml_conv_transpose_1d_vits(ctx, w, x, stride, 0, 1);
        
        // Crop the sequence dimension (ne[0]) by 'padding' from both ends
        int64_t cropped_seq_len = conv_t_raw->ne[0] - 2 * padding;
        size_t offset_bytes = padding * conv_t_raw->nb[0];
        
        struct ggml_tensor* cropped_view = ggml_view_2d(
            ctx,
            conv_t_raw,
            cropped_seq_len,
            conv_t_raw->ne[1],
            conv_t_raw->nb[1],
            offset_bytes
        );
        
        // Make the view contiguous
        conv_t = ggml_cont(ctx, cropped_view);
    } else {
        conv_t = ggml_conv_transpose_1d_vits(ctx, w, x, stride, padding, 1);
    }

    struct ggml_tensor* b_reshaped = ggml_reshape_2d(ctx, b, 1, b->ne[0]);
    return ggml_add(ctx, conv_t, b_reshaped);
}

// Multi-Receptive Field Fusion (MRF) Residual Block for BigVGAN (Optimized: operates entirely on [seq_len, channels] layout)
static struct ggml_tensor* mrf_resblock_no_transpose(
    struct ggml_context* ctx,
    struct ggml_tensor* x,      // [seq_len, channels]
    VITSModel& model,
    int block_idx,
    int channels,
    int kernel_size,
    const std::vector<int>& dilations
) {
    struct ggml_tensor* current_x = x;

    for (int l = 0; l < 3; ++l) {
        int dilation = dilations[l];
        int padding = (kernel_size - 1) * dilation / 2;

        // Retrieve weights
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

        // xt = LeakyReLU(current_x, 0.1)
        struct ggml_tensor* xt = ggml_leaky_relu(ctx, current_x, 0.1f, false);

        // xt = convs1[l](xt)
        xt = ggml_conv_1d_with_bias_no_transpose(ctx, xt, c1_w, c1_b, 1, dilation_effective, padding);
        if (block_idx == 0 && std::getenv("GPT_SOVITS_DEBUG") != nullptr) {
            model.debug_res0_convs1[l] = ggml_cont(ctx, ggml_transpose(ctx, xt));
        }

        // xt = LeakyReLU(xt, 0.1)
        xt = ggml_leaky_relu(ctx, xt, 0.1f, false);

        // xt = convs2[l](xt)
        xt = ggml_conv_1d_with_bias_no_transpose(ctx, xt, c2_w, c2_b, 1, 1, (kernel_size - 1) / 2);
        if (block_idx == 0 && std::getenv("GPT_SOVITS_DEBUG") != nullptr) {
            model.debug_res0_convs2[l] = ggml_cont(ctx, ggml_transpose(ctx, xt));
        }

        // current_x = current_x + xt
        current_x = ggml_add(ctx, xt, current_x);
    }

    return current_x;
}

// Multi-Receptive Field Fusion (MRF) Residual Block for BigVGAN
static struct ggml_tensor* mrf_resblock(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    VITSModel& model,
    int block_idx,
    int channels,
    int kernel_size,
    const std::vector<int>& dilations
) {
    struct ggml_tensor* current_x = x;

    for (int l = 0; l < 3; ++l) {
        int dilation = dilations[l];
        int padding = (kernel_size - 1) * dilation / 2;

        // Retrieve weights
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

        // xt = LeakyReLU(current_x, 0.1)
        struct ggml_tensor* xt = ggml_leaky_relu(ctx, current_x, 0.1f, false);

        // xt = convs1[l](xt)
        xt = ggml_conv_1d_with_bias(ctx, xt, c1_w, c1_b, 1, dilation_effective, padding);
        if (block_idx == 0) {
            model.debug_res0_convs1[l] = xt;
        }

        // xt = LeakyReLU(xt, 0.1)
        xt = ggml_leaky_relu(ctx, xt, 0.1f, false);

        // xt = convs2[l](xt)
        xt = ggml_conv_1d_with_bias(ctx, xt, c2_w, c2_b, 1, 1, (kernel_size - 1) / 2);
        if (block_idx == 0) {
            model.debug_res0_convs2[l] = xt;
    }

        // current_x = current_x + xt
        current_x = ggml_add(ctx, xt, current_x);
    }

    return current_x;
}

static struct ggml_tensor* build_vits_generator(
    struct ggml_context* ctx_graph,
    struct ggml_tensor* latent,
    struct ggml_tensor* speaker_embedding,
    VITSModel& model
) {
    struct ggml_tensor* dec_conv_pre_w = model.get_tensor("dec.conv_pre.weight");
    struct ggml_tensor* dec_conv_pre_b = model.get_tensor("dec.conv_pre.bias");
    if (!dec_conv_pre_w || !dec_conv_pre_b) {
        std::cerr << "[VITS] Error: Missing dec.conv_pre weights!" << std::endl;
        return nullptr;
    }

    // 1. Transpose latent once from [channels, seq_len] to [seq_len, channels]
    struct ggml_tensor* latent_transposed = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, latent));

    // 2. Initial convolution on [seq_len, channels] input
    struct ggml_tensor* h = ggml_conv_1d_with_bias_no_transpose(ctx_graph, latent_transposed, dec_conv_pre_w, dec_conv_pre_b, 1, 1, 3);
    
    // Debug assignment (transposed back to original shape for alignment tests)
    model.debug_conv_pre = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, h));

    // 3. Add speaker embedding projection if provided
    if (speaker_embedding != nullptr) {
        struct ggml_tensor* cond_w = model.get_tensor("dec.cond.weight");
        struct ggml_tensor* cond_b = model.get_tensor("dec.cond.bias");
        if (cond_w && cond_b) {
            // speaker_embedding is [256, 1]. conv_1d expects [seq_len, channels] so [1, 256].
            struct ggml_tensor* g_proj_t = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, speaker_embedding));
            struct ggml_tensor* g_proj = ggml_conv_1d_with_bias_no_transpose(ctx_graph, g_proj_t, cond_w, cond_b, 1, 1, 0); // [1, channels]
            
            // Debug assignment (transposed back to [channels, 1])
            model.debug_cond = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, g_proj));
            
            if (std::getenv("GPT_SOVITS_DEBUG") != nullptr) {
                std::cout << "[VITS Debug] speaker_embedding shape: [" << speaker_embedding->ne[0] << ", " << speaker_embedding->ne[1] << ", " << speaker_embedding->ne[2] << "]"
                          << " | cond_w shape: [" << cond_w->ne[0] << ", " << cond_w->ne[1] << ", " << cond_w->ne[2] << "]"
                          << " | g_proj shape: [" << g_proj->ne[0] << ", " << g_proj->ne[1] << ", " << g_proj->ne[2] << "]"
                          << " | h shape: [" << h->ne[0] << ", " << h->ne[1] << ", " << h->ne[2] << "]" << std::endl;
            }
            
            // Natively broadcast-add [1, channels] speaker embedding to [seq_len, channels] h!
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
        h = ggml_conv_transpose_1d_with_bias_no_transpose(ctx_graph, h, ups0_w, ups0_b, 10, 3);
        if (std::getenv("GPT_SOVITS_DEBUG") != nullptr) {
            model.debug_ups[0] = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, h));
        }
        
        struct ggml_tensor* r0 = mrf_resblock_no_transpose(ctx_graph, h, model, 0, 256, 3, dilations);
        struct ggml_tensor* r1 = mrf_resblock_no_transpose(ctx_graph, h, model, 1, 256, 7, dilations);
        struct ggml_tensor* r2 = mrf_resblock_no_transpose(ctx_graph, h, model, 2, 256, 11, dilations);
        
        if (std::getenv("GPT_SOVITS_DEBUG") != nullptr) {
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
        h = ggml_conv_transpose_1d_with_bias_no_transpose(ctx_graph, h, ups1_w, ups1_b, 8, 4);
        if (std::getenv("GPT_SOVITS_DEBUG") != nullptr) {
            model.debug_ups[1] = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, h));
        }
        
        struct ggml_tensor* r3 = mrf_resblock_no_transpose(ctx_graph, h, model, 3, 128, 3, dilations);
        struct ggml_tensor* r4 = mrf_resblock_no_transpose(ctx_graph, h, model, 4, 128, 7, dilations);
        struct ggml_tensor* r5 = mrf_resblock_no_transpose(ctx_graph, h, model, 5, 128, 11, dilations);
        
        if (std::getenv("GPT_SOVITS_DEBUG") != nullptr) {
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
        h = ggml_conv_transpose_1d_with_bias_no_transpose(ctx_graph, h, ups2_w, ups2_b, 2, 3);
        if (std::getenv("GPT_SOVITS_DEBUG") != nullptr) {
            model.debug_ups[2] = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, h));
        }
        
        struct ggml_tensor* r6 = mrf_resblock_no_transpose(ctx_graph, h, model, 6, 64, 3, dilations);
        struct ggml_tensor* r7 = mrf_resblock_no_transpose(ctx_graph, h, model, 7, 64, 7, dilations);
        struct ggml_tensor* r8 = mrf_resblock_no_transpose(ctx_graph, h, model, 8, 64, 11, dilations);
        
        if (std::getenv("GPT_SOVITS_DEBUG") != nullptr) {
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
        h = ggml_conv_transpose_1d_with_bias_no_transpose(ctx_graph, h, ups3_w, ups3_b, 2, 0);
        if (std::getenv("GPT_SOVITS_DEBUG") != nullptr) {
            model.debug_ups[3] = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, h));
        }
        
        struct ggml_tensor* r9 = mrf_resblock_no_transpose(ctx_graph, h, model, 9, 32, 3, dilations);
        struct ggml_tensor* r10 = mrf_resblock_no_transpose(ctx_graph, h, model, 10, 32, 7, dilations);
        struct ggml_tensor* r11 = mrf_resblock_no_transpose(ctx_graph, h, model, 11, 32, 11, dilations);
        
        if (std::getenv("GPT_SOVITS_DEBUG") != nullptr) {
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
        h = ggml_conv_transpose_1d_with_bias_no_transpose(ctx_graph, h, ups4_w, ups4_b, 2, 0);
        if (std::getenv("GPT_SOVITS_DEBUG") != nullptr) {
            model.debug_ups[4] = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, h));
        }
        
        struct ggml_tensor* r12 = mrf_resblock_no_transpose(ctx_graph, h, model, 12, 16, 3, dilations);
        struct ggml_tensor* r13 = mrf_resblock_no_transpose(ctx_graph, h, model, 13, 16, 7, dilations);
        struct ggml_tensor* r14 = mrf_resblock_no_transpose(ctx_graph, h, model, 14, 16, 11, dilations);
        
        if (std::getenv("GPT_SOVITS_DEBUG") != nullptr) {
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

    // Final LeakyReLU (slope=0.01) before conv_post
    h = ggml_leaky_relu(ctx_graph, h, 0.01f, false);

    // Run final convolution in [seq_len, channels] layout
    struct ggml_tensor* conv = ggml_conv_1d_vits(ctx_graph, conv_post_w, h, 1, 3, 1); // [out_seq_len, 1]

    // Transpose conv back to [1, out_seq_len] (so the audio vector layout is correct)
    struct ggml_tensor* audio = ggml_cont(ctx_graph, ggml_transpose(ctx_graph, conv));
    model.debug_conv_post = audio;
    return ggml_tanh(ctx_graph, audio);
}

struct ggml_tensor* VITSModel::forward_from_latent(
    struct ggml_context* ctx_graph,
    struct ggml_tensor* latent,
    struct ggml_tensor* speaker_embedding,
    ggml_backend_t backend
) {
    current_vits_backend = backend;
    g_conv_1d_direct_params_pool.clear();
    return build_vits_generator(ctx_graph, latent, speaker_embedding, *this);
}

static struct ggml_tensor* ggml_mish(
    struct ggml_context* ctx,
    struct ggml_tensor* x
) {
    // Ensure x is FP32 (must match type for element-wise ops)
    struct ggml_tensor* x_f32 = (x->type == GGML_TYPE_F32) ? x : ggml_cont(ctx, ggml_cast(ctx, x, GGML_TYPE_F32));
    struct ggml_tensor* exp_x = ggml_exp(ctx, x_f32);
    struct ggml_tensor* ones = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, x_f32->ne[0], x_f32->ne[1]);
    ones = ggml_fill(ctx, ones, 1.0f);
    struct ggml_tensor* sp = ggml_log(ctx, ggml_add(ctx, exp_x, ones));
    return ggml_mul(ctx, x_f32, ggml_tanh(ctx, sp));
}

// Helper: Linear layer (matmul + bias) — weight cast to FP32 if needed (cuBLAS FP16 gemm not supported on all GPUs)
// x: [T, in_features] meaning ne0=T, ne1=in_features (T rows, in_features cols)
// w: GGUF loaded as ne0=in_features, ne1=out_features
// gm(a,b) = a^T * b. For Linear: w^T * x → [out×in] * [in×T] = [out×T]
static struct ggml_tensor* ggml_linear(
    struct ggml_context* ctx,
    struct ggml_tensor* x,     // ne0=in_features, ne1=T
    struct ggml_tensor* w,     // ne0=in_features, ne1=out_features (or 3D: [k=1, in, out])
    struct ggml_tensor* b      // [out_features] or nullptr
) {
    // Handle 3D Conv1d weight [kernel=1, in, out] → reshape to 2D [in, out]
    if (w->ne[2] > 1 && w->ne[0] == 1) {
        w = ggml_cont(ctx, ggml_reshape_2d(ctx, w, w->ne[1], w->ne[2]));  // [in, out]
    }
    // Cast weight to FP32 for CUDA cuBLAS compatibility
    struct ggml_tensor* w_f32 = force_w_f32(ctx, w);
    struct ggml_tensor* out = ggml_mul_mat(ctx, w_f32, x);
    ggml_mul_mat_set_prec(out, GGML_PREC_F32);
    if (b) {
        struct ggml_tensor* b2d = ggml_reshape_2d(ctx, b, b->ne[0], 1);
        out = ggml_add(ctx, out, b2d);
    }
    return out;
}

// Conv1dGLU helper: Conv1d(128, 256, 5) -> split -> GLU -> residual
static struct ggml_tensor* conv1d_glu(
    struct ggml_context* ctx,
    struct ggml_tensor* x,     // [C, T]
    struct ggml_tensor* w,     // [kernel, C, 2*C]
    struct ggml_tensor* b      // [2*C]
) {
    if (x) if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[conv1d_glu Debug] x: [" << x->ne[0] << ", " << x->ne[1] << ", " << x->ne[2] << "]" << std::endl;
    if (w) if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[conv1d_glu Debug] w: [" << w->ne[0] << ", " << w->ne[1] << ", " << w->ne[2] << "]" << std::endl;
    if (b) if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[conv1d_glu Debug] b: [" << b->ne[0] << ", " << b->ne[1] << ", " << b->ne[2] << "]" << std::endl;
    int in_ch = (int)x->ne[0];
    struct ggml_tensor* x_t = ggml_cont(ctx, ggml_transpose(ctx, x));
    int pad = (int)(w->ne[0] - 1) / 2;
    struct ggml_tensor* conv = ggml_conv_1d_vits(ctx, w, x_t, 1, pad, 1);
    if (conv) if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[conv1d_glu Debug] conv: [" << conv->ne[0] << ", " << conv->ne[1] << ", " << conv->ne[2] << "], nelements=" << ggml_nelements(conv) << std::endl;
    struct ggml_tensor* conv_t = ggml_cont(ctx, ggml_transpose(ctx, conv));  // [2*C, T]
    if (conv_t) if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[conv1d_glu Debug] conv_t: [" << conv_t->ne[0] << ", " << conv_t->ne[1] << ", " << conv_t->ne[2] << "], nelements=" << ggml_nelements(conv_t) << std::endl;

    // Add bias
    struct ggml_tensor* b2d = ggml_reshape_2d(ctx, b, b->ne[0], 1);
    conv_t = ggml_add(ctx, conv_t, b2d);

    // Split into x1, x2 halves directly via views (bypassing redundant ggml_cont copies!)
    struct ggml_tensor* x1 = ggml_view_2d(ctx, conv_t, in_ch, conv_t->ne[1], conv_t->nb[1], 0);
    struct ggml_tensor* x2 = ggml_view_2d(ctx, conv_t, in_ch, conv_t->ne[1], conv_t->nb[1], in_ch * sizeof(float));

    // GLU: x1 * sigmoid(x2)
    struct ggml_tensor* glu = ggml_mul(ctx, x1, ggml_sigmoid(ctx, x2));

    // Residual connection: x + glu (x was already [C, T])
    return ggml_add(ctx, x, glu);
}

// =============================================================================
// TextEncoder (enc_p) building blocks
// =============================================================================

// LayerNorm: normalize along ne0, then scale + shift
static struct ggml_tensor* ggml_layer_norm(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* gamma,  // [channels]
    struct ggml_tensor* beta,   // [channels]
    float eps
) {
    struct ggml_tensor* norm = ggml_norm(ctx, x, eps);
    struct ggml_tensor* g2d = ggml_reshape_2d(ctx, gamma, gamma->ne[0], 1);
    struct ggml_tensor* b2d = ggml_reshape_2d(ctx, beta, beta->ne[0], 1);
    struct ggml_tensor* scaled = ggml_mul(ctx, norm, g2d);
    return ggml_add(ctx, scaled, b2d);
}

// fused_add_tanh_sigmoid_multiply: x = tanh(x_half1) * sigmoid(x_half2)
// Input has 2*C channels, first C go to tanh, last C to sigmoid
static struct ggml_tensor* ggml_gated_tanh_sigmoid(
    struct ggml_context* ctx,
    struct ggml_tensor* x,   // [2*C, T]
    int hidden_channels
) {
    // Use standard stride ne0*sizeof(float) instead of x->nb[1] for reliability
    size_t stride = x->ne[0] * sizeof(float);
    struct ggml_tensor* t_act = ggml_view_2d(ctx, x, hidden_channels, x->ne[1], stride, 0);
    struct ggml_tensor* s_act = ggml_view_2d(ctx, x, hidden_channels, x->ne[1], stride, hidden_channels * sizeof(float));
    struct ggml_tensor* tanh_part = ggml_tanh(ctx, ggml_cont(ctx, t_act));
    struct ggml_tensor* sigm_part = ggml_sigmoid(ctx, ggml_cont(ctx, s_act));
    return ggml_mul(ctx, tanh_part, sigm_part);
}

// One encoder layer: self-attention + FFN with residual connections and layer norm
// rel_scores_bias: optional [T_k, T_q, n_head] relative position bias for attention scores
// rel_out_bias: optional [C, T] relative position bias for attention output
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
    struct ggml_tensor* emb_rel_k = nullptr,  // [d_k, 9, n_head]
    struct ggml_tensor* emb_rel_v = nullptr   // [d_k, 9, n_head]
) {
    int C = (int)x->ne[0];  // hidden_channels = 192

    // Self-attention: QKV from Conv1d(kernel=1) = Linear, use conv_1d_with_bias directly
    struct ggml_tensor* q_w = model.get_tensor(prefix + "conv_q.weight"); // [1, C, C]
    struct ggml_tensor* q_b = model.get_tensor(prefix + "conv_q.bias");
    struct ggml_tensor* k_w = model.get_tensor(prefix + "conv_k.weight");
    struct ggml_tensor* k_b = model.get_tensor(prefix + "conv_k.bias");
    struct ggml_tensor* v_w = model.get_tensor(prefix + "conv_v.weight");
    struct ggml_tensor* v_b = model.get_tensor(prefix + "conv_v.bias");
    struct ggml_tensor* o_w = model.get_tensor(prefix + "conv_o.weight");
    struct ggml_tensor* o_b = model.get_tensor(prefix + "conv_o.bias");

    // Conv1d(192, 192, 1) = same shape output
    struct ggml_tensor* q = ggml_conv_1d_with_bias(ctx, x, q_w, q_b, 1, 1, 0);
    struct ggml_tensor* k = ggml_conv_1d_with_bias(ctx, x, k_w, k_b, 1, 1, 0);
    struct ggml_tensor* v = ggml_conv_1d_with_bias(ctx, x, v_w, v_b, 1, 1, 0);

    // Debug: save Q for first encoder_ssl layer
    if (prefix.find("enc_p.encoder_ssl.attn_layers.0.") != std::string::npos) {
        model.debug_enc_q = q;
    }

    // Multi-head attention: reshape Q/K/V to [d_k, n_head, T, 1], permute to [d_k, T, n_head, 1], cont
    q = ggml_cont(ctx, ggml_reshape_4d(ctx, q, d_k, n_head, T, 1));
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3)); // [d_k, T, n_head, 1]
    k = ggml_cont(ctx, ggml_reshape_4d(ctx, k, d_k, n_head, T, 1));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_reshape_4d(ctx, v, d_k, n_head, T, 1));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    // llama.cpp pattern: K^T @ Q → [T_k, T_q, n_head, 1] (ne0=T_k, ne1=T_q)
    // mul_mat(k, q): ne0 = k->ne1 = T_k (key), ne1 = q->ne1 = T_q (query)
    // No permute needed — softmax along ne0=T_k (key dim) directly
    // Cast to FP32 if needed (CUDA cuBLAS FP16 gemm may not be supported on all GPUs)
    float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);
    struct ggml_tensor* k_f32 = (k->type == GGML_TYPE_F32) ? k : ggml_cast(ctx, k, GGML_TYPE_F32);
    struct ggml_tensor* q_f32 = (q->type == GGML_TYPE_F32) ? q : ggml_cast(ctx, q, GGML_TYPE_F32);
    struct ggml_tensor* scores = ggml_mul_mat(ctx, k_f32, q_f32);
    scores = ggml_scale(ctx, scores, inv_sqrt_dk);
    struct ggml_tensor* scores_before_rel = scores;  // snapshot for debugging

    // Add dynamic relative position scores bias if present
    if (emb_rel_k != nullptr) {
        // 1. Slice relative keys embeddings based on length T and window_size = 4
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

        // 2. Compute dynamic relative logits: (Q/sqrt(dk)) @ rel_emb_k^T
        // Python: rel_logits = matmul(query / sqrt(dk), key_relative_embeddings.transpose(-2,-1))
        struct ggml_tensor* q_scaled = ggml_scale(ctx, q, inv_sqrt_dk);
        struct ggml_tensor* rel_emb_k_f32 = (rel_emb_k->type == GGML_TYPE_F32) ? rel_emb_k : ggml_cast(ctx, rel_emb_k, GGML_TYPE_F32);
        struct ggml_tensor* q_scaled_f32 = (q_scaled->type == GGML_TYPE_F32) ? q_scaled : ggml_cast(ctx, q_scaled, GGML_TYPE_F32);
        struct ggml_tensor* rel_logits = ggml_mul_mat(ctx, rel_emb_k_f32, q_scaled_f32);

        // 3. Apply coordinate shift trick in GGML
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

        // 4. Add dynamic relative position bias to attention scores (force contiguity for CUDA)
        struct ggml_tensor* scores_local_cont = ggml_cont(ctx, scores_local);
        scores = ggml_add(ctx, scores, scores_local_cont);
    }

    struct ggml_tensor* attn_w = ggml_soft_max(ctx, scores);  // [T_k, T_q, n_head, 1], ne0=T_k

    // v_t: [T_k, d_k, n_head, 1] — ne0=T_k (key dim) for mul_mat contraction
    // Cast to FP32 if needed (CUDA cuBLAS FP16 gemm may not be supported)
    struct ggml_tensor* v_t = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
    struct ggml_tensor* v_t_f32 = (v_t->type == GGML_TYPE_F32) ? v_t : ggml_cast(ctx, v_t, GGML_TYPE_F32);
    struct ggml_tensor* attn_w_f32 = (attn_w->type == GGML_TYPE_F32) ? attn_w : ggml_cast(ctx, attn_w, GGML_TYPE_F32);
    struct ggml_tensor* out = ggml_mul_mat(ctx, v_t_f32, attn_w_f32);  // v_t^T * attn_w → [d_k, T_q, n_head, 1]

    // Permute to [d_k, n_head, T, 1] then reshape_2d(ne0*ne1, ne2*ne3) = [C, T]
    out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
    struct ggml_tensor* attn_raw = ggml_reshape_2d(ctx, out, d_k * n_head, T);

    // Add dynamic relative position value bias if present
    if (emb_rel_v != nullptr) {
        // 1. Slice relative values embeddings based on length T and window_size = 4
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

        // 2. Pad attention weights attn_w along ne[0] by T - 1 columns of zeros
        struct ggml_tensor* zeros_cols = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T - 1, T, n_head);
        zeros_cols = ggml_fill(ctx, zeros_cols, 0.0f);
        struct ggml_tensor* x_padded = ggml_concat(ctx, attn_w, zeros_cols, 0); // [2*T-1, T, n_head]

        // 3. Flatten to 2D
        struct ggml_tensor* x_flat = ggml_reshape_2d(ctx, x_padded, T * (2 * T - 1), n_head);

        // 4. Pad at the beginning by T zeros
        struct ggml_tensor* zeros_beg = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, n_head);
        zeros_beg = ggml_fill(ctx, zeros_beg, 0.0f);
        struct ggml_tensor* x_flat_padded = ggml_concat(ctx, zeros_beg, x_flat, 0);

        // 5. Reshape to [2*T, T, n_head]
        struct ggml_tensor* x_final = ggml_reshape_3d(ctx, x_flat_padded, 2 * T, T, n_head);

        // 6. Slice to [2*T-1, T, n_head] taking columns 1 onwards (force contiguity for CUDA)
        size_t offset_w = 1 * sizeof(float);
        struct ggml_tensor* rel_weights = ggml_cont(ctx, ggml_view_3d(ctx, x_final, 2 * T - 1, T, n_head, x_final->nb[1], x_final->nb[2], offset_w));

        // 7. Transpose rel_emb_v to [2*T-1, d_k, n_head]
        struct ggml_tensor* rel_emb_v_t = ggml_cont(ctx, ggml_transpose(ctx, rel_emb_v));

        // 8. Multiply: rel_weights^T @ rel_emb_v_t -> [T, d_k, n_head] (force contiguity for repeated tensor to avoid cuBLAS 0-stride crashes)
        struct ggml_tensor* dummy = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, slice_len, d_k, n_head);
        struct ggml_tensor* rel_emb_v_t_repeated = ggml_cont(ctx, ggml_repeat(ctx, rel_emb_v_t, dummy));
        struct ggml_tensor* rel_weights_f32 = (rel_weights->type == GGML_TYPE_F32) ? rel_weights : ggml_cast(ctx, rel_weights, GGML_TYPE_F32);
        struct ggml_tensor* rel_emb_v_t_repeated_f32 = (rel_emb_v_t_repeated->type == GGML_TYPE_F32) ? rel_emb_v_t_repeated : ggml_cast(ctx, rel_emb_v_t_repeated, GGML_TYPE_F32);
        struct ggml_tensor* rel_out_bias = ggml_mul_mat(ctx, rel_weights_f32, rel_emb_v_t_repeated_f32);

        // 9. Permute [T, d_k, n_head] -> [d_k, n_head, T] then reshape to [d_k*n_head, T] = [C, T]
        rel_out_bias = ggml_cont(ctx, ggml_permute(ctx, rel_out_bias, 2, 0, 1, 3));
        struct ggml_tensor* rel_out_bias_flat = ggml_reshape_2d(ctx, rel_out_bias, d_k * n_head, T);

        // 10. Add relative values bias to attention output
        attn_raw = ggml_add(ctx, attn_raw, rel_out_bias_flat);
    }

    // Debug: save attention intermediates for first encoder_ssl layer
    if (prefix.find("enc_p.encoder_ssl.attn_layers.0.") != std::string::npos) {
        model.debug_enc_fa_raw = scores_before_rel;  // scores before rel pos bias
        model.debug_enc_scores = scores;             // scores after rel pos bias
        model.debug_enc_attn_w = attn_w;             // softmax weights
        model.debug_enc_out_raw = attn_raw;          // attention output
    }

    struct ggml_tensor* attn_proj = ggml_conv_1d_with_bias(ctx, attn_raw, o_w, o_b, 1, 1, 0);

    // Normal attention path
    struct ggml_tensor* x_attn = ggml_add(ctx, x, attn_proj);

    // Debug: save attn_proj for first encoder_ssl layer
    struct ggml_tensor* ln1_g = model.get_tensor(norm1_prefix + ".gamma");
    struct ggml_tensor* ln1_b = model.get_tensor(norm1_prefix + ".beta");
    x_attn = ggml_layer_norm(ctx, x_attn, ln1_g, ln1_b, 1e-5f);

    // FFN: Conv1d(C, 4*C, 3) + Conv1d(4*C, C, 3)
    struct ggml_tensor* ffn_w1 = model.get_tensor(ffn1_prefix + ".weight");
    struct ggml_tensor* ffn_b1 = model.get_tensor(ffn1_prefix + ".bias");
    struct ggml_tensor* ffn_w2 = model.get_tensor(ffn2_prefix + ".weight");
    struct ggml_tensor* ffn_b2 = model.get_tensor(ffn2_prefix + ".bias");

    struct ggml_tensor* ffn_out = ggml_conv_1d_with_bias(ctx, x_attn, ffn_w1, ffn_b1, 1, 1, 1);  // kernel=3, pad=1
    ffn_out = ggml_relu(ctx, ffn_out);  // Python FFN uses ReLU (activation=None in Encoder)
    ffn_out = ggml_conv_1d_with_bias(ctx, ffn_out, ffn_w2, ffn_b2, 1, 1, 1);

    // Residual + LayerNorm 2
    struct ggml_tensor* x_out = ggml_add(ctx, x_attn, ffn_out);
    struct ggml_tensor* ln2_g = model.get_tensor(norm2_prefix + ".gamma");
    struct ggml_tensor* ln2_b = model.get_tensor(norm2_prefix + ".beta");
    struct ggml_tensor* result = ggml_layer_norm(ctx, x_out, ln2_g, ln2_b, 1e-5f);

    // Debug: save full layer output for first encoder_ssl layer
    if (prefix.find("enc_p.encoder_ssl.attn_layers.0.") != std::string::npos) {
        model.debug_enc_attn = result;
    }

    return result;
}

// Full Encoder with n_layers
static struct ggml_tensor* build_encoder(
    struct ggml_context* ctx,
    struct ggml_tensor* x,        // [C, T], ne0=C, ne1=T
    struct ggml_tensor* x_mask,   // mask
    VITSModel& model,
    const std::string& base_prefix,  // "enc_p.encoder_ssl"
    int n_layers,
    int n_head,
    int d_k,
    int T
) {
    // Extract short encoder name from base_prefix (e.g., "enc_p.encoder_ssl" -> "encoder_ssl")
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
        if (l == 0) if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS] " << enc_short << " layer0: emb_rel_k=" << emb_rel_k << " emb_rel_v=" << emb_rel_v << std::endl;
        if (emb_rel_k && emb_rel_k->type != GGML_TYPE_F32) {
            emb_rel_k = ggml_cast(ctx, emb_rel_k, GGML_TYPE_F32);
    }
        if (emb_rel_v && emb_rel_v->type != GGML_TYPE_F32) {
            emb_rel_v = ggml_cast(ctx, emb_rel_v, GGML_TYPE_F32);
    }

        // Fallback: load emb_rel_k/v from external files if not in GGUF
        // Use persistent buffers keyed by encoder+layer to avoid overwriting
        static std::unordered_map<std::string, std::vector<float>> rel_k_bufs, rel_v_bufs;
        if (!emb_rel_k) {
            std::string fk = "scratch/enc_relk_" + enc_short + "_layer" + std::to_string(l) + ".f32";
            std::ifstream f(fk, std::ios::binary);
            if (l == 0) if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS] Trying to load emb_rel_k from: " << fk << " (found=" << f.is_open() << ")" << std::endl;
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
                if (l == 0) if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS] Loaded emb_rel_v for " << enc_short << " layer0 from file" << std::endl;
    }
    }

        x = build_encoder_layer(ctx, x, x_mask, model, lp, n1p, n2p, f1p, f2p, n_head, d_k, T, emb_rel_k, emb_rel_v);
    }
    return x;
}

// MRTE cross-attention module (simplified skip for now)
static struct ggml_tensor* build_mrte(
    struct ggml_context* ctx,
    struct ggml_tensor* y,        // SSL features [192, T_y]
    struct ggml_tensor* y_mask,   // [1, T_y]
    struct ggml_tensor* text,     // Text features [192, T_x]
    struct ggml_tensor* text_mask,// [1, T_x]
    struct ggml_tensor* ge,       // Speaker embedding [512, 1]
    VITSModel& model
) {
    (void)text_mask;
    int T_y = (int)y->ne[1];
    int T_x = (int)text->ne[1];

    // c_pre: Conv1d(192, 512, 1)
    struct ggml_tensor* c_pre_w = model.get_tensor("enc_p.mrte.c_pre.weight");
    struct ggml_tensor* c_pre_b = model.get_tensor("enc_p.mrte.c_pre.bias");
    struct ggml_tensor* y_proj = ggml_linear(ctx, y, c_pre_w, c_pre_b);  // [512, T_y]

    // text_pre: Conv1d(192, 512, 1)
    struct ggml_tensor* text_pre_w = model.get_tensor("enc_p.mrte.text_pre.weight");
    struct ggml_tensor* text_pre_b = model.get_tensor("enc_p.mrte.text_pre.bias");
    struct ggml_tensor* text_proj = ggml_linear(ctx, text, text_pre_w, text_pre_b);  // [512, T_x]

    // Debug: save MRTE c_pre and text_pre
    if (std::getenv("ENC_ALIGNMENT")) {
        model.debug_ref_enc_spectral_0 = y_proj;
        model.debug_ref_enc_spectral_3 = text_proj;
    }

    // Cross-attention: y attends to text
    // Q from y_proj, K,V from text_proj. n_head=4, d_k=128
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

    // Q from y_proj [512, T_y], K,V from text_proj [512, T_x]
    struct ggml_tensor* q_mrte = ggml_linear(ctx, y_proj, q_w, q_b);  // [512, T_y]
    struct ggml_tensor* k_mrte = ggml_linear(ctx, text_proj, k_w, k_b);  // [512, T_x]
    struct ggml_tensor* v_mrte = ggml_linear(ctx, text_proj, v_w, v_b);  // [512, T_x]

    // Debug: save MRTE Q (before reshape for flash_attn)
    if (std::getenv("ENC_ALIGNMENT")) {
        model.debug_enc_q_cont = q_mrte;
        model.debug_enc_fa_raw = k_mrte;
    }

    // Manual attention (same pattern as self-attention, verified working)
    // llama.cpp pattern: Q/K/V [d_k, seq, n_head, 1], scores = K^T @ Q, softmax along ne0
    q_mrte = ggml_cont(ctx, ggml_reshape_3d(ctx, q_mrte, mrte_d_k, mrte_n_head, T_y));
    q_mrte = ggml_cont(ctx, ggml_permute(ctx, q_mrte, 0, 2, 1, 3));  // [128, T_y, 4]

    k_mrte = ggml_cont(ctx, ggml_reshape_3d(ctx, k_mrte, mrte_d_k, mrte_n_head, T_x));
    k_mrte = ggml_cont(ctx, ggml_permute(ctx, k_mrte, 0, 2, 1, 3));  // [128, T_x, 4]

    v_mrte = ggml_cont(ctx, ggml_reshape_3d(ctx, v_mrte, mrte_d_k, mrte_n_head, T_x));
    v_mrte = ggml_cont(ctx, ggml_permute(ctx, v_mrte, 0, 2, 1, 3));  // [128, T_x, 4]

    // scores = K^T @ Q / sqrt(dk) → [T_x, T_y, n_head] (ne0=T_x=key len)
    float mrte_scale = 1.0f / sqrtf((float)mrte_d_k);
    struct ggml_tensor* k_mrte_f32 = (k_mrte->type == GGML_TYPE_F32) ? k_mrte : ggml_cast(ctx, k_mrte, GGML_TYPE_F32);
    struct ggml_tensor* q_mrte_f32 = (q_mrte->type == GGML_TYPE_F32) ? q_mrte : ggml_cast(ctx, q_mrte, GGML_TYPE_F32);
    struct ggml_tensor* scores_mrte = ggml_mul_mat(ctx, k_mrte_f32, q_mrte_f32);
    scores_mrte = ggml_scale(ctx, scores_mrte, mrte_scale);

    // Softmax along ne0 (key dim)
    struct ggml_tensor* attn_w_mrte = ggml_soft_max(ctx, scores_mrte);

    // v_t: [T_x, d_k, n_head] (ne0=T_x for mul_mat contraction)
    struct ggml_tensor* v_t_mrte = ggml_cont(ctx, ggml_permute(ctx, v_mrte, 1, 0, 2, 3));
    struct ggml_tensor* v_t_mrte_f32 = (v_t_mrte->type == GGML_TYPE_F32) ? v_t_mrte : ggml_cast(ctx, v_t_mrte, GGML_TYPE_F32);
    struct ggml_tensor* attn_w_mrte_f32 = (attn_w_mrte->type == GGML_TYPE_F32) ? attn_w_mrte : ggml_cast(ctx, attn_w_mrte, GGML_TYPE_F32);
    struct ggml_tensor* out_mrte = ggml_mul_mat(ctx, v_t_mrte_f32, attn_w_mrte_f32);  // [d_k, T_y, n_head]

    // Permute + reshape: [d_k, n_head, T_y] → [512, T_y]
    out_mrte = ggml_cont(ctx, ggml_permute(ctx, out_mrte, 0, 2, 1, 3));
    struct ggml_tensor* cross_out = ggml_reshape_2d(ctx, out_mrte, mrte_d_k * mrte_n_head, T_y);

    // Output projection: Conv1d(512, 512, 1)
    struct ggml_tensor* cross_proj = ggml_linear(ctx, cross_out, o_w, o_b);  // [512, T_y]

    // Debug: save cross_out
    if (std::getenv("ENC_ALIGNMENT")) {
        model.debug_enc_vt = cross_out;
        model.debug_ref_enc_temporal_0 = cross_proj;
    }

    // Residual: y_proj + cross_proj + ge
    // Use same pattern as ggml_linear bias addition: reshape ge to [512, 1] and add
    struct ggml_tensor* ge_2d = ggml_reshape_2d(ctx, ge, 512, 1);
    struct ggml_tensor* mrte_res = ggml_add(ctx, y_proj, cross_proj);
    mrte_res = ggml_add(ctx, mrte_res, ge_2d);

    // Debug: save MRTE residual
    if (std::getenv("ENC_ALIGNMENT")) {
        model.debug_enc_vt = cross_out;           // cross-attention output
        model.debug_ref_enc_pre_attn = mrte_res;  // residual
    }

    // c_post: Conv1d(512, 192, 1)
    struct ggml_tensor* c_post_w = model.get_tensor("enc_p.mrte.c_post.weight");
    struct ggml_tensor* c_post_b = model.get_tensor("enc_p.mrte.c_post.bias");
    struct ggml_tensor* result = ggml_linear(ctx, mrte_res, c_post_w, c_post_b);  // [192, T_y]

    // Debug: dump MRTE intermediates
    if (std::getenv("ENC_ALIGNMENT")) {
        model.debug_enc_mrte_out = result;
    }

    return result;
}

// =============================================================================
// WN (WaveNet) and Flow building blocks
// =============================================================================

// WN forward: x -> gated convolutions -> output
// x: [hidden_channels, T], g: [2*hidden*n_layers, T] or nullptr
static struct ggml_tensor* build_wn(
    struct ggml_context* ctx,
    struct ggml_tensor* x,        // [C, T]
    struct ggml_tensor* x_mask,   // [1, T] or nullptr
    struct ggml_tensor* g,        // conditioning, projetead by cond_layer externally, or nullptr
    VITSModel& model,
    const std::string& prefix,    // "flow.flows.0.enc."
    int hidden_channels,          // 192
    int kernel_size,              // 5
    int dilation_rate,            // 1
    int n_layers                  // 4
) {
    // Create zero-initialized output tensor (llama.cpp pattern: ggml_fill)
    struct ggml_tensor* output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_channels, x->ne[1]);
    output = ggml_fill(ctx, output, 0.0f);

    for (int i = 0; i < n_layers; ++i) {
        int dilation = (int)std::pow((float)dilation_rate, i);
        int padding = (int)((kernel_size * dilation - dilation) / 2);

        // in_layer: Conv1d(C, 2*C, kernel_size, dilation=dilation, padding=padding)
        std::string ilp = prefix + "in_layers." + std::to_string(i);
        struct ggml_tensor* in_w = model.get_tensor(ilp + ".weight");
        struct ggml_tensor* in_b = model.get_tensor(ilp + ".bias");
        struct ggml_tensor* x_in = ggml_conv_1d_with_bias(ctx, x, in_w, in_b, 1, dilation, padding);

        // Add conditioning if present
        if (g) {
            int cond_offset = i * 2 * hidden_channels;
            struct ggml_tensor* g_l = ggml_view_2d(ctx, g, 2 * hidden_channels, g->ne[1],
                g->nb[1], cond_offset * sizeof(float));
            x_in = ggml_add(ctx, x_in, ggml_cont(ctx, g_l));
        }

        // fused_add_tanh_sigmoid_multiply
        struct ggml_tensor* acts = ggml_gated_tanh_sigmoid(ctx, x_in, hidden_channels);

        // Debug: save WN layer 0 act and in for first flow
        if (prefix.find("flow.flows.6.enc.") != std::string::npos && i == 0) {
            model.debug_ref_enc_post_fc = acts;
        }

        // res_skip_layer: Conv1d(C, res_skip_channels, 1)
        std::string rsp = prefix + "res_skip_layers." + std::to_string(i);
        struct ggml_tensor* rs_w = model.get_tensor(rsp + ".weight");
        struct ggml_tensor* rs_b = model.get_tensor(rsp + ".bias");
        struct ggml_tensor* res_skip = ggml_conv_1d_with_bias(ctx, acts, rs_w, rs_b, 1, 1, 0);

        if (i < n_layers - 1) {
            // res_acts = first half of res_skip
            size_t rs_stride = res_skip->ne[0] * sizeof(float);
            struct ggml_tensor* res_acts = ggml_cont(ctx, ggml_view_2d(ctx, res_skip, hidden_channels,
                res_skip->ne[1], rs_stride, 0));
            // x = (x + res_acts) * x_mask
            x = ggml_add(ctx, x, res_acts);
            if (x_mask) {
                x = ggml_mul(ctx, x, ggml_repeat(ctx, x_mask, x));
            }
            // output += second half of res_skip
            struct ggml_tensor* skip_acts = ggml_cont(ctx, ggml_view_2d(ctx, res_skip, hidden_channels,
                res_skip->ne[1], rs_stride, hidden_channels * sizeof(float)));
            output = ggml_add(ctx, output, skip_acts);
            // Debug: save layer 0 skip
            if (prefix.find("flow.flows.6.enc.") != std::string::npos && i == 0) {
                model.debug_ref_enc_pre_pool = skip_acts;
    }
    } else {
            // Last layer: output += all of res_skip
            output = ggml_add(ctx, output, res_skip);
    }
    }
    if (x_mask) {
        output = ggml_mul(ctx, output, ggml_repeat(ctx, x_mask, output));
    }
    return output;
}

// ResidualCouplingLayer: x -> split -> WN -> affine coupling -> concat
// In reverse mode: x1 = (x1 - mean) * exp(-logs) with mean_only=true → x1 = x1 - mean
static struct ggml_tensor* build_coupling_layer(
    struct ggml_context* ctx,
    struct ggml_tensor* x,        // [channels, T] = [192, T], full channel count
    struct ggml_tensor* x_mask,   // [1, T]
    struct ggml_tensor* g,        // [512, 1] speaker embedding
    VITSModel& model,
    const std::string& prefix,    // "flow.flows.0."
    int channels,                 // 192
    int hidden_channels,          // 192
    int kernel_size,              // 5
    int dilation_rate,            // 1
    int n_layers,                 // 4
    bool reverse                  // true for inference
) {
    int half_c = channels / 2;  // 96

    // Split x into x0, x1: [96, T] each — use standard stride
    size_t x_stride = x->ne[0] * sizeof(float);
    struct ggml_tensor* x0 = ggml_view_2d(ctx, x, half_c, x->ne[1], x_stride, 0);
    struct ggml_tensor* x1 = ggml_view_2d(ctx, x, half_c, x->ne[1], x_stride, half_c * sizeof(float));

    // Debug: save x0 as cont'd copy for reliable dump (view dumps unreliable)
    if (prefix.find("flow.flows.0.") != std::string::npos) {
        model.debug_ref_enc_temporal_0 = x0;
    }

    // pre: Conv1d(half_c, hidden_channels, 1)
    struct ggml_tensor* pre_w = model.get_tensor(prefix + "pre.weight");
    struct ggml_tensor* pre_b = model.get_tensor(prefix + "pre.bias");
    // pre: Conv1d(half_c, hidden_channels, 1)
    struct ggml_tensor* h = ggml_conv_1d_with_bias(ctx, x0, pre_w, pre_b, 1, 1, 0);
    if (x_mask) {
        h = ggml_mul(ctx, h, ggml_repeat(ctx, x_mask, h));
    }

    // Debug: save pre output for first coupling layer
    if (prefix.find("flow.flows.0.") != std::string::npos) {
        model.debug_ref_enc_pre_attn = h;
    }

    // WN encoder
    struct ggml_tensor* g_proj = nullptr;
    struct ggml_tensor* cond_w = model.get_tensor(prefix + "enc.cond_layer.weight");
    struct ggml_tensor* cond_b = model.get_tensor(prefix + "enc.cond_layer.bias");
    if (cond_w && cond_b && g) {
        // cond_layer: Conv1d(512, 2*hidden*n_layers, 1) applied to g
        g_proj = ggml_conv_1d_with_bias(ctx, g, cond_w, cond_b, 1, 1, 0);
    }

    h = build_wn(ctx, h, x_mask, g_proj, model, prefix + "enc.",
        hidden_channels, kernel_size, dilation_rate, n_layers);

    if (prefix.find("flow.flows.0.") != std::string::npos) {
        model.debug_ref_enc_post_attn = h;
    }

    // post: Conv1d(hidden_channels, half_c, 1) — mean_only=true produces [half_c] channels
    struct ggml_tensor* post_w = model.get_tensor(prefix + "post.weight");
    struct ggml_tensor* post_b = model.get_tensor(prefix + "post.bias");
    struct ggml_tensor* mean = ggml_conv_1d_with_bias(ctx, h, post_w, post_b, 1, 1, 0);
    if (x_mask) {
        mean = ggml_mul(ctx, mean, ggml_repeat(ctx, x_mask, mean));
    }
    // Debug: save mean for first flow
    if (prefix.find("flow.flows.0.") != std::string::npos) {
        model.debug_enc_vt = mean;
    }

    // Combine x0 [half_c, T] and new_x1 [half_c, T] into [channels, T]
    // Use ggml_cont on ggml_concat to create a non-view tensor
    if (!reverse) {
        struct ggml_tensor* new_x1 = ggml_add(ctx, mean, ggml_cont(ctx, x1));
        return ggml_cont(ctx, ggml_concat(ctx, x0, new_x1, 0));
    } else {
        struct ggml_tensor* new_x1 = ggml_sub(ctx, ggml_cont(ctx, x1), mean);
        return ggml_cont(ctx, ggml_concat(ctx, x0, new_x1, 0));
    }
}

// MelStyleEncoder: compute speaker embedding from mel spectrogram
static struct ggml_tensor* build_ref_enc(
    struct ggml_context* ctx,
    struct ggml_tensor* mel_spec,   // [n_mel=704, T=155]
    VITSModel& model
) {
    int64_t T = mel_spec->ne[1];
    int64_t n_mel = mel_spec->ne[0];

    // Input: mel_spec [704, 155], ne0=704 (n_mel), ne1=155 (T)
    // ggml_mul_mat(a,b) computes a * b^T → result ne0=a->ne1, ne1=b->ne1
    // For Linear: w [ne0=in, ne1=out], x needs ne0=in. mel_spec already has ne0=704=in ✓
    struct ggml_tensor* x = mel_spec;  // no transpose needed!
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[RefEnc] Input: ne0=" << x->ne[0] << " ne1=" << x->ne[1] << std::endl;

    // Step 2: spectral[0] - LinearNorm(704, 128)
    struct ggml_tensor* s0_w = model.get_tensor("ref_enc.spectral.0.fc.weight"); // ne0=704 ne1=128
    struct ggml_tensor* s0_b = model.get_tensor("ref_enc.spectral.0.fc.bias");
    if (s0_w) {
        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[RefEnc Debug] s0_w: ne0=" << s0_w->ne[0] << ", ne1=" << s0_w->ne[1]
                  << ", ne2=" << s0_w->ne[2] << ", ne3=" << s0_w->ne[3] << ", type=" << s0_w->type
                  << ", nelements=" << ggml_nelements(s0_w) << std::endl;
    } else {
        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[RefEnc Debug] s0_w NOT FOUND" << std::endl;
    }
    if (s0_b) {
        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[RefEnc Debug] s0_b: ne0=" << s0_b->ne[0] << ", ne1=" << s0_b->ne[1]
                  << ", ne2=" << s0_b->ne[2] << ", ne3=" << s0_b->ne[3] << ", type=" << s0_b->type
                  << ", nelements=" << ggml_nelements(s0_b) << std::endl;
    } else {
        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[RefEnc Debug] s0_b NOT FOUND" << std::endl;
    }
    x = ggml_linear(ctx, x, s0_w, s0_b);  // result: ne0=128 (out), ne1=155 (T)
    model.debug_ref_enc_spectral_0 = x;
    x = ggml_mish(ctx, x);

    // spectral[3] - LinearNorm(128, 128)
    struct ggml_tensor* s3_w = model.get_tensor("ref_enc.spectral.3.fc.weight"); // ne0=128 ne1=128
    struct ggml_tensor* s3_b = model.get_tensor("ref_enc.spectral.3.fc.bias");
    if (s3_w) {
        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[RefEnc Debug] s3_w: ne0=" << s3_w->ne[0] << ", ne1=" << s3_w->ne[1]
                  << ", ne2=" << s3_w->ne[2] << ", ne3=" << s3_w->ne[3] << ", type=" << s3_w->type
                  << ", nelements=" << ggml_nelements(s3_w) << std::endl;
    }
    if (s3_b) {
        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[RefEnc Debug] s3_b: ne0=" << s3_b->ne[0] << ", ne1=" << s3_b->ne[1]
                  << ", ne2=" << s3_b->ne[2] << ", ne3=" << s3_b->ne[3] << ", type=" << s3_b->type
                  << ", nelements=" << ggml_nelements(s3_b) << std::endl;
    }
    x = ggml_linear(ctx, x, s3_w, s3_b);  // ne0=128, ne1=155
    model.debug_ref_enc_spectral_3 = x;
    x = ggml_mish(ctx, x);

    // Step 3: temporal Conv1dGLU x2
    // x is already channels-first: ne0=128 (ch), ne1=155 (T)
    // conv1d_glu expects [C, T] format → same as current!

    struct ggml_tensor* t0_w = model.get_tensor("ref_enc.temporal.0.conv1.conv.weight"); // [5, 128, 256]
    struct ggml_tensor* t0_b = model.get_tensor("ref_enc.temporal.0.conv1.conv.bias");
    x = conv1d_glu(ctx, x, t0_w, t0_b);  // ne0=128, ne1=155
    model.debug_ref_enc_temporal_0 = x;

    struct ggml_tensor* t1_w = model.get_tensor("ref_enc.temporal.1.conv1.conv.weight");
    struct ggml_tensor* t1_b = model.get_tensor("ref_enc.temporal.1.conv1.conv.bias");
    x = conv1d_glu(ctx, x, t1_w, t1_b);  // ne0=128, ne1=155
    model.debug_ref_enc_temporal_1 = x;

    // Step 4: Self-attention. x is channels-first: ne0=128, ne1=155
    // QKV weights have ne0=128=in_features → matches x->ne0 ✓
    // Parameters: n_head=2, d_model=128, d_k=64, d_v=64
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

        // Project Q, K, V: [T, 128] -> [T, 128] each
        struct ggml_tensor* q = ggml_linear(ctx, x, w_qs, b_qs);
        struct ggml_tensor* k = ggml_linear(ctx, x, w_ks, b_ks);
        struct ggml_tensor* v = ggml_linear(ctx, x, w_vs, b_vs);

        // Reshape to flash_attn format: [d_head, seq_len, n_head, batch]
        // Current: [T, n_head*d_k], ne0=n_head*d_k, ne1=T
        // Target: [d_k, T, n_head, 1]
        // Step: reshape to [d_k, n_head, T], permute to [d_k, T, n_head], add batch dim
        q = ggml_cont(ctx, ggml_reshape_3d(ctx, q, d_k, n_head, T));  // [d_k, n_head, T]
        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));        // [d_k, T, n_head]
        q = ggml_reshape_4d(ctx, q, d_k, T, n_head, 1);                // [d_k, T, n_head, 1]

        k = ggml_cont(ctx, ggml_reshape_3d(ctx, k, d_k, n_head, T));
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        k = ggml_reshape_4d(ctx, k, d_k, T, n_head, 1);

        v = ggml_cont(ctx, ggml_reshape_3d(ctx, v, d_v, n_head, T));
        v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
        v = ggml_reshape_4d(ctx, v, d_v, T, n_head, 1);

        // Standard dot-product attention
        float scale = 1.0f / sqrtf(128.0f);  // Python uses sqrt(d_model)=sqrt(128), not sqrt(d_k)
        struct ggml_tensor* k_f32 = (k->type == GGML_TYPE_F32) ? k : ggml_cast(ctx, k, GGML_TYPE_F32);
        struct ggml_tensor* q_f32 = (q->type == GGML_TYPE_F32) ? q : ggml_cast(ctx, q, GGML_TYPE_F32);
        struct ggml_tensor* kq = ggml_mul_mat(ctx, k_f32, q_f32); // [T, T, n_head]
        kq = ggml_scale(ctx, kq, scale);
        kq = ggml_soft_max(ctx, kq);

        struct ggml_tensor* v_perm = ggml_permute(ctx, v, 1, 0, 2, 3); // [T, d_v, n_head, 1]
        struct ggml_tensor* v_cont = ggml_cont(ctx, v_perm);
        struct ggml_tensor* v_cont_f32 = (v_cont->type == GGML_TYPE_F32) ? v_cont : ggml_cast(ctx, v_cont, GGML_TYPE_F32);
        struct ggml_tensor* kq_f32 = (kq->type == GGML_TYPE_F32) ? kq : ggml_cast(ctx, kq, GGML_TYPE_F32);
        struct ggml_tensor* attn_out = ggml_mul_mat(ctx, v_cont_f32, kq_f32); // [d_v, T, n_head, 1]

        // attn_out shape: [d_v, T, n_head, 1] -> reshape to [T, d_v*n_head] with ne1=channels
        attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 1, 0, 2, 3));  // [T, d_v, n_head, 1]
        attn_out = ggml_cont(ctx, ggml_reshape_2d(ctx, attn_out, d_v * n_head, T));  // [128, T], ne0=128, ne1=T
        // attn_out: [128, 155] ne0=128=in for FC. Same as residual.
        struct ggml_tensor* output = ggml_linear(ctx, attn_out, attn_fc_w, attn_fc_b);
        x = ggml_add(ctx, output, residual);  // both ne0=128, ne1=155
        model.debug_ref_enc_post_attn = x;
    }
    // Final FC: Linear 128 → 512
    // x ne0=128=in_features for FC ✓
    struct ggml_tensor* fc_w = model.get_tensor("ref_enc.fc.fc.weight"); // ne0=128, ne1=512
    struct ggml_tensor* fc_b = model.get_tensor("ref_enc.fc.fc.bias");
    x = ggml_linear(ctx, x, fc_w, fc_b);  // ne0=512 (out), ne1=155 (T)
    model.debug_ref_enc_post_fc = x;

    // Temporal average pooling: mean over time (ne1)
    // x: ne0=512 (ch), ne1=155 (T). Transpose so ne0=T for sum_rows:
    x = ggml_cont(ctx, ggml_transpose(ctx, x));  // ne0=155, ne1=512
    model.debug_ref_enc_pre_pool = x;
    struct ggml_tensor* summed = ggml_sum_rows(ctx, x);  // sums ne0=T, result ne=[1, 512]
    struct ggml_tensor* ge = ggml_scale(ctx, summed, 1.0f / (float)T);
    ge = ggml_cont(ctx, ge);  // [1, 512]
    return ge;
}

struct ggml_tensor* VITSModel::compute_speaker_embedding(
    struct ggml_context* ctx_graph,
    struct ggml_tensor* mel_spec,
    ggml_backend_t backend
) {
    current_vits_backend = backend;
    return build_ref_enc(ctx_graph, mel_spec, *this);
}

// Helper: VQ decode - look up semantic token IDs in the quantizer codebook
static struct ggml_tensor* vq_decode(
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

// Helper: nearest-neighbor interpolation, 2x along the time dimension
static struct ggml_tensor* interp_nearest_2x(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    VITSModel& model
) {
    int64_t C = x->ne[0];
    int64_t T = x->ne[1];

    // Reshape x: [C, T] -> [C, 1, T]
    struct ggml_tensor* x_3d = ggml_reshape_3d(ctx, x, C, 1, T);

    // Create target shape tensor: [C, 2, T]
    struct ggml_tensor* target = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, 2, T);

    // Repeat 2x along the middle dimension
    struct ggml_tensor* repeated = ggml_repeat(ctx, x_3d, target);

    // Reshape back to 2D: [C, 2, T] -> [C, T * 2] and make it contiguous
    return ggml_cont(ctx, ggml_reshape_2d(ctx, repeated, C, T * 2));
}

struct ggml_tensor* VITSModel::forward(
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
    current_vits_backend = backend;
    g_conv_1d_direct_params_pool.clear();
    (void)phone_lengths;
    (void)word2ph;
    (void)bert_features;
    (void)refer_audio;

    int semantic_len = (int)prompt_semantics->ne[0];
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS] VITS Inference Graph - semantic_len: " << semantic_len
              << ", speed: " << speed << std::endl;

    // Step 1: VQ Decode - semantic token IDs -> continuous features [768, N]
    struct ggml_tensor* decoded = vq_decode(ctx_graph, prompt_semantics, *this);
    if (!decoded) {
        std::cerr << "[VITS] Error: VQ decode failed!" << std::endl;
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
        y = ggml_conv_1d_with_bias(ctx_graph, interp, ssl_proj_w, ssl_proj_b, 1, 1, 0);
        debug_ssl_proj = y;
        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS] After ssl_proj shape: [" << y->ne[0] << ", " << y->ne[1] << "]" << std::endl;
    } else {
        std::cerr << "[VITS] Warning: ssl_proj weights missing, feeding raw VQ features to generator." << std::endl;
    }

    // Step 4: Load speaker embedding (ge)
    // refer_audio is now repurposed as ge tensor (512-dim speaker embedding from ref_enc)
    // passed from gpt_sovits.cpp synthesize_with_cache
    struct ggml_tensor* ge = refer_audio;  // [512, 1] or [512] from ref_enc
    if (ge) {
        // Reshape to [512, 1] if it's flat [512] (ne[1] == 0)
        if (ge->ne[1] == 0) {
            ge = ggml_reshape_2d(ctx_graph, ge, 512, 1);
        }
        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS] Using passed-in ge tensor from ref_enc: ne0=" << ge->ne[0] << " ne1=" << ge->ne[1] << std::endl;
    } else {
        // Fallback: try to load from disk (debug/alignment mode only)
        static float ge_data[512];
        bool ge_loaded = false;
        {
            std::ifstream ge_file("scratch/ref_enc_cpp_ge.f32", std::ios::binary);
            if (!ge_file.is_open()) ge_file.open("scratch/vits_alignment_py_ge.f32", std::ios::binary);
            if (ge_file.is_open()) {
                ge_file.read(reinterpret_cast<char*>(ge_data), 512 * sizeof(float));
                ge_loaded = true;
            }
        }
        if (!ge_loaded) {
            std::memset(ge_data, 0, 512 * sizeof(float));
        }
        ge = ggml_new_tensor_2d(ctx_graph, GGML_TYPE_F32, 512, 1);
        ge = ggml_fill(ctx_graph, ge, 0.0f);  // zero-init via ggml op (ensures proper backend buffer)
        // Note: disk-loaded ge_data not used in this fallback path;
        // for alignment with disk data, caller should upload via ggml_backend_tensor_set
        static bool ge_warned = false;
        if (!ge_warned) { ge_warned = true; if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS] ge fallback: using zero speaker embedding (alignment mode). ge_loaded=" << ge_loaded << std::endl; }
    }
    int n_head = 2;
    int d_k = 96;  // 192 / 2

    // Step 5: encoder_ssl (3 layers) on ssl features
    struct ggml_tensor* y_enc = build_encoder(ctx_graph, y, nullptr, *this, "enc_p.encoder_ssl", 3, n_head, d_k, T_y);
    debug_enc_ssl_out = y_enc;
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS] encoder_ssl done." << std::endl;

    // Step 6: encoder_text (6 layers) on phone embeddings
    struct ggml_tensor* text_emb_w = get_tensor("enc_p.text_embedding.weight");
    int text_len = (int)phone_ids->ne[0];
    struct ggml_tensor* text_emb = ggml_get_rows(ctx_graph, text_emb_w, phone_ids);  // [192, text_len]
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS] text_emb: [" << text_emb->ne[0] << ", " << text_emb->ne[1] << "]" << std::endl;

    struct ggml_tensor* text_enc = build_encoder(ctx_graph, text_emb, nullptr, *this, "enc_p.encoder_text", 6, n_head, d_k, text_len);
    debug_enc_text_out = text_enc;
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS] encoder_text done." << std::endl;

    // Step 7: MRTE - cross-attention between y_enc and text_enc with speaker conditioning
    struct ggml_tensor* mrte_out = build_mrte(ctx_graph, y_enc, nullptr, text_enc, nullptr, ge, *this);
    debug_enc_mrte_out = mrte_out;

    // Step 8: encoder2 (3 layers)
    struct ggml_tensor* y2 = build_encoder(ctx_graph, mrte_out, nullptr, *this, "enc_p.encoder2", 3, n_head, d_k, T_y);
    debug_enc_enc2_out = y2;

    // Step 9: Speed scaling
    if (speed != 1.0f && speed > 0.0f) {
        int target_frames = (int)std::round((float)T_y / speed);
        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS] Speed scaling: " << T_y << " -> " << target_frames << " frames" << std::endl;
        struct ggml_tensor* target = ggml_new_tensor_2d(ctx_graph, GGML_TYPE_F32, 192, target_frames);
        y2 = ggml_repeat(ctx_graph, y2, target);
    }

    // Step 10: proj - Conv1d(192, 384, 1) -> split to m_p (192) and logs (192)
    struct ggml_tensor* proj_w = get_tensor("enc_p.proj.weight");
    struct ggml_tensor* proj_b = get_tensor("enc_p.proj.bias");
    struct ggml_tensor* stats = ggml_conv_1d_with_bias(ctx_graph, y2, proj_w, proj_b, 1, 1, 0);  // [384, T_y]
    // llama.cpp pattern: cont + reshape to extract first 192 channels
    // ggml_view_2d may have stride issues; use ggml_cont on the view
    struct ggml_tensor* m_p = ggml_view_2d(ctx_graph, stats, 192, stats->ne[1], stats->nb[1], 0);
    m_p = ggml_cont(ctx_graph, m_p);
    debug_enc_m_p = m_p;

    // Step 11: Flow reverse (ResidualCouplingBlock)
    // Reverse order: Flip6, RCL6, Flip4, RCL4, Flip2, RCL2, Flip0, RCL0
    // Permutation matrix to reverse 192 channels contiguously (matching PyTorch's torch.flip(x, [1]))
    if (!VITSModel::flip_data_ready) {
        std::memset(VITSModel::flip_data, 0, sizeof(VITSModel::flip_data));
        for (int r = 0; r < 192; ++r) {
            VITSModel::flip_data[r * 192 + (191 - r)] = 1.0f;
        }
        VITSModel::flip_data_ready = true;
    }

    auto flip_ch = [&](struct ggml_tensor* t) -> struct ggml_tensor* {
        // Upload permutation matrix data after backend alloc via upload_pending_data()
        struct ggml_tensor* P = ggml_new_tensor_2d(ctx_graph, GGML_TYPE_F32, 192, 192);
        upload_entries.push_back({P, std::vector<uint8_t>((uint8_t*)flip_data, (uint8_t*)(flip_data + 192*192))});
        return ggml_cont(ctx_graph, ggml_mul_mat(ctx_graph, P, t));
    };

    // TEST: compare mul_mat arg order — encoder puts weight as 2nd arg
    size_t s = m_p->ne[0] * sizeof(float);
    struct ggml_tensor* x0 = ggml_view_2d(ctx_graph, m_p, 96, m_p->ne[1], s, 0);
    struct ggml_tensor* pw = get_tensor("flow.flows.6.pre.weight");
    struct ggml_tensor* pb = get_tensor("flow.flows.6.pre.bias");

    struct ggml_tensor* z = m_p;
    for (int fi : {6, 4, 2, 0}) {
        z = flip_ch(z);
        std::string flow_p = "flow.flows." + std::to_string(fi) + ".";
        z = build_coupling_layer(ctx_graph, z, nullptr, ge, *this, flow_p,
            192, 192, 5, 1, 4, true);
        if (fi == 6) debug_ref_enc_spectral_0 = z;
        if (fi == 4) debug_ref_enc_spectral_3 = z;
        if (fi == 2) debug_ref_enc_temporal_1 = z;
    }
    z = ggml_cont(ctx_graph, z);
    debug_enc_z = z;
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS] Flow done. z: [" << z->ne[0] << ", " << z->ne[1] << "]" << std::endl;

    return build_vits_generator(ctx_graph, z, ge, *this);
}

} // namespace gpt_sovits
