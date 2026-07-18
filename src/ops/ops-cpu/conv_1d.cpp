#include "ops/ops.h"
#include "ops/cpu.h"
#define GGML_COMMON_DECL_CPP
#include "ggml.h"
#include "ggml-common.h"
#include "quantized_block_avx2.h"
#include "matmul_f32.h"
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <cmath>
#include <immintrin.h>

namespace ggml_ops_ext {
namespace cpu {

inline float inline_vec_dot_f32(int n, const float * x, const float * y) {
    float result = 0.0f;
    int i = 0;
#if defined(__AVX2__)
    const int step = 8;
    int np = (n & ~(step - 1));
    __m256 sum = _mm256_setzero_ps();
    for (; i < np; i += step) {
        sum = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i), sum);
    }
    float buffer[8];
    _mm256_storeu_ps(buffer, sum);
    result = buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] + buffer[7];
#endif
    for (; i < n; ++i) {
        result += x[i] * y[i];
    }
    return result;
}

inline void inline_vec_dot_f32_x4(int n, const float * x, 
                                  const float * y0, const float * y1, const float * y2, const float * y3,
                                  float * dot0, float * dot1, float * dot2, float * dot3) {
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    int i = 0;
#if defined(__AVX2__)
    const int step = 8;
    int np = (n & ~(step - 1));
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();
    for (; i < np; i += step) {
        __m256 vx = _mm256_loadu_ps(x + i);
        s0 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(y0 + i), s0);
        s1 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(y1 + i), s1);
        s2 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(y2 + i), s2);
        s3 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(y3 + i), s3);
    }
    float b0[8], b1[8], b2[8], b3[8];
    _mm256_storeu_ps(b0, s0);
    _mm256_storeu_ps(b1, s1);
    _mm256_storeu_ps(b2, s2);
    _mm256_storeu_ps(b3, s3);
    sum0 = b0[0] + b0[1] + b0[2] + b0[3] + b0[4] + b0[5] + b0[6] + b0[7];
    sum1 = b1[0] + b1[1] + b1[2] + b1[3] + b1[4] + b1[5] + b1[6] + b1[7];
    sum2 = b2[0] + b2[1] + b2[2] + b2[3] + b2[4] + b2[5] + b2[6] + b2[7];
    sum3 = b3[0] + b3[1] + b3[2] + b3[3] + b3[4] + b3[5] + b3[6] + b3[7];
#endif
    for (; i < n; ++i) {
        float vx = x[i];
        sum0 += vx * y0[i];
        sum1 += vx * y1[i];
        sum2 += vx * y2[i];
        sum3 += vx * y3[i];
    }
    *dot0 = sum0; *dot1 = sum1; *dot2 = sum2; *dot3 = sum3;
}

bool ops_cpu_op_conv_1d(ggml_backend_t backend, struct ggml_tensor* node) {
    const int omp_threads = backend_thread_count(backend);

    ops_conv_1d_params params;
    if (!ops_extract_conv_1d_params(node, params)) return false;

    const struct ggml_tensor * w   = params.w;
    const struct ggml_tensor * x   = params.x;
    struct ggml_tensor *       dst = node;

    int stride   = params.stride;
    int padding  = params.padding;
    int dilation = params.dilation;

    ops_conv_weight_desc weight_desc = {};
    if (!ops_describe_conv_weight(GGML_OP_OPS_VIRT_CONV_1D, w, x, params.groups, weight_desc)) return false;
    const int64_t kW          = weight_desc.kernel;
    const int64_t C_in_group  = weight_desc.input_channels_per_group;
    const int64_t C_out       = weight_desc.output_channels;
    const int64_t L_in        = x->ne[0];
    const int64_t batch       = (x->ne[2] > 0) ? x->ne[2] : 1;
    const int64_t groups      = params.groups;
    const int64_t C_out_group = C_out / groups;

    const int64_t L_out = (L_in + 2 * padding - dilation * (kW - 1) - 1) / stride + 1;
    GGML_ASSERT(L_out == dst->ne[0]);
    GGML_ASSERT(x->type   == GGML_TYPE_F32 || x->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);

    std::vector<float> bias_vec(C_out, 0.0f);
    if (params.bias) {
        const struct ggml_tensor* bias = params.bias;
        for (int64_t oc = 0; oc < C_out; ++oc) {
            float val = 0.0f;
            if (bias->type == GGML_TYPE_F32) {
                val = *(const float *)((const char *)bias->data + oc * bias->nb[0]);
            } else if (bias->type == GGML_TYPE_F16) {
                val = ggml_fp16_to_fp32(*(const ggml_fp16_t *)((const char *)bias->data + oc * bias->nb[0]));
            }
            bias_vec[oc] = val;
        }
    }

    if (w->type != GGML_TYPE_F32) {
        const bool quantized = ggml_is_quantized(w->type);
        const int64_t block_size = quantized ? ggml_blck_size(w->type) : 1;
        const size_t row_size = quantized ? ggml_row_size(w->type, C_in_group) : 0;
        if (quantized &&
            w->type != GGML_TYPE_Q4_0 && w->type != GGML_TYPE_Q4_K && w->type != GGML_TYPE_Q8_0) {
            return false;
        }
        if (quantized && (block_size <= 0 || block_size > QK_K || row_size == 0)) {
            return false;
        }
        std::vector<float> input_rows((size_t)batch * groups * L_in * C_in_group);
        #pragma omp parallel for collapse(2) num_threads(omp_threads)
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t group = 0; group < groups; ++group) {
                float* group_rows = input_rows.data() + ((size_t)b * groups + group) * L_in * C_in_group;
                for (int64_t iw = 0; iw < L_in; ++iw) {
                    float* row = group_rows + iw * C_in_group;
                    for (int64_t local_ic = 0; local_ic < C_in_group; ++local_ic) {
                        const int64_t global_ic = group * C_in_group + local_ic;
                        const char* input_ptr = static_cast<const char*>(x->data) +
                            b * x->nb[2] + global_ic * x->nb[1] + iw * x->nb[0];
                        row[local_ic] = x->type == GGML_TYPE_F16
                            ? ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(input_ptr))
                            : *reinterpret_cast<const float*>(input_ptr);
                    }
                }
            }
        }
        if (quantized) {
            constexpr int64_t time_tile = 8;
            const size_t quant_block_bytes = ggml_type_size(w->type);
            #pragma omp parallel for collapse(2) num_threads(omp_threads)
            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t oc = 0; oc < C_out; ++oc) {
                    const int64_t group = oc / C_out_group;
                    const float* group_rows = input_rows.data() +
                        ((size_t)b * groups + group) * L_in * C_in_group;
                    for (int64_t output_start = 0; output_start < L_out; output_start += time_tile) {
                        float sums[time_tile];
                        for (int64_t t = 0; t < time_tile; ++t) sums[t] = bias_vec[oc];
                        for (int64_t kw = 0; kw < kW; ++kw) {
                            const int64_t weight_row = oc * kW + kw;
                            const char* row_data = static_cast<const char*>(w->data) + weight_row * row_size;
                            for (int64_t block = 0; block < C_in_group; block += 32) {
                                const int64_t storage_block = block / block_size;
                                const int64_t block_offset = block % block_size;
                                quantized_32_microtile tile;
                                decode_quantized_32_avx2(
                                    w->type, row_data + storage_block * quant_block_bytes, block_offset, tile);
                                for (int64_t t = 0; t < time_tile; ++t) {
                                    const int64_t ow = output_start + t;
                                    const int64_t iw = ow * stride - padding + kw * dilation;
                                    if (ow >= L_out || iw < 0 || iw >= L_in) continue;
                                    sums[t] += dot_quantized_32_microtile_avx2(
                                        w->type, tile, group_rows + iw * C_in_group + block);
                                }
                            }
                        }
                        for (int64_t t = 0; t < time_tile; ++t) {
                            const int64_t ow = output_start + t;
                            if (ow >= L_out) continue;
                            char* output_ptr = static_cast<char*>(dst->data) +
                                b * dst->nb[2] + oc * dst->nb[1] + ow * dst->nb[0];
                            if (dst->type == GGML_TYPE_F16) {
                                *reinterpret_cast<ggml_fp16_t*>(output_ptr) = ggml_fp32_to_fp16(sums[t]);
                            } else {
                                *reinterpret_cast<float*>(output_ptr) = sums[t];
                            }
                        }
                    }
                }
            }
        } else {
            #pragma omp parallel for collapse(2) num_threads(omp_threads)
            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t oc = 0; oc < C_out; ++oc) {
                    alignas(32) float decoded[32];
                    const int64_t group = oc / C_out_group;
                    const float* group_rows = input_rows.data() +
                        ((size_t)b * groups + group) * L_in * C_in_group;
                    for (int64_t ow = 0; ow < L_out; ++ow) {
                        float sum = bias_vec[oc];
                        for (int64_t kw = 0; kw < kW; ++kw) {
                            const int64_t iw = ow * stride - padding + kw * dilation;
                            if (iw < 0 || iw >= L_in) continue;
                            const float* input_row = group_rows + iw * C_in_group;
                            constexpr int64_t chunk = 32;
                            for (int64_t block = 0; block < C_in_group; block += chunk) {
                                const int64_t count = std::min<int64_t>(chunk, C_in_group - block);
                                for (int64_t lane = 0; lane < count; ++lane) {
                                    const int64_t local_ic = block + lane;
                                    const size_t offset = oc * w->nb[2] + local_ic * w->nb[1] + kw * w->nb[0];
                                    decoded[lane] = ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(
                                        static_cast<const char*>(w->data) + offset));
                                }
                                sum += inline_vec_dot_f32((int)count, input_row + block, decoded);
                            }
                        }
                        char* output_ptr = static_cast<char*>(dst->data) +
                            b * dst->nb[2] + oc * dst->nb[1] + ow * dst->nb[0];
                        if (dst->type == GGML_TYPE_F16) {
                            *reinterpret_cast<ggml_fp16_t*>(output_ptr) = ggml_fp32_to_fp16(sum);
                        } else {
                            *reinterpret_cast<float*>(output_ptr) = sum;
                        }
                    }
                }
            }
        }
        return true;
    }

    std::vector<float> w_transposed(C_out * kW * C_in_group);
    for (int64_t oc = 0; oc < C_out; ++oc) {
        for (int64_t ic = 0; ic < C_in_group; ++ic) {
            for (int64_t k = 0; k < kW; ++k) {
                const size_t offset = oc * w->nb[2] + ic * w->nb[1] + k * w->nb[0];
                w_transposed[oc * (kW * C_in_group) + k * C_in_group + ic] =
                    *reinterpret_cast<const float*>(static_cast<const char*>(w->data) + offset);
            }
        }
    }

    // Calculate static boundaries for fast path (where padding checks are not needed)
    int64_t ow_start = (stride > 0) ? (padding + stride - 1) / stride : 0;
    int64_t ow_end = (stride > 0) ? (L_in + padding - (kW - 1) * dilation) / stride : 0;
    if (ow_end < ow_start) {
        ow_end = ow_start;
    }

    if (batch * groups >= 4) {
        #pragma omp parallel num_threads(omp_threads)
        {
            std::vector<float> x_transposed(L_in * C_in_group);
            #pragma omp for collapse(2)
            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t g = 0; g < groups; ++g) {
                    // Transpose input slice of group g to [L_in, C_in_group] row-major layout
                    for (int64_t ic = 0; ic < C_in_group; ++ic) {
                        for (int64_t iw = 0; iw < L_in; ++iw) {
                            int64_t global_ic = g * C_in_group + ic;
                            size_t offset = b * x->nb[2] + global_ic * x->nb[1] + iw * x->nb[0];
                            if (x->type == GGML_TYPE_F16) {
                                x_transposed[iw * C_in_group + ic] = ggml_fp16_to_fp32(*(const ggml_fp16_t *)((const char *)x->data + offset));
                            } else {
                                x_transposed[iw * C_in_group + ic] = *(const float *)((const char *)x->data + offset);
                            }
                        }
                    }

                    // Direct Vectorized Convolution loop with 4x channel blocking
                    for (int64_t ow = 0; ow < L_out; ++ow) {
                        for (int64_t oc_in_group = 0; oc_in_group < C_out_group; oc_in_group += 4) {
                            int64_t oc = g * C_out_group + oc_in_group;
                            if (oc_in_group + 3 < C_out_group) {
                                const float * vec_w0_base = w_transposed.data() + (oc + 0) * (kW * C_in_group);
                                const float * vec_w1_base = w_transposed.data() + (oc + 1) * (kW * C_in_group);
                                const float * vec_w2_base = w_transposed.data() + (oc + 2) * (kW * C_in_group);
                                const float * vec_w3_base = w_transposed.data() + (oc + 3) * (kW * C_in_group);

                                float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
                                if (ow >= ow_start && ow < ow_end) {
                                    for (int64_t k = 0; k < kW; ++k) {
                                        int64_t iw = ow * stride - padding + k * dilation;
                                        const float * vec_x = x_transposed.data() + iw * C_in_group;
                                        
                                        float dot0, dot1, dot2, dot3;
                                        inline_vec_dot_f32_x4((int)C_in_group, vec_x, 
                                                              vec_w0_base + k * C_in_group,
                                                              vec_w1_base + k * C_in_group,
                                                              vec_w2_base + k * C_in_group,
                                                              vec_w3_base + k * C_in_group,
                                                              &dot0, &dot1, &dot2, &dot3);
                                        sum0 += dot0; sum1 += dot1; sum2 += dot2; sum3 += dot3;
                                    }
                                } else {
                                    for (int64_t k = 0; k < kW; ++k) {
                                        int64_t iw = ow * stride - padding + k * dilation;
                                        if (iw >= 0 && iw < L_in) {
                                            const float * vec_x = x_transposed.data() + iw * C_in_group;
                                            
                                            float dot0, dot1, dot2, dot3;
                                            inline_vec_dot_f32_x4((int)C_in_group, vec_x, 
                                                                  vec_w0_base + k * C_in_group,
                                                                  vec_w1_base + k * C_in_group,
                                                                  vec_w2_base + k * C_in_group,
                                                                  vec_w3_base + k * C_in_group,
                                                                  &dot0, &dot1, &dot2, &dot3);
                                            sum0 += dot0; sum1 += dot1; sum2 += dot2; sum3 += dot3;
                                        }
                                    }
                                }
                                if (dst->type == GGML_TYPE_F16) {
                                    *(ggml_fp16_t *)((char *)dst->data + (oc + 0) * dst->nb[1] + b * dst->nb[2] + ow * dst->nb[0]) = ggml_fp32_to_fp16(sum0 + bias_vec[oc + 0]);
                                    *(ggml_fp16_t *)((char *)dst->data + (oc + 1) * dst->nb[1] + b * dst->nb[2] + ow * dst->nb[0]) = ggml_fp32_to_fp16(sum1 + bias_vec[oc + 1]);
                                    *(ggml_fp16_t *)((char *)dst->data + (oc + 2) * dst->nb[1] + b * dst->nb[2] + ow * dst->nb[0]) = ggml_fp32_to_fp16(sum2 + bias_vec[oc + 2]);
                                    *(ggml_fp16_t *)((char *)dst->data + (oc + 3) * dst->nb[1] + b * dst->nb[2] + ow * dst->nb[0]) = ggml_fp32_to_fp16(sum3 + bias_vec[oc + 3]);
                                } else {
                                    *(float *)((char *)dst->data + (oc + 0) * dst->nb[1] + b * dst->nb[2] + ow * dst->nb[0]) = sum0 + bias_vec[oc + 0];
                                    *(float *)((char *)dst->data + (oc + 1) * dst->nb[1] + b * dst->nb[2] + ow * dst->nb[0]) = sum1 + bias_vec[oc + 1];
                                    *(float *)((char *)dst->data + (oc + 2) * dst->nb[1] + b * dst->nb[2] + ow * dst->nb[0]) = sum2 + bias_vec[oc + 2];
                                    *(float *)((char *)dst->data + (oc + 3) * dst->nb[1] + b * dst->nb[2] + ow * dst->nb[0]) = sum3 + bias_vec[oc + 3];
                                }
                            } else {
                                // Fallback path for any remaining trailing channels
                                for (int64_t soc = oc; soc < (g + 1) * C_out_group; ++soc) {
                                    const float * vec_w_base = w_transposed.data() + soc * (kW * C_in_group);
                                    float sum = 0.0f;
                                    if (ow >= ow_start && ow < ow_end) {
                                        for (int64_t k = 0; k < kW; ++k) {
                                            int64_t iw = ow * stride - padding + k * dilation;
                                            const float * vec_x = x_transposed.data() + iw * C_in_group;
                                            sum += inline_vec_dot_f32((int)C_in_group, vec_x, vec_w_base + k * C_in_group);
                                        }
                                    } else {
                                        for (int64_t k = 0; k < kW; ++k) {
                                            int64_t iw = ow * stride - padding + k * dilation;
                                            if (iw >= 0 && iw < L_in) {
                                                const float * vec_x = x_transposed.data() + iw * C_in_group;
                                                sum += inline_vec_dot_f32((int)C_in_group, vec_x, vec_w_base + k * C_in_group);
                                            }
                                        }
                                    }
                                    if (dst->type == GGML_TYPE_F16) {
                                        *(ggml_fp16_t *)((char *)dst->data + soc * dst->nb[1] + b * dst->nb[2] + ow * dst->nb[0]) = ggml_fp32_to_fp16(sum + bias_vec[soc]);
                                    } else {
                                        *(float *)((char *)dst->data + soc * dst->nb[1] + b * dst->nb[2] + ow * dst->nb[0]) = sum + bias_vec[soc];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        // Inner loop parallelization (optimized for standard convolutions with groups=1)
        std::vector<float> x_transposed(L_in * C_in_group);
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t g = 0; g < groups; ++g) {
                // Transpose input slice of group g to [L_in, C_in_group] row-major layout
                #pragma omp parallel for collapse(2) num_threads(omp_threads)
                for (int64_t ic = 0; ic < C_in_group; ++ic) {
                    for (int64_t iw = 0; iw < L_in; ++iw) {
                        int64_t global_ic = g * C_in_group + ic;
                        size_t offset = b * x->nb[2] + global_ic * x->nb[1] + iw * x->nb[0];
                        if (x->type == GGML_TYPE_F16) {
                            x_transposed[iw * C_in_group + ic] = ggml_fp16_to_fp32(*(const ggml_fp16_t *)((const char *)x->data + offset));
                        } else {
                            x_transposed[iw * C_in_group + ic] = *(const float *)((const char *)x->data + offset);
                        }
                    }
                }

                // Direct Vectorized Convolution loop with 4x channel blocking
                #pragma omp parallel for num_threads(omp_threads)
                for (int64_t ow = 0; ow < L_out; ++ow) {
                    for (int64_t oc_in_group = 0; oc_in_group < C_out_group; oc_in_group += 4) {
                        int64_t oc = g * C_out_group + oc_in_group;
                        if (oc_in_group + 3 < C_out_group) {
                            const float * vec_w0_base = w_transposed.data() + (oc + 0) * (kW * C_in_group);
                            const float * vec_w1_base = w_transposed.data() + (oc + 1) * (kW * C_in_group);
                            const float * vec_w2_base = w_transposed.data() + (oc + 2) * (kW * C_in_group);
                            const float * vec_w3_base = w_transposed.data() + (oc + 3) * (kW * C_in_group);

                            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
                            if (ow >= ow_start && ow < ow_end) {
                                for (int64_t k = 0; k < kW; ++k) {
                                    int64_t iw = ow * stride - padding + k * dilation;
                                    const float * vec_x = x_transposed.data() + iw * C_in_group;
                                    
                                    float dot0, dot1, dot2, dot3;
                                    inline_vec_dot_f32_x4((int)C_in_group, vec_x, 
                                                          vec_w0_base + k * C_in_group,
                                                          vec_w1_base + k * C_in_group,
                                                          vec_w2_base + k * C_in_group,
                                                          vec_w3_base + k * C_in_group,
                                                          &dot0, &dot1, &dot2, &dot3);
                                    sum0 += dot0; sum1 += dot1; sum2 += dot2; sum3 += dot3;
                                }
                            } else {
                                for (int64_t k = 0; k < kW; ++k) {
                                    int64_t iw = ow * stride - padding + k * dilation;
                                    if (iw >= 0 && iw < L_in) {
                                        const float * vec_x = x_transposed.data() + iw * C_in_group;
                                        
                                        float dot0, dot1, dot2, dot3;
                                        inline_vec_dot_f32_x4((int)C_in_group, vec_x, 
                                                              vec_w0_base + k * C_in_group,
                                                              vec_w1_base + k * C_in_group,
                                                              vec_w2_base + k * C_in_group,
                                                              vec_w3_base + k * C_in_group,
                                                              &dot0, &dot1, &dot2, &dot3);
                                        sum0 += dot0; sum1 += dot1; sum2 += dot2; sum3 += dot3;
                                    }
                                }
                            }
                            if (dst->type == GGML_TYPE_F16) {
                                *(ggml_fp16_t *)((char *)dst->data + (oc + 0) * dst->nb[1] + b * dst->nb[2] + ow * dst->nb[0]) = ggml_fp32_to_fp16(sum0 + bias_vec[oc + 0]);
                                *(ggml_fp16_t *)((char *)dst->data + (oc + 1) * dst->nb[1] + b * dst->nb[2] + ow * dst->nb[0]) = ggml_fp32_to_fp16(sum1 + bias_vec[oc + 1]);
                                *(ggml_fp16_t *)((char *)dst->data + (oc + 2) * dst->nb[1] + b * dst->nb[2] + ow * dst->nb[0]) = ggml_fp32_to_fp16(sum2 + bias_vec[oc + 2]);
                                *(ggml_fp16_t *)((char *)dst->data + (oc + 3) * dst->nb[1] + b * dst->nb[2] + ow * dst->nb[0]) = ggml_fp32_to_fp16(sum3 + bias_vec[oc + 3]);
                            } else {
                                *(float *)((char *)dst->data + (oc + 0) * dst->nb[1] + b * dst->nb[2] + ow * dst->nb[0]) = sum0 + bias_vec[oc + 0];
                                *(float *)((char *)dst->data + (oc + 1) * dst->nb[1] + b * dst->nb[2] + ow * dst->nb[0]) = sum1 + bias_vec[oc + 1];
                                *(float *)((char *)dst->data + (oc + 2) * dst->nb[1] + b * dst->nb[2] + ow * dst->nb[0]) = sum2 + bias_vec[oc + 2];
                                *(float *)((char *)dst->data + (oc + 3) * dst->nb[1] + b * dst->nb[2] + ow * dst->nb[0]) = sum3 + bias_vec[oc + 3];
                            }
                        } else {
                            // Fallback path for any remaining trailing channels
                            for (int64_t soc = oc; soc < (g + 1) * C_out_group; ++soc) {
                                const float * vec_w_base = w_transposed.data() + soc * (kW * C_in_group);
                                float sum = 0.0f;
                                if (ow >= ow_start && ow < ow_end) {
                                    for (int64_t k = 0; k < kW; ++k) {
                                        int64_t iw = ow * stride - padding + k * dilation;
                                        const float * vec_x = x_transposed.data() + iw * C_in_group;
                                        sum += inline_vec_dot_f32((int)C_in_group, vec_x, vec_w_base + k * C_in_group);
                                    }
                                } else {
                                    for (int64_t k = 0; k < kW; ++k) {
                                        int64_t iw = ow * stride - padding + k * dilation;
                                        if (iw >= 0 && iw < L_in) {
                                            const float * vec_x = x_transposed.data() + iw * C_in_group;
                                            sum += inline_vec_dot_f32((int)C_in_group, vec_x, vec_w_base + k * C_in_group);
                                        }
                                    }
                                }
                                if (dst->type == GGML_TYPE_F16) {
                                    *(ggml_fp16_t *)((char *)dst->data + soc * dst->nb[1] + b * dst->nb[2] + ow * dst->nb[0]) = ggml_fp32_to_fp16(sum + bias_vec[soc]);
                                } else {
                                    *(float *)((char *)dst->data + soc * dst->nb[1] + b * dst->nb[2] + ow * dst->nb[0]) = sum + bias_vec[soc];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return true;
}

} // namespace cpu
} // namespace ggml_ops_ext
