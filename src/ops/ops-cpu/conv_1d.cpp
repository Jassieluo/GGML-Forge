#include "ops/ops.h"
#define GGML_COMMON_DECL_CPP
#include "ggml.h"
#include "ggml-common.h"
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
    (void)backend;

    ops_conv_1d_params params;
    if (!ops_extract_conv_1d_params(node, params)) return false;

    const struct ggml_tensor * w   = params.w;
    const struct ggml_tensor * x   = params.x;
    struct ggml_tensor *       dst = node;

    int stride   = params.stride;
    int padding  = params.padding;
    int dilation = params.dilation;

    const int64_t kW    = w->ne[0];
    const int64_t C_in  = w->ne[1];
    const int64_t C_out = w->ne[2];
    const int64_t L_in  = x->ne[0];
    const int64_t batch = (x->ne[2] > 0) ? x->ne[2] : 1;

    const int64_t L_out = (L_in + 2 * padding - dilation * (kW - 1) - 1) / stride + 1;
    GGML_ASSERT(L_out == dst->ne[0]);
    GGML_ASSERT(x->type   == GGML_TYPE_F32 || x->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);

    // Transpose weights to [C_out, kW, C_in] row-major layout using actual strides
    std::vector<float> w_transposed(C_out * kW * C_in);
    for (int64_t oc = 0; oc < C_out; ++oc) {
        for (int64_t ic = 0; ic < C_in; ++ic) {
            for (int64_t k = 0; k < kW; ++k) {
                float val = 0.0f;
                if (w->type == GGML_TYPE_F32) {
                    size_t offset = oc * w->nb[2] + ic * w->nb[1] + k * w->nb[0];
                    val = *(const float *)((const char *)w->data + offset);
                } else if (w->type == GGML_TYPE_F16) {
                    size_t offset = oc * w->nb[2] + ic * w->nb[1] + k * w->nb[0];
                    val = ggml_fp16_to_fp32(*(const ggml_fp16_t *)((const char *)w->data + offset));
                } else if (w->type == GGML_TYPE_Q8_0) {
                    const block_q8_0 * blocks = (const block_q8_0 *)w->data;
                    size_t flat_index = oc * (C_in * kW) + ic * kW + k;
                    size_t ib = flat_index / 32;
                    size_t is = flat_index % 32;
                    val = ggml_fp16_to_fp32(blocks[ib].d) * blocks[ib].qs[is];
                } else if (w->type == GGML_TYPE_Q4_0) {
                    const block_q4_0 * blocks = (const block_q4_0 *)w->data;
                    size_t flat_index = oc * (C_in * kW) + ic * kW + k;
                    size_t ib = flat_index / 32;
                    size_t is = flat_index % 32;
                    uint8_t vi = (blocks[ib].qs[is / 2] >> ((is % 2) * 4)) & 0x0F;
                    val = ggml_fp16_to_fp32(blocks[ib].d) * (vi - 8.0f);
                }
                w_transposed[oc * (kW * C_in) + k * C_in + ic] = val;
            }
        }
    }

    std::vector<float> x_transposed(L_in * C_in);

    // Calculate static boundaries for fast path (where padding checks are not needed)
    int64_t ow_start = (stride > 0) ? (padding + stride - 1) / stride : 0;
    int64_t ow_end = (stride > 0) ? (L_in + padding - (kW - 1) * dilation) / stride : 0;
    if (ow_end < ow_start) {
        ow_end = ow_start;
    }

    for (int64_t b = 0; b < batch; ++b) {
        // Transpose input to [L_in, C_in] row-major layout using actual strides
        // Swapping the loop order to make the read from x sequential (perfect cache line prefetching)
        #pragma omp parallel for collapse(2)
        for (int64_t ic = 0; ic < C_in; ++ic) {
            for (int64_t iw = 0; iw < L_in; ++iw) {
                size_t offset = b * x->nb[2] + ic * x->nb[1] + iw * x->nb[0];
                if (x->type == GGML_TYPE_F16) {
                    x_transposed[iw * C_in + ic] = ggml_fp16_to_fp32(*(const ggml_fp16_t *)((const char *)x->data + offset));
                } else {
                    x_transposed[iw * C_in + ic] = *(const float *)((const char *)x->data + offset);
                }
            }
        }

        // Direct Vectorized Convolution loop with 4x channel blocking
        // Parallelizing over ow (sequence length) and looping over oc (channels) internally
        // maximizes L1 cache reuse of the input vector x_transposed[iw] across output channels.
        #pragma omp parallel for
        for (int64_t ow = 0; ow < L_out; ++ow) {
            for (int64_t oc = 0; oc < C_out; oc += 4) {
                if (oc + 3 < C_out) {
                    const float * vec_w0_base = w_transposed.data() + (oc + 0) * (kW * C_in);
                    const float * vec_w1_base = w_transposed.data() + (oc + 1) * (kW * C_in);
                    const float * vec_w2_base = w_transposed.data() + (oc + 2) * (kW * C_in);
                    const float * vec_w3_base = w_transposed.data() + (oc + 3) * (kW * C_in);

                    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
                    if (ow >= ow_start && ow < ow_end) {
                        for (int64_t k = 0; k < kW; ++k) {
                            int64_t iw = ow * stride - padding + k * dilation;
                            const float * vec_x = x_transposed.data() + iw * C_in;
                            
                            float dot0, dot1, dot2, dot3;
                            inline_vec_dot_f32_x4((int)C_in, vec_x, 
                                                  vec_w0_base + k * C_in,
                                                  vec_w1_base + k * C_in,
                                                  vec_w2_base + k * C_in,
                                                  vec_w3_base + k * C_in,
                                                  &dot0, &dot1, &dot2, &dot3);
                            sum0 += dot0; sum1 += dot1; sum2 += dot2; sum3 += dot3;
                        }
                    } else {
                        for (int64_t k = 0; k < kW; ++k) {
                            int64_t iw = ow * stride - padding + k * dilation;
                            if (iw >= 0 && iw < L_in) {
                                const float * vec_x = x_transposed.data() + iw * C_in;
                                
                                float dot0, dot1, dot2, dot3;
                                inline_vec_dot_f32_x4((int)C_in, vec_x, 
                                                      vec_w0_base + k * C_in,
                                                      vec_w1_base + k * C_in,
                                                      vec_w2_base + k * C_in,
                                                      vec_w3_base + k * C_in,
                                                      &dot0, &dot1, &dot2, &dot3);
                                sum0 += dot0; sum1 += dot1; sum2 += dot2; sum3 += dot3;
                            }
                        }
                    }
                    if (dst->type == GGML_TYPE_F16) {
                        *(ggml_fp16_t *)((char *)dst->data + (oc + 0) * dst->nb[1] + b * dst->nb[2] + ow * dst->nb[0]) = ggml_fp32_to_fp16(sum0);
                        *(ggml_fp16_t *)((char *)dst->data + (oc + 1) * dst->nb[1] + b * dst->nb[2] + ow * dst->nb[0]) = ggml_fp32_to_fp16(sum1);
                        *(ggml_fp16_t *)((char *)dst->data + (oc + 2) * dst->nb[1] + b * dst->nb[2] + ow * dst->nb[0]) = ggml_fp32_to_fp16(sum2);
                        *(ggml_fp16_t *)((char *)dst->data + (oc + 3) * dst->nb[1] + b * dst->nb[2] + ow * dst->nb[0]) = ggml_fp32_to_fp16(sum3);
                    } else {
                        *(float *)((char *)dst->data + (oc + 0) * dst->nb[1] + b * dst->nb[2] + ow * dst->nb[0]) = sum0;
                        *(float *)((char *)dst->data + (oc + 1) * dst->nb[1] + b * dst->nb[2] + ow * dst->nb[0]) = sum1;
                        *(float *)((char *)dst->data + (oc + 2) * dst->nb[1] + b * dst->nb[2] + ow * dst->nb[0]) = sum2;
                        *(float *)((char *)dst->data + (oc + 3) * dst->nb[1] + b * dst->nb[2] + ow * dst->nb[0]) = sum3;
                    }
                } else {
                    // Fallback path for any remaining trailing channels
                    for (int64_t soc = oc; soc < C_out; ++soc) {
                        const float * vec_w_base = w_transposed.data() + soc * (kW * C_in);
                        float sum = 0.0f;
                        if (ow >= ow_start && ow < ow_end) {
                            for (int64_t k = 0; k < kW; ++k) {
                                int64_t iw = ow * stride - padding + k * dilation;
                                const float * vec_x = x_transposed.data() + iw * C_in;
                                sum += inline_vec_dot_f32((int)C_in, vec_x, vec_w_base + k * C_in);
                            }
                        } else {
                            for (int64_t k = 0; k < kW; ++k) {
                                int64_t iw = ow * stride - padding + k * dilation;
                                if (iw >= 0 && iw < L_in) {
                                    const float * vec_x = x_transposed.data() + iw * C_in;
                                    sum += inline_vec_dot_f32((int)C_in, vec_x, vec_w_base + k * C_in);
                                }
                            }
                        }
                        if (dst->type == GGML_TYPE_F16) {
                            *(ggml_fp16_t *)((char *)dst->data + soc * dst->nb[1] + b * dst->nb[2] + ow * dst->nb[0]) = ggml_fp32_to_fp16(sum);
                        } else {
                            *(float *)((char *)dst->data + soc * dst->nb[1] + b * dst->nb[2] + ow * dst->nb[0]) = sum;
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
