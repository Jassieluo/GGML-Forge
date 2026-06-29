#include "ops/ops.h"
#include "ggml.h"
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
    const int step = 32;
    const int epr = 8;
    int np = (n & ~(step - 1));
    if (np > 0) {
        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps();
        __m256 s3 = _mm256_setzero_ps();
        for (; i < np; i += step) {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i),          _mm256_loadu_ps(y + i),          s0);
            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + epr),      _mm256_loadu_ps(y + i + epr),      s1);
            s2 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 2 * epr),  _mm256_loadu_ps(y + i + 2 * epr),  s2);
            s3 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 3 * epr),  _mm256_loadu_ps(y + i + 3 * epr),  s3);
        }
        __m256 s01 = _mm256_add_ps(s0, s1);
        __m256 s23 = _mm256_add_ps(s2, s3);
        __m256 s = _mm256_add_ps(s01, s23);
        __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
        t0 = _mm_hadd_ps(t0, t0);
        result = _mm_cvtss_f32(_mm_hadd_ps(t0, t0));
    }
#endif
    for (; i < n; ++i) {
        result += x[i] * y[i];
    }
    return result;
}

inline void inline_vec_dot_f32_x4(int n, const float * x, 
                                  const float * y0, const float * y1, const float * y2, const float * y3,
                                  float * r0, float * r1, float * r2, float * r3) {
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    int i = 0;
#if defined(__AVX2__)
    const int step = 8;
    int np = (n & ~(step - 1));
    if (np > 0) {
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
        auto horizontal_add = [](__m256 s) -> float {
            __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
            t0 = _mm_hadd_ps(t0, t0);
            return _mm_cvtss_f32(_mm_hadd_ps(t0, t0));
        };
        sum0 = horizontal_add(s0);
        sum1 = horizontal_add(s1);
        sum2 = horizontal_add(s2);
        sum3 = horizontal_add(s3);
    }
#endif
    for (; i < n; ++i) {
        float vx = x[i];
        sum0 += vx * y0[i];
        sum1 += vx * y1[i];
        sum2 += vx * y2[i];
        sum3 += vx * y3[i];
    }
    *r0 = sum0;
    *r1 = sum1;
    *r2 = sum2;
    *r3 = sum3;
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
    GGML_ASSERT(x->type   == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const float * x_d   = (const float *)x->data;
    float *       dst_d = (float *)dst->data;

    // Pre-convert F16 weight → F32
    std::vector<float> w_f32;
    const float * w_d = nullptr;
    if (w->type == GGML_TYPE_F32) {
        w_d = (const float *)w->data;
    } else if (w->type == GGML_TYPE_F16) {
        int64_t n = ggml_nelements(w);
        w_f32.resize(n);
        const ggml_fp16_t * w16 = (const ggml_fp16_t *)w->data;
        for (int64_t i = 0; i < n; ++i) w_f32[i] = ggml_fp16_to_fp32(w16[i]);
        w_d = w_f32.data();
    } else return false;

    // Transpose weights to [C_out, kW, C_in] row-major layout
    std::vector<float> w_transposed(C_out * kW * C_in);
    for (int64_t oc = 0; oc < C_out; ++oc) {
        for (int64_t k = 0; k < kW; ++k) {
            for (int64_t ic = 0; ic < C_in; ++ic) {
                w_transposed[oc * (kW * C_in) + k * C_in + ic] = w_d[oc * (C_in * kW) + ic * kW + k];
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
        const float * x_b = x_d + b * (C_in * L_in);
        float * dst_b = dst_d + b * (C_out * L_out);

        // Transpose input to [L_in, C_in] row-major layout
        #pragma omp parallel for collapse(2)
        for (int64_t iw = 0; iw < L_in; ++iw) {
            for (int64_t ic = 0; ic < C_in; ++ic) {
                x_transposed[iw * C_in + ic] = x_b[ic * L_in + iw];
            }
        }

        // Direct Vectorized Convolution loop with 4x channel blocking
        // Reduces input vector loads by 4x and avoids L1 Cache Port contention
        #pragma omp parallel for
        for (int64_t oc = 0; oc < C_out; oc += 4) {
            if (oc + 3 < C_out) {
                const float * vec_w0_base = w_transposed.data() + (oc + 0) * (kW * C_in);
                const float * vec_w1_base = w_transposed.data() + (oc + 1) * (kW * C_in);
                const float * vec_w2_base = w_transposed.data() + (oc + 2) * (kW * C_in);
                const float * vec_w3_base = w_transposed.data() + (oc + 3) * (kW * C_in);

                for (int64_t ow = 0; ow < L_out; ++ow) {
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
                    dst_b[(oc + 0) * L_out + ow] = sum0;
                    dst_b[(oc + 1) * L_out + ow] = sum1;
                    dst_b[(oc + 2) * L_out + ow] = sum2;
                    dst_b[(oc + 3) * L_out + ow] = sum3;
                }
            } else {
                // Fallback path for any remaining trailing channels
                for (int64_t soc = oc; soc < C_out; ++soc) {
                    const float * vec_w_base = w_transposed.data() + soc * (kW * C_in);
                    for (int64_t ow = 0; ow < L_out; ++ow) {
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
                        dst_b[soc * L_out + ow] = sum;
                    }
                }
            }
        }
    }

    return true;
}

} // namespace cpu
} // namespace ggml_ops_ext
