#include "ops/ops.h"
#include "ops/cpu.h"
#include "ggml.h"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

namespace ggml_ops_ext {
namespace cpu {

static bool cpu_has_avx2() {
    static bool has = []() {
#if defined(_MSC_VER)
        int cpuInfo[4];
        __cpuid(cpuInfo, 0);
        if (cpuInfo[0] < 7) return false;
        __cpuidex(cpuInfo, 7, 0);
        return (cpuInfo[1] & (1 << 5)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
        unsigned int eax, ebx, ecx, edx;
        if (__get_cpuid_max(0, nullptr) < 7) return false;
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        return (ebx & (1 << 5)) != 0;
#else
        return false;
#endif
    }();
    return has;
}

static bool cpu_has_avx512() {
    static bool has = []() {
#if defined(_MSC_VER)
        int cpuInfo[4];
        __cpuid(cpuInfo, 0);
        if (cpuInfo[0] < 7) return false;
        __cpuidex(cpuInfo, 7, 0);
        return (cpuInfo[1] & (1 << 16)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
        unsigned int eax, ebx, ecx, edx;
        if (__get_cpuid_max(0, nullptr) < 7) return false;
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        return (ebx & (1 << 16)) != 0;
#else
        return false;
#endif
    }();
    return has;
}

// Declarations of optimized implementations
void ggml_vec_ext_double_swish_f32_avx2(const int n, float * y, const float * x);
void ggml_vec_ext_double_swish_f32_avx512(const int n, float * y, const float * x);

void ggml_vec_ext_double_swish_f32(const int n, float * y, const float * x) {
    // Fallback generic scalar implementation
    for (int i = 0; i < n; ++i) {
        float val = x[i];
        float neg_xm1 = -(val - 1.0f);
        float clamped = std::max(-20.0f, std::min(neg_xm1, 20.0f));
        y[i] = val / (1.0f + std::exp(clamped));
    }
}

bool ops_cpu_op_double_swish(ggml_backend_t backend, struct ggml_tensor* node) {
    const int omp_threads = backend_thread_count(backend);

    struct ggml_tensor* x = node->src[0];
    struct ggml_tensor* dst = node;

    const float* x_d = (const float*)x->data;
    float* dst_d = (float*)dst->data;

    int64_t nelements = ggml_nelements(dst);

    int64_t ne0 = dst->ne[0];
    int64_t ne1 = dst->ne[1];
    int64_t ne2 = dst->ne[2];
    int64_t ne3 = dst->ne[3];

    size_t nb_x0 = x->nb[0];
    size_t nb_x1 = x->nb[1];
    size_t nb_x2 = x->nb[2];
    size_t nb_x3 = x->nb[3];

    size_t nb_dst0 = dst->nb[0];
    size_t nb_dst1 = dst->nb[1];
    size_t nb_dst2 = dst->nb[2];
    size_t nb_dst3 = dst->nb[3];

    if (x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        if (ggml_is_contiguous(x) && ggml_is_contiguous(dst)) {
            ggml_vec_ext_double_swish_f32(nelements, dst_d, x_d);
        } else if (nb_x0 == sizeof(float) && nb_dst0 == sizeof(float)) {
            #pragma omp parallel for collapse(3) num_threads(omp_threads)
            for (int64_t i3 = 0; i3 < ne3; ++i3) {
                for (int64_t i2 = 0; i2 < ne2; ++i2) {
                    for (int64_t i1 = 0; i1 < ne1; ++i1) {
                        const float* px = (const float*)((const char*)x_d + i3*nb_x3 + i2*nb_x2 + i1*nb_x1);
                        float* pdst = (float*)((char*)dst_d + i3*nb_dst3 + i2*nb_dst2 + i1*nb_dst1);
                        ggml_vec_ext_double_swish_f32(ne0, pdst, px);
                    }
                }
            }
        } else {
            for (int64_t i3 = 0; i3 < ne3; ++i3) {
                for (int64_t i2 = 0; i2 < ne2; ++i2) {
                    for (int64_t i1 = 0; i1 < ne1; ++i1) {
                        for (int64_t i0 = 0; i0 < ne0; ++i0) {
                            const float* px = (const float*)((const char*)x_d + i3*nb_x3 + i2*nb_x2 + i1*nb_x1 + i0*nb_x0);
                            float* pdst = (float*)((char*)dst_d + i3*nb_dst3 + i2*nb_dst2 + i1*nb_dst1 + i0*nb_dst0);
                            float val = *px;
                            float neg_xm1 = -(val - 1.0f);
                            float clamped = std::max(-20.0f, std::min(neg_xm1, 20.0f));
                            *pdst = val / (1.0f + std::exp(clamped));
                        }
                    }
                }
            }
        }
    } else {
        // F16/mixed type path with local float buffer conversion to keep AVX performance
        #pragma omp parallel num_threads(omp_threads)
        {
            std::vector<float> x_buf(ne0);
            std::vector<float> dst_buf(ne0);
            #pragma omp for collapse(3)
            for (int64_t i3 = 0; i3 < ne3; ++i3) {
                for (int64_t i2 = 0; i2 < ne2; ++i2) {
                    for (int64_t i1 = 0; i1 < ne1; ++i1) {
                        for (int64_t i0 = 0; i0 < ne0; ++i0) {
                            const void* px = (const char*)x->data + i3*nb_x3 + i2*nb_x2 + i1*nb_x1 + i0*nb_x0;
                            if (x->type == GGML_TYPE_F16) {
                                x_buf[i0] = ggml_fp16_to_fp32(*(const ggml_fp16_t*)px);
                            } else {
                                x_buf[i0] = *(const float*)px;
                            }
                        }
                        ggml_vec_ext_double_swish_f32(ne0, dst_buf.data(), x_buf.data());
                        for (int64_t i0 = 0; i0 < ne0; ++i0) {
                            void* pdst = (char*)dst->data + i3*nb_dst3 + i2*nb_dst2 + i1*nb_dst1 + i0*nb_dst0;
                            if (dst->type == GGML_TYPE_F16) {
                                *(ggml_fp16_t*)pdst = ggml_fp32_to_fp16(dst_buf[i0]);
                            } else {
                                *(float*)pdst = dst_buf[i0];
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
