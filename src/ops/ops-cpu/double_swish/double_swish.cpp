#include "ops/ops.h"
#include "ggml.h"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <algorithm>

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
    (void)backend;

    struct ggml_tensor* x = node->src[0];
    struct ggml_tensor* dst = node;

    std::cout << "[T2S CPU Trace] double_swish node: " << node->name 
              << ", x: " << (x ? x->name : "null")
              << ", x->type: " << (x ? (int)x->type : -1)
              << ", x->data: " << (x ? x->data : nullptr)
              << ", dst->data: " << dst->data
              << ", nelements: " << ggml_nelements(dst)
              << ", x_contig: " << ggml_is_contiguous(x)
              << ", dst_contig: " << ggml_is_contiguous(dst)
              << ", ne: " << dst->ne[0] << "x" << dst->ne[1] << "x" << dst->ne[2] << "x" << dst->ne[3]
              << ", nb_x: " << x->nb[0] << "," << x->nb[1] << "," << x->nb[2] << "," << x->nb[3]
              << ", nb_dst: " << dst->nb[0] << "," << dst->nb[1] << "," << dst->nb[2] << "," << dst->nb[3] << "\n";
    std::fflush(stdout);

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

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

    if (ggml_is_contiguous(x) && ggml_is_contiguous(dst)) {
        ggml_vec_ext_double_swish_f32(nelements, dst_d, x_d);
    } else if (nb_x0 == sizeof(float) && nb_dst0 == sizeof(float)) {
        #pragma omp parallel for collapse(3)
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
    return true;
}

} // namespace cpu
} // namespace ggml_ops_ext
