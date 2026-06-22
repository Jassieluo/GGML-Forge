#include "ops/ops.h"
#include "ggml.h"
#include <cmath>
#include <cstdio>
#include <algorithm>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

namespace ggml_ops_ext {
namespace cpu {

// Custom self-contained runtime CPUID checks
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

// Declarations of optimized implementation in other files
void ggml_vec_ext_mish_f32_avx2(const int n, float * y, const float * x);
void ggml_vec_ext_mish_f32_avx512(const int n, float * y, const float * x);

// Dynamic SIMD dispatcher for Mish
void ggml_vec_ext_mish_f32(const int n, float * y, const float * x) {
    if (cpu_has_avx512()) {
        ggml_vec_ext_mish_f32_avx512(n, y, x);
    } else if (cpu_has_avx2()) {
        ggml_vec_ext_mish_f32_avx2(n, y, x);
    } else {
        // Fallback generic scalar implementation
        for (int i = 0; i < n; ++i) {
            float val = x[i];
            float clamped = std::max(-20.0f, std::min(val, 20.0f));
            float ex = std::exp(clamped);
            float ex1 = ex + 1.0f;
            float ex1_sq = ex1 * ex1;
            y[i] = val * (ex1_sq - 1.0f) / (ex1_sq + 1.0f);
        }
    }
}

bool ops_cpu_op_mish(ggml_backend_t backend, struct ggml_tensor* node) {
    (void)backend;
    
    struct ggml_tensor* x = node->src[0];
    struct ggml_tensor* dst = node;

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

    // If both are contiguous, we can loop over all elements directly
    if (ggml_is_contiguous(x) && ggml_is_contiguous(dst)) {
        ggml_vec_ext_mish_f32(nelements, dst_d, x_d);
    } else if (nb_x0 == sizeof(float) && nb_dst0 == sizeof(float)) {
        // Optimized row-by-row dispatch
        #pragma omp parallel for collapse(3)
        for (int64_t i3 = 0; i3 < ne3; ++i3) {
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                for (int64_t i1 = 0; i1 < ne1; ++i1) {
                    const float* px = (const float*)((const char*)x_d + i3*nb_x3 + i2*nb_x2 + i1*nb_x1);
                    float* pdst = (float*)((char*)dst_d + i3*nb_dst3 + i2*nb_dst2 + i1*nb_dst1);
                    ggml_vec_ext_mish_f32(ne0, pdst, px);
                }
            }
        }
    } else {
        // Fallback for non-standard strides
        for (int64_t i3 = 0; i3 < ne3; ++i3) {
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                for (int64_t i1 = 0; i1 < ne1; ++i1) {
                    for (int64_t i0 = 0; i0 < ne0; ++i0) {
                        const float* px = (const float*)((const char*)x_d + i3*nb_x3 + i2*nb_x2 + i1*nb_x1 + i0*nb_x0);
                        float* pdst = (float*)((char*)dst_d + i3*nb_dst3 + i2*nb_dst2 + i1*nb_dst1 + i0*nb_dst0);
                        float val = *px;
                        float clamped_val = std::max(-20.0f, std::min(val, 20.0f));
                        float ex = std::exp(clamped_val);
                        float ex1 = ex + 1.0f;
                        float ex1_sq = ex1 * ex1;
                        *pdst = val * (ex1_sq - 1.0f) / (ex1_sq + 1.0f);
                    }
                }
            }
        }
    }
    return true;
}

} // namespace cpu
} // namespace ggml_ops_ext
