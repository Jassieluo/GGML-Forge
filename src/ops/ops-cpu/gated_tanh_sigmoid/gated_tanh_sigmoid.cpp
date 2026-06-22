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
void ggml_vec_ext_gated_tanh_sigmoid_f32_avx2(const int n, float * y, const float * xa, const float * xb);
void ggml_vec_ext_gated_tanh_sigmoid_f32_avx512(const int n, float * y, const float * xa, const float * xb);

// Dynamic SIMD dispatcher for Gated Tanh Sigmoid
void ggml_vec_ext_gated_tanh_sigmoid_f32(const int n, float * y, const float * xa, const float * xb) {
    if (cpu_has_avx512()) {
        ggml_vec_ext_gated_tanh_sigmoid_f32_avx512(n, y, xa, xb);
    } else if (cpu_has_avx2()) {
        ggml_vec_ext_gated_tanh_sigmoid_f32_avx2(n, y, xa, xb);
    } else {
        // Fallback generic scalar implementation
        for (int i = 0; i < n; ++i) {
            float val_a = xa[i];
            float val_b = xb[i];
            y[i] = std::tanh(val_a) * (1.0f / (1.0f + std::exp(-val_b)));
        }
    }
}

bool ops_cpu_op_gated_tanh_sigmoid(ggml_backend_t backend, struct ggml_tensor* node) {
    (void)backend;

    struct ggml_tensor* x = node->src[0];
    struct ggml_tensor* dst = node;

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    int32_t* params = (int32_t*)node->op_params;
    int hidden_channels = params[0];

    const float* x_d = (const float*)x->data;
    float* dst_d = (float*)dst->data;

    int64_t ne0 = dst->ne[0]; // C (hidden_channels)
    int64_t ne1 = dst->ne[1]; // T (seq_len)
    int64_t ne2 = dst->ne[2]; // batch
    int64_t ne3 = dst->ne[3];

    // Strides
    size_t nb_x0 = x->nb[0];
    size_t nb_x1 = x->nb[1];
    size_t nb_x2 = x->nb[2];
    size_t nb_x3 = x->nb[3];

    size_t nb_dst0 = dst->nb[0];
    size_t nb_dst1 = dst->nb[1];
    size_t nb_dst2 = dst->nb[2];
    size_t nb_dst3 = dst->nb[3];

    // Hidden channels C must be equal to ne0
    GGML_ASSERT(hidden_channels == ne0);

    if (nb_x0 == sizeof(float) && nb_dst0 == sizeof(float)) {
        #pragma omp parallel for collapse(3)
        for (int64_t i3 = 0; i3 < ne3; ++i3) {
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                for (int64_t i1 = 0; i1 < ne1; ++i1) {
                    const float* px_l = (const float*)((const char*)x_d + i3*nb_x3 + i2*nb_x2 + i1*nb_x1);
                    const float* px_r = (const float*)((const char*)x_d + i3*nb_x3 + i2*nb_x2 + i1*nb_x1 + hidden_channels * nb_x0);
                    float* pdst = (float*)((char*)dst_d + i3*nb_dst3 + i2*nb_dst2 + i1*nb_dst1);
                    ggml_vec_ext_gated_tanh_sigmoid_f32(ne0, pdst, px_l, px_r);
                }
            }
        }
    } else {
        // Fallback for non-standard strides
        for (int64_t i3 = 0; i3 < ne3; ++i3) {
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                for (int64_t i1 = 0; i1 < ne1; ++i1) {
                    for (int64_t i0 = 0; i0 < ne0; ++i0) {
                        // Left half (tanh part): channel i0
                        const float* px_l = (const float*)((const char*)x_d + i3*nb_x3 + i2*nb_x2 + i1*nb_x1 + i0*nb_x0);
                        // Right half (sigmoid part): channel i0 + hidden_channels
                        const float* px_r = (const float*)((const char*)x_d + i3*nb_x3 + i2*nb_x2 + i1*nb_x1 + (i0 + hidden_channels)*nb_x0);

                        float* pdst = (float*)((char*)dst_d + i3*nb_dst3 + i2*nb_dst2 + i1*nb_dst1 + i0*nb_dst0);

                        float val_l = *px_l;
                        float val_r = *px_r;

                        // Tanh calculation: clamp val_l to [-10.0, 10.0] as it is scaled by 2.0
                        float clamped_l = std::max(-10.0f, std::min(val_l, 10.0f));
                        float sigm_l = 1.0f / (1.0f + std::expf(-2.0f * clamped_l));
                        float tanh_val = 2.0f * sigm_l - 1.0f;

                        // Sigmoid calculation: clamp val_r to [-20.0, 20.0]
                        float clamped_r = std::max(-20.0f, std::min(val_r, 20.0f));
                        float sigm_r = 1.0f / (1.0f + std::expf(-clamped_r));

                        *pdst = tanh_val * sigm_r;
                    }
                }
            }
        }
    }

    return true;
}

} // namespace cpu
} // namespace ggml_ops_ext
