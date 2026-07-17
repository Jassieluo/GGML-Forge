#include "ops/ops.h"
#include "ops/cpu.h"
#include "ops_cpu_common.h"
#include <cmath>
#include <omp.h>
#include <algorithm>
#include <vector>

namespace ggml_ops_ext {
namespace cpu {

template <typename Tx, typename Td>
static void compute_instance_norm(
    const Tx* x_d, const float* gamma_d, const float* beta_d, Td* dst_d,
    int64_t T, int64_t C, float eps, int omp_threads
) {
    #pragma omp parallel for num_threads(omp_threads)
    for (int64_t c = 0; c < C; ++c) {
        // Step 1: Compute mean
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (int64_t t = 0; t < T; ++t) {
            sum += read_val(&x_d[c * T + t]);
        }
        float mean = sum / T;

        // Step 2: Compute variance
        float sum_sq = 0.0f;
        #pragma omp simd reduction(+:sum_sq)
        for (int64_t t = 0; t < T; ++t) {
            float diff = read_val(&x_d[c * T + t]) - mean;
            sum_sq += diff * diff;
        }
        float var = sum_sq / T;
        float inv_std = 1.0f / std::sqrt(var + eps);

        float g = gamma_d ? gamma_d[c] : 1.0f;
        float b = beta_d ? beta_d[c] : 0.0f;

        // Step 3: Normalize, scale, and shift
        for (int64_t t = 0; t < T; ++t) {
            float val = (read_val(&x_d[c * T + t]) - mean) * inv_std * g + b;
            write_val(&dst_d[c * T + t], val);
        }
    }
}

bool ops_cpu_op_instance_norm(ggml_backend_t backend, struct ggml_tensor* node) {
    const int omp_threads = backend_thread_count(backend);
    if ((int)node->op != GGML_OP_OPS_VIRT_INSTANCE_NORM) return false;

    ops_instance_norm_params params;
    if (!ops_extract_instance_norm_params(node, params)) return false;

    struct ggml_tensor* x = params.x;
    struct ggml_tensor* gamma = params.gamma;
    struct ggml_tensor* beta = params.beta;
    struct ggml_tensor* dst = node;
    float eps = params.eps;

    int64_t T = x->ne[0];
    int64_t C = x->ne[1];

    std::vector<float> gamma_f32;
    if (gamma) {
        if (gamma->type == GGML_TYPE_F16) {
            gamma_f32.resize(C);
            const ggml_fp16_t* p = (const ggml_fp16_t*)gamma->data;
            for (int64_t c = 0; c < C; ++c) gamma_f32[c] = ggml_fp16_to_fp32(p[c]);
        }
    }
    const float* gamma_ptr = gamma ? (gamma->type == GGML_TYPE_F32 ? (const float*)gamma->data : gamma_f32.data()) : nullptr;

    std::vector<float> beta_f32;
    if (beta) {
        if (beta->type == GGML_TYPE_F16) {
            beta_f32.resize(C);
            const ggml_fp16_t* p = (const ggml_fp16_t*)beta->data;
            for (int64_t c = 0; c < C; ++c) beta_f32[c] = ggml_fp16_to_fp32(p[c]);
        }
    }
    const float* beta_ptr = beta ? (beta->type == GGML_TYPE_F32 ? (const float*)beta->data : beta_f32.data()) : nullptr;

    if (x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        compute_instance_norm((const float*)x->data, gamma_ptr, beta_ptr, (float*)dst->data, T, C, eps, omp_threads);
    } else if (x->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
        compute_instance_norm((const ggml_fp16_t*)x->data, gamma_ptr, beta_ptr, (float*)dst->data, T, C, eps, omp_threads);
    } else if (x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        compute_instance_norm((const float*)x->data, gamma_ptr, beta_ptr, (ggml_fp16_t*)dst->data, T, C, eps, omp_threads);
    } else if (x->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        compute_instance_norm((const ggml_fp16_t*)x->data, gamma_ptr, beta_ptr, (ggml_fp16_t*)dst->data, T, C, eps, omp_threads);
    } else {
        return false;
    }

    return true;
}

} // namespace cpu
} // namespace ggml_ops_ext
