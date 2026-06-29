#include "ops/ops.h"
#include <cmath>
#include <omp.h>
#include <algorithm>

namespace ggml_ops_ext {
namespace cpu {

bool ops_cpu_op_instance_norm(ggml_backend_t backend, struct ggml_tensor* node) {
    (void)backend;
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

    const float* x_d = (const float*)x->data;
    const float* gamma_d = gamma ? (const float*)gamma->data : nullptr;
    const float* beta_d = beta ? (const float*)beta->data : nullptr;
    float* dst_d = (float*)dst->data;

    #pragma omp parallel for
    for (int64_t c = 0; c < C; ++c) {
        // Step 1: Compute mean
        double sum = 0.0;
        for (int64_t t = 0; t < T; ++t) {
            sum += x_d[c * T + t];
        }
        float mean = (float)(sum / T);

        // Step 2: Compute variance
        double sum_sq = 0.0;
        for (int64_t t = 0; t < T; ++t) {
            float diff = x_d[c * T + t] - mean;
            sum_sq += diff * diff;
        }
        float var = (float)(sum_sq / T);
        float inv_std = 1.0f / std::sqrt(var + eps);

        float g = gamma_d ? gamma_d[c] : 1.0f;
        float b = beta_d ? beta_d[c] : 0.0f;

        // Step 3: Normalize, scale, and shift
        for (int64_t t = 0; t < T; ++t) {
            dst_d[c * T + t] = (x_d[c * T + t] - mean) * inv_std * g + b;
        }
    }

    return true;
}

} // namespace cpu
} // namespace ggml_ops_ext
