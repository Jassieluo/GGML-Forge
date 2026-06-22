#include "ops/ops.h"
#include "ggml.h"
#include <cmath>
#include <cstring>
#include <cstdio>

namespace ggml_ops_ext {
namespace cpu {

bool ops_cpu_op_layer_norm(ggml_backend_t backend, struct ggml_tensor* node) {
    (void)backend;

    struct ggml_tensor* x = node->src[0];
    struct ggml_tensor* gamma = node->src[1];
    struct ggml_tensor* beta = node->src[2];
    struct ggml_tensor* dst = node;

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(gamma->type == GGML_TYPE_F32);
    GGML_ASSERT(beta->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    float eps;
    std::memcpy(&eps, node->op_params, sizeof(float));

    const float* x_d = (const float*)x->data;
    const float* gamma_d = (const float*)gamma->data;
    const float* beta_d = (const float*)beta->data;
    float* dst_d = (float*)dst->data;

    int64_t ne0 = dst->ne[0]; // Columns
    int64_t ne1 = dst->ne[1]; // Rows
    int64_t ne2 = dst->ne[2]; // Batch
    int64_t ne3 = dst->ne[3];

    size_t nb_x0 = x->nb[0];
    size_t nb_x1 = x->nb[1];
    size_t nb_x2 = x->nb[2];
    size_t nb_x3 = x->nb[3];

    size_t nb_gamma0 = gamma->nb[0];
    size_t nb_beta0 = beta->nb[0];

    size_t nb_dst0 = dst->nb[0];
    size_t nb_dst1 = dst->nb[1];
    size_t nb_dst2 = dst->nb[2];
    size_t nb_dst3 = dst->nb[3];

    #pragma omp parallel for collapse(3)
    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            for (int64_t i1 = 0; i1 < ne1; ++i1) {
                // Compute mean of this row
                double sum = 0.0;
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    const float* px = (const float*)((const char*)x_d + i3*nb_x3 + i2*nb_x2 + i1*nb_x1 + i0*nb_x0);
                    sum += *px;
                }
                float mean = (float)(sum / ne0);

                // Compute variance of this row
                double sum_sq = 0.0;
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    const float* px = (const float*)((const char*)x_d + i3*nb_x3 + i2*nb_x2 + i1*nb_x1 + i0*nb_x0);
                    float diff = *px - mean;
                    sum_sq += diff * diff;
                }
                float variance = (float)(sum_sq / ne0);
                float inv_std = 1.0f / std::sqrt(variance + eps);

                // Normalize, scale and shift
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    const float* px = (const float*)((const char*)x_d + i3*nb_x3 + i2*nb_x2 + i1*nb_x1 + i0*nb_x0);
                    const float* pgamma = (const float*)((const char*)gamma_d + i0*nb_gamma0);
                    const float* pbeta = (const float*)((const char*)beta_d + i0*nb_beta0);

                    float* pdst = (float*)((char*)dst_d + i3*nb_dst3 + i2*nb_dst2 + i1*nb_dst1 + i0*nb_dst0);

                    float norm_val = (*px - mean) * inv_std;
                    *pdst = norm_val * (*pgamma) + (*pbeta);
                }
            }
        }
    }

    return true;
}

} // namespace cpu
} // namespace ggml_ops_ext
