#include "ops/ops.h"
#include "ops_sycl.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "common.hpp" // From ggml-sycl
#include <iostream>

namespace ggml_ops_ext {
namespace sycl {

class LayerNormSYCLKernel;

bool ggml_sycl_op_layer_norm(
    ggml_backend_t backend,
    struct ggml_tensor* x,
    struct ggml_tensor* gamma,
    struct ggml_tensor* beta,
    struct ggml_tensor* dst,
    float eps
) {
    ::sycl::queue* q = (::sycl::queue*)ggml_ops_ext_bridge_sycl_get_queue(backend);
    if (!q) return false;
    if (!q) return false;

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(gamma->type == GGML_TYPE_F32);
    GGML_ASSERT(beta->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

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

    int64_t num_rows = ne1 * ne2 * ne3;

    q->submit([&](::sycl::handler &cgh) {
        cgh.parallel_for<LayerNormSYCLKernel>(
            ::sycl::range<1>(num_rows),
            [=](::sycl::id<1> id) {
                int64_t bid = id[0];
                int64_t i1 = bid % ne1;
                int64_t tmp = bid / ne1;
                int64_t i2 = tmp % ne2;
                int64_t i3 = tmp / ne2;

                if (i3 < ne3) {
                    // 1. Calculate mean
                    double sum = 0.0;
                    for (int64_t i0 = 0; i0 < ne0; ++i0) {
                        const float* px = (const float*)((const char*)x_d + i3*nb_x3 + i2*nb_x2 + i1*nb_x1 + i0*nb_x0);
                        sum += *px;
                    }
                    float mean = (float)(sum / ne0);

                    // 2. Calculate variance
                    double sum_sq = 0.0;
                    for (int64_t i0 = 0; i0 < ne0; ++i0) {
                        const float* px = (const float*)((const char*)x_d + i3*nb_x3 + i2*nb_x2 + i1*nb_x1 + i0*nb_x0);
                        float diff = *px - mean;
                        sum_sq += diff * diff;
                    }
                    float variance = (float)(sum_sq / ne0);
                    float inv_std = 1.0f / ::sycl::sqrt(variance + eps);

                    // 3. Normalize, scale, and shift
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
        );
    });

    q->wait();
    return true;
}

bool ggml_sycl_op_layer_norm_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    float eps;
    std::memcpy(&eps, node->op_params, sizeof(float));
    return ggml_sycl_op_layer_norm(backend, node->src[0], node->src[1], node->src[2], node, eps);
}

} // namespace sycl
} // namespace ggml_ops_ext
