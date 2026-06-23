#include "ops/ops.h"
#include "ops_sycl.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "common.hpp" // From ggml-sycl
#include <iostream>

namespace ggml_ops_ext {
namespace sycl {

class MishSYCLKernel;
class MishStridedSYCLKernel;

bool ggml_sycl_op_mish(
    ggml_backend_t backend,
    struct ggml_tensor* x,
    struct ggml_tensor* dst
) {
    ::sycl::queue* q = (::sycl::queue*)ggml_ops_ext_bridge_sycl_get_queue(backend);
    if (!q) return false;
    if (!q) return false;

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const float* x_d = (const float*)x->data;
    float* dst_d = (float*)dst->data;

    int64_t nelements = ggml_nelements(dst);

    if (ggml_is_contiguous(x) && ggml_is_contiguous(dst)) {
        q->submit([&](::sycl::handler &cgh) {
            cgh.parallel_for<MishSYCLKernel>(
                ::sycl::range<1>(nelements),
                [=](::sycl::id<1> id) {
                    int64_t idx = id[0];
                    float val = x_d[idx];
                    float clamped_val = ::sycl::fmax(-20.0f, ::sycl::fmin(val, 20.0f));
                    float ex = ::sycl::exp(clamped_val);
                    float ex1 = ex + 1.0f;
                    float ex1_sq = ex1 * ex1;
                    dst_d[idx] = val * (ex1_sq - 1.0f) / (ex1_sq + 1.0f);
                }
            );
        });
    } else {
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

        int64_t total = ne0 * ne1 * ne2 * ne3;

        q->submit([&](::sycl::handler &cgh) {
            cgh.parallel_for<MishStridedSYCLKernel>(
                ::sycl::range<1>(total),
                [=](::sycl::id<1> id) {
                    int64_t idx = id[0];
                    int64_t i0 = idx % ne0;
                    int64_t tmp = idx / ne0;
                    int64_t i1 = tmp % ne1;
                    tmp = tmp / ne1;
                    int64_t i2 = tmp % ne2;
                    int64_t i3 = tmp / ne2;

                    const float* px = (const float*)((const char*)x_d + i3*nb_x3 + i2*nb_x2 + i1*nb_x1 + i0*nb_x0);
                    float* pdst = (float*)((char*)dst_d + i3*nb_dst3 + i2*nb_dst2 + i1*nb_dst1 + i0*nb_dst0);

                    float val = *px;
                    float clamped_val = ::sycl::fmax(-20.0f, ::sycl::fmin(val, 20.0f));
                    float ex = ::sycl::exp(clamped_val);
                    float ex1 = ex + 1.0f;
                    float ex1_sq = ex1 * ex1;
                    *pdst = val * (ex1_sq - 1.0f) / (ex1_sq + 1.0f);
                }
            );
        });
    }

    q->wait();
    return true;
}

bool ggml_sycl_op_mish_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    return ggml_sycl_op_mish(backend, node->src[0], node);
}

} // namespace sycl
} // namespace ggml_ops_ext
