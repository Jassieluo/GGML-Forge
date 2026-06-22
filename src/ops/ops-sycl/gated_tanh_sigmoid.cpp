#include "ops/ops.h"
#include "ops_sycl.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "common.hpp" // From ggml-sycl
#include <iostream>

namespace ggml_ops_ext {
namespace sycl {

class GatedTanhSigmoidSYCLKernel;

bool ggml_sycl_op_gated_tanh_sigmoid(
    ggml_backend_t backend,
    struct ggml_tensor* x,
    struct ggml_tensor* dst,
    int hidden_channels
) {
    ggml_backend_sycl_context* sycl_ctx = (ggml_backend_sycl_context*)backend->context;
    if (!sycl_ctx) return false;
    ::sycl::queue* q = sycl_ctx->stream();
    if (!q) return false;

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const float* x_d = (const float*)x->data;
    float* dst_d = (float*)dst->data;

    int64_t seq_len = dst->ne[1];
    int64_t batch = dst->ne[2];

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

    int64_t total = hidden_channels * seq_len * batch;

    q->submit([&](::sycl::handler &cgh) {
        cgh.parallel_for<GatedTanhSigmoidSYCLKernel>(
            ::sycl::range<1>(total),
            [=](::sycl::id<1> id) {
                int64_t idx = id[0];
                int64_t i0 = idx % hidden_channels;
                int64_t tmp = idx / hidden_channels;
                int64_t i1 = tmp % seq_len;
                int64_t i2 = tmp / seq_len;

                // Left: channel i0
                const float* px_l = (const float*)((const char*)x_d + i2*nb_x2 + i1*nb_x1 + i0*nb_x0);
                // Right: channel i0 + hidden_channels
                const float* px_r = (const float*)((const char*)x_d + i2*nb_x2 + i1*nb_x1 + (i0 + hidden_channels)*nb_x0);

                float* pdst = (float*)((char*)dst_d + i2*nb_dst2 + i1*nb_dst1 + i0*nb_dst0);

                float val_l = *px_l;
                float val_r = *px_r;

                // Tanh: clamp val_l to [-10.f, 10.f] as it is multiplied by -2.f inside sycl::exp
                float clamped_l = ::sycl::fmax(-10.0f, ::sycl::fmin(val_l, 10.0f));
                float sigm_l = 1.0f / (1.0f + ::sycl::exp(-2.0f * clamped_l));
                float tanh_val = 2.0f * sigm_l - 1.0f;

                // Sigmoid: clamp val_r to [-20.f, 20.f]
                float clamped_r = ::sycl::fmax(-20.0f, ::sycl::fmin(val_r, 20.0f));
                float sigm_r = 1.0f / (1.0f + ::sycl::exp(-clamped_r));

                *pdst = tanh_val * sigm_r;
            }
        );
    });

    q->wait();
    return true;
}

bool ggml_sycl_op_gated_tanh_sigmoid_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    int32_t* params = (int32_t*)node->op_params;
    int hidden_channels = params[0];
    return ggml_sycl_op_gated_tanh_sigmoid(backend, node->src[0], node, hidden_channels);
}

} // namespace sycl
} // namespace ggml_ops_ext
