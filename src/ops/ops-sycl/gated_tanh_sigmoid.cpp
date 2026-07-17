#include "ops/ops.h"
#include "ops_sycl.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "common.hpp" // From ggml-sycl
#include <iostream>

namespace ggml_ops_ext {
namespace sycl {

class GatedTanhSigmoidSYCLKernelF32;
class GatedTanhSigmoidSYCLKernelF16;

template <typename T, typename KernelName>
void launch_gated_tanh_sigmoid_sycl(
    ::sycl::queue* q, const T* x_d, T* dst_d,
    int hidden_channels, int64_t seq_len, int64_t batch,
    size_t nb_x0, size_t nb_x1, size_t nb_x2,
    size_t nb_dst0, size_t nb_dst1, size_t nb_dst2
) {
    int64_t total = hidden_channels * seq_len * batch;
    q->submit([&](::sycl::handler &cgh) {
        cgh.parallel_for<KernelName>(
            ::sycl::range<1>(total),
            [=](::sycl::id<1> id) {
                int64_t idx = id[0];
                int64_t i0 = idx % hidden_channels;
                int64_t tmp = idx / hidden_channels;
                int64_t i1 = tmp % seq_len;
                int64_t i2 = tmp / seq_len;

                // Left: channel i0
                const T* px_l = (const T*)((const char*)x_d + i2*nb_x2 + i1*nb_x1 + i0*nb_x0);
                // Right: channel i0 + hidden_channels
                const T* px_r = (const T*)((const char*)x_d + i2*nb_x2 + i1*nb_x1 + (i0 + hidden_channels)*nb_x0);

                T* pdst = (T*)((char*)dst_d + i2*nb_dst2 + i1*nb_dst1 + i0*nb_dst0);

                float val_l = (float)*px_l;
                float val_r = (float)*px_r;

                // Tanh: clamp val_l to [-10.f, 10.f] as it is multiplied by -2.f inside sycl::exp
                float clamped_l = ::sycl::fmax(-10.0f, ::sycl::fmin(val_l, 10.0f));
                float sigm_l = 1.0f / (1.0f + ::sycl::exp(-2.0f * clamped_l));
                float tanh_val = 2.0f * sigm_l - 1.0f;

                // Sigmoid: clamp val_r to [-20.f, 20.f]
                float clamped_r = ::sycl::fmax(-20.0f, ::sycl::fmin(val_r, 20.0f));
                float sigm_r = 1.0f / (1.0f + ::sycl::exp(-clamped_r));

                *pdst = (T)(tanh_val * sigm_r);
            }
        );
    });
}

bool ggml_sycl_op_gated_tanh_sigmoid(
    ggml_backend_t backend,
    struct ggml_tensor* x,
    struct ggml_tensor* dst,
    int hidden_channels
) {
    ::sycl::queue* q = (::sycl::queue*)ggml_ops_ext_bridge_sycl_get_queue(backend);
    if (!q) return false;

    GGML_ASSERT(x->type == dst->type);

    int64_t seq_len = dst->ne[1];
    int64_t batch = dst->ne[2];

    if (x->type == GGML_TYPE_F32) {
        launch_gated_tanh_sigmoid_sycl<float, GatedTanhSigmoidSYCLKernelF32>(
            q, (const float*)x->data, (float*)dst->data,
            hidden_channels, seq_len, batch,
            x->nb[0], x->nb[1], x->nb[2],
            dst->nb[0], dst->nb[1], dst->nb[2]
        );
    } else if (x->type == GGML_TYPE_F16) {
        launch_gated_tanh_sigmoid_sycl<::sycl::half, GatedTanhSigmoidSYCLKernelF16>(
            q, (const ::sycl::half*)x->data, (::sycl::half*)dst->data,
            hidden_channels, seq_len, batch,
            x->nb[0], x->nb[1], x->nb[2],
            dst->nb[0], dst->nb[1], dst->nb[2]
        );
    } else {
        std::cerr << "[ops-sycl] Gated Tanh Sigmoid error: unsupported data type: " << x->type << std::endl;
        return false;
    }
    return true;
}

bool ggml_sycl_op_gated_tanh_sigmoid_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    int32_t* params = (int32_t*)node->op_params;
    int hidden_channels = params[0];
    return ggml_sycl_op_gated_tanh_sigmoid(backend, node->src[0], node, hidden_channels);
}

} // namespace sycl
} // namespace ggml_ops_ext
