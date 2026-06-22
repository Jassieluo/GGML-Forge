#include "ops/ops.h"
#include "ops_sycl.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "common.hpp" // From ggml-sycl
#include <iostream>

namespace ggml_ops_ext {
namespace sycl {

template <typename WeightT>
class ConvTranspose1DSYCLKernel;

template <typename WeightT>
void run_conv_transpose_1d_gpu(
    ::sycl::queue* q,
    const WeightT* w_d,
    const float* x_d,
    float* dst_d,
    int stride,
    int padding,
    int dilation,
    int64_t L_in, int64_t C_in,
    int64_t kW, int64_t C_out,
    int64_t L_out, int64_t batch,
    int64_t nb_w0, int64_t nb_w1, int64_t nb_w2,
    int64_t nb_x0, int64_t nb_x1, int64_t nb_x2,
    int64_t nb_dst0, int64_t nb_dst1, int64_t nb_dst2
) {
    int64_t output_size = L_out * C_out * batch;
    q->submit([&](::sycl::handler &cgh) {
        cgh.parallel_for<ConvTranspose1DSYCLKernel<WeightT>>(
            ::sycl::range<1>(output_size),
            [=](::sycl::id<1> id) {
                int64_t global_index = id[0];
                int64_t ow = global_index % L_out;
                int64_t tmp = global_index / L_out;
                int64_t c_out = tmp % C_out;
                int64_t b = tmp / C_out;

                float accumulator = 0.0f;

                for (int64_t c_in = 0; c_in < C_in; ++c_in) {
                    for (int64_t kw = 0; kw < kW; ++kw) {
                        int64_t iw_stride = ow + padding - kw * dilation;
                        if (iw_stride >= 0 && iw_stride % stride == 0) {
                            int64_t iw = iw_stride / stride;
                            if (iw < L_in) {
                                const WeightT* ptr_w = (const WeightT*)((const char*)w_d + c_in * nb_w2 + c_out * nb_w1 + kw * nb_w0);
                                const float* ptr_x = (const float*)((const char*)x_d + b * nb_x2 + c_in * nb_x1 + iw * nb_x0);
                                accumulator += (float)(*ptr_w) * (*ptr_x);
                            }
                        }
                    }
                }

                float* ptr_dst = (float*)((char*)dst_d + b * nb_dst2 + c_out * nb_dst1 + ow * nb_dst0);
                *ptr_dst = accumulator;
            }
        );
    });
}

bool ggml_sycl_op_conv_transpose_1d(
    ggml_backend_t backend,
    struct ggml_tensor* w,
    struct ggml_tensor* x,
    struct ggml_tensor* dst,
    int stride,
    int padding,
    int dilation
) {
    ggml_backend_sycl_context* sycl_ctx = (ggml_backend_sycl_context*)backend->context;
    if (!sycl_ctx) return false;
    ::sycl::queue* q = sycl_ctx->stream();
    if (!q) return false;

    int64_t L_in = x->ne[0];
    int64_t C_in = x->ne[1];
    int64_t batch = x->ne[2];

    int64_t kW = w->ne[0];
    int64_t C_out = w->ne[1];

    int64_t L_out = dst->ne[0];

    const float* x_d = (const float*)x->data;
    float* dst_d = (float*)dst->data;

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    if (w->type == GGML_TYPE_F32) {
        const float* w_d = (const float*)w->data;
        run_conv_transpose_1d_gpu<float>(
            q, w_d, x_d, dst_d,
            stride, padding, dilation,
            L_in, C_in, kW, C_out, L_out, batch,
            w->nb[0], w->nb[1], w->nb[2],
            x->nb[0], x->nb[1], x->nb[2],
            dst->nb[0], dst->nb[1], dst->nb[2]
        );
    } else if (w->type == GGML_TYPE_F16) {
        const ::sycl::half* w_d = (const ::sycl::half*)w->data;
        run_conv_transpose_1d_gpu<::sycl::half>(
            q, w_d, x_d, dst_d,
            stride, padding, dilation,
            L_in, C_in, kW, C_out, L_out, batch,
            w->nb[0], w->nb[1], w->nb[2],
            x->nb[0], x->nb[1], x->nb[2],
            dst->nb[0], dst->nb[1], dst->nb[2]
        );
    } else {
        std::cerr << "[ops-sycl] compute_conv_transpose_1d error: unsupported weight type: " << w->type << std::endl;
        return false;
    }

    q->wait();
    return true;
}

bool ggml_sycl_op_conv_transpose_1d_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    ops_conv_transpose_1d_params params;
    if (!ops_extract_conv_transpose_1d_params(node, params)) {
        return false;
    }
    return ggml_sycl_op_conv_transpose_1d(backend, params.w, params.x, node, params.stride, params.padding, params.dilation);
}

} // namespace sycl
} // namespace ggml_ops_ext
