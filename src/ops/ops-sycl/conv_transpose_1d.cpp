#include "ops/ops.h"
#include "ops_sycl.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "common.hpp" // From ggml-sycl
#include <iostream>
#include <algorithm>

namespace ggml_ops_ext {
namespace sycl {

template <typename T_w, typename T_x, typename T_dst>
class DirectConvTranspose1DSYCLKernel;

template <typename T_w, typename T_x, typename T_dst>
static void direct_conv_transpose_1d_sycl_template(
    ::sycl::queue* q,
    const void* w,
    const void* x,
    const void* bias,
    int bias_type,
    void* dst,
    int64_t C_in, int64_t L_in, int64_t L_out, int64_t kW, int64_t C_out, int64_t batch,
    int stride, int padding, int dilation,
    size_t nb_w0, size_t nb_w1, size_t nb_w2,
    size_t nb_x0, size_t nb_x1, size_t nb_x2,
    size_t nb_dst0, size_t nb_dst1, size_t nb_dst2
) {
    int64_t total = batch * C_out * L_out;
    q->submit([&](::sycl::handler &cgh) {
        cgh.parallel_for<DirectConvTranspose1DSYCLKernel<T_w, T_x, T_dst>>(::sycl::range<1>(total), [=](::sycl::id<1> id) {
            int64_t idx = id[0];
            int64_t ow = idx % L_out;
            int64_t tmp = idx / L_out;
            int64_t c = tmp % C_out;
            int64_t n = tmp / C_out;

            float sum = 0.0f;
            for (int64_t ik = 0; ik < kW; ++ik) {
                int64_t iw_stride = ow + padding - ik * dilation;
                if (iw_stride % stride == 0) {
                    int64_t iw = iw_stride / stride;
                    if (iw >= 0 && iw < L_in) {
                        for (int64_t ic = 0; ic < C_in; ++ic) {
                            T_w val_w_typed = *(const T_w*)((const char*)w + ik * nb_w0 + c * nb_w1 + ic * nb_w2);
                            T_x val_x_typed = *(const T_x*)((const char*)x + iw * nb_x0 + ic * nb_x1 + n * nb_x2);
                            sum += (float)val_w_typed * (float)val_x_typed;
                        }
                    }
                }
            }
            float b_val = 0.0f;
            if (bias) {
                if (bias_type == 0) {
                    b_val = ((const float*)bias)[c];
                } else {
                    b_val = (float)((const ::sycl::half*)bias)[c];
                }
            }
            T_dst* pdst = (T_dst*)((char*)dst + ow * nb_dst0 + c * nb_dst1 + n * nb_dst2);
            *pdst = (T_dst)(sum + b_val);
        });
    });
}

bool ggml_sycl_op_conv_transpose_1d(
    ggml_backend_t backend,
    struct ggml_tensor* w,
    struct ggml_tensor* x,
    struct ggml_tensor* bias,
    struct ggml_tensor* dst,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    ::sycl::queue* q = (::sycl::queue*)ggml_ops_ext_bridge_sycl_get_queue(backend);
    if (!q) return false;

    int64_t L_in = x->ne[0];
    int64_t C_in = x->ne[1];
    int64_t batch = (x->ne[2] > 0) ? x->ne[2] : 1;

    int64_t kW = w->ne[0];
    int64_t C_out = w->ne[1];
    int64_t L_out = dst->ne[0];

    const void* bias_data = bias ? bias->data : nullptr;
    int bias_type = bias ? ((bias->type == GGML_TYPE_F32) ? 0 : 1) : 0;

    int64_t C_in_group = C_in / groups;
    int64_t C_out_group = C_out;

    size_t bias_elem_size = bias ? ((bias->type == GGML_TYPE_F32) ? sizeof(float) : sizeof(::sycl::half)) : sizeof(float);

    for (int g = 0; g < groups; ++g) {
        const void* w_g = (const char*)w->data + g * C_in_group * w->nb[2];
        const void* x_g = (const char*)x->data + g * C_in_group * x->nb[1];
        void* dst_g = (char*)dst->data + g * C_out_group * dst->nb[1];
        const void* bias_g = bias_data ? (const char*)bias_data + g * C_out_group * bias_elem_size : nullptr;

        if (w->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            direct_conv_transpose_1d_sycl_template<float, float, float>(
                q, w_g, x_g, bias_g, bias_type, dst_g,
                C_in_group, L_in, L_out, kW, C_out_group, batch,
                stride, padding, dilation,
                w->nb[0], w->nb[1], w->nb[2],
                x->nb[0], x->nb[1], x->nb[2],
                dst->nb[0], dst->nb[1], dst->nb[2]
            );
        } else if (w->type == GGML_TYPE_F16 && x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            direct_conv_transpose_1d_sycl_template<::sycl::half, float, float>(
                q, w_g, x_g, bias_g, bias_type, dst_g,
                C_in_group, L_in, L_out, kW, C_out_group, batch,
                stride, padding, dilation,
                w->nb[0], w->nb[1], w->nb[2],
                x->nb[0], x->nb[1], x->nb[2],
                dst->nb[0], dst->nb[1], dst->nb[2]
            );
        } else if (w->type == GGML_TYPE_F16 && x->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
            direct_conv_transpose_1d_sycl_template<::sycl::half, ::sycl::half, ::sycl::half>(
                q, w_g, x_g, bias_g, bias_type, dst_g,
                C_in_group, L_in, L_out, kW, C_out_group, batch,
                stride, padding, dilation,
                w->nb[0], w->nb[1], w->nb[2],
                x->nb[0], x->nb[1], x->nb[2],
                dst->nb[0], dst->nb[1], dst->nb[2]
            );
        } else {
            std::cerr << "[ops-sycl] Conv Transpose 1D error: unsupported type combination" << std::endl;
            return false;
        }
    }
    q->wait();
    return true;
}

bool ggml_sycl_op_conv_transpose_1d_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    ops_conv_transpose_1d_params params;
    if (!ops_extract_conv_transpose_1d_params(node, params)) {
        return false;
    }
    return ggml_sycl_op_conv_transpose_1d(backend, params.w, params.x, params.bias, node, params.stride, params.padding, params.dilation, params.groups);
}

} // namespace sycl
} // namespace ggml_ops_ext
