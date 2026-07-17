#include "ops/ops.h"
#include "ops_sycl.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "common.hpp"

namespace ggml_ops_ext {
namespace sycl {

template <typename TW, typename TR>
class RelativePeValuesSYCLKernel;

template <typename TW, typename TR>
static void launch_relative_pe_values(
    ::sycl::queue* queue, const TW* w, const TR* r, TW* dst,
    int64_t d_k, int64_t T, int64_t n_head, int32_t W
) {
    queue->submit([&](::sycl::handler &cgh) {
        cgh.parallel_for<RelativePeValuesSYCLKernel<TW, TR>>(
            ::sycl::range<3>(T, n_head, d_k),
            [=](::sycl::id<3> id) {
                int64_t i = id[0];
                int64_t h = id[1];
                int64_t d = id[2];
                int64_t r_len = 2 * W + 1;
                float sum = 0.0f;
                int64_t j_start = i - W;
                if (j_start < 0) j_start = 0;
                int64_t j_end = i + W;
                if (j_end >= T) j_end = T - 1;
                for (int64_t j = j_start; j <= j_end; ++j) {
                    int64_t r_idx = (j - i) + W;
                    sum += (float)w[h * T * T + i * T + j] *
                           (float)r[h * r_len * d_k + r_idx * d_k + d];
                }
                dst[i * n_head * d_k + h * d_k + d] = (TW)sum;
            }
        );
    });
}

bool ggml_sycl_op_relative_pe_values(
    ggml_backend_t backend,
    struct ggml_tensor* node
) {
    ops_relative_pe_values_params params;
    if (!ops_extract_relative_pe_values_params(node, params)) return false;

    ::sycl::queue* q_queue = (::sycl::queue*)ggml_ops_ext_bridge_sycl_get_queue(backend);
    if (!q_queue) return false;

    struct ggml_tensor* attn_w = params.attn_w;
    struct ggml_tensor* emb_rel_v = params.emb_rel_v;
    struct ggml_tensor* dst = node;

    int32_t W = params.window_size;

    int64_t T = attn_w->ne[1];
    int64_t d_k = emb_rel_v->ne[0];
    int64_t n_head = attn_w->ne[2];

    if (attn_w->type == GGML_TYPE_F32) {
        const float* w_d = (const float*)attn_w->data;
        const float* r_d = (const float*)emb_rel_v->data;
        float* dst_d = (float*)dst->data;

        if (emb_rel_v->type == GGML_TYPE_F32) launch_relative_pe_values(q_queue, w_d, r_d, dst_d, d_k, T, n_head, W);
        else launch_relative_pe_values(q_queue, w_d, (const ::sycl::half*)emb_rel_v->data, dst_d, d_k, T, n_head, W);
    } else if (attn_w->type == GGML_TYPE_F16) {
        if (emb_rel_v->type == GGML_TYPE_F32) launch_relative_pe_values(q_queue, (const ::sycl::half*)attn_w->data, (const float*)emb_rel_v->data, (::sycl::half*)dst->data, d_k, T, n_head, W);
        else launch_relative_pe_values(q_queue, (const ::sycl::half*)attn_w->data, (const ::sycl::half*)emb_rel_v->data, (::sycl::half*)dst->data, d_k, T, n_head, W);
    } else {
        return false;
    }
    return true;
}

bool ggml_sycl_op_relative_pe_values_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    return ggml_sycl_op_relative_pe_values(backend, node);
}

} // namespace sycl
} // namespace ggml_ops_ext
