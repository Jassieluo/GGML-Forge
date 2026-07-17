#include "ops/ops.h"
#include "ops_sycl.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "common.hpp"

namespace ggml_ops_ext {
namespace sycl {

template <typename TQ, typename TR>
class RelativePeKeysSYCLKernel;

template <typename TQ, typename TR>
static void launch_relative_pe_keys(
    ::sycl::queue* queue, const TQ* q, const TR* r, TQ* dst,
    int64_t d_k, int64_t T, int64_t n_head, int32_t W, float scale
) {
    queue->submit([&](::sycl::handler &cgh) {
        cgh.parallel_for<RelativePeKeysSYCLKernel<TQ, TR>>(
            ::sycl::range<3>(n_head, T, T),
            [=](::sycl::id<3> id) {
                int64_t h = id[0];
                int64_t i = id[1];
                int64_t j = id[2];
                int64_t k_idx = j - i;
                float val = 0.0f;
                if (k_idx >= -W && k_idx <= W) {
                    int64_t r_idx = k_idx + W;
                    int64_t r_len = 2 * W + 1;
                    const TQ* q_vec = q + h * T * d_k + i * d_k;
                    const TR* r_vec = r + h * r_len * d_k + r_idx * d_k;
                    float sum = 0.0f;
                    for (int64_t d = 0; d < d_k; ++d) sum += (float)q_vec[d] * (float)r_vec[d];
                    val = sum * scale;
                }
                dst[h * T * T + i * T + j] = (TQ)val;
            }
        );
    });
}

bool ggml_sycl_op_relative_pe_keys(
    ggml_backend_t backend,
    struct ggml_tensor* node
) {
    ops_relative_pe_keys_params params;
    if (!ops_extract_relative_pe_keys_params(node, params)) return false;

    ::sycl::queue* q_queue = (::sycl::queue*)ggml_ops_ext_bridge_sycl_get_queue(backend);
    if (!q_queue) return false;

    struct ggml_tensor* q = params.q;
    struct ggml_tensor* emb_rel_k = params.emb_rel_k;
    struct ggml_tensor* dst = node;

    float scale = params.scale;
    int32_t W = params.window_size;

    int64_t d_k = q->ne[0];
    int64_t T = q->ne[1];
    int64_t n_head = q->ne[2];

    if (q->type == GGML_TYPE_F32) {
        const float* q_d = (const float*)q->data;
        const float* r_d = (const float*)emb_rel_k->data;
        float* dst_d = (float*)dst->data;

        if (emb_rel_k->type == GGML_TYPE_F32) launch_relative_pe_keys(q_queue, q_d, r_d, dst_d, d_k, T, n_head, W, scale);
        else launch_relative_pe_keys(q_queue, q_d, (const ::sycl::half*)emb_rel_k->data, dst_d, d_k, T, n_head, W, scale);
    } else if (q->type == GGML_TYPE_F16) {
        if (emb_rel_k->type == GGML_TYPE_F32) launch_relative_pe_keys(q_queue, (const ::sycl::half*)q->data, (const float*)emb_rel_k->data, (::sycl::half*)dst->data, d_k, T, n_head, W, scale);
        else launch_relative_pe_keys(q_queue, (const ::sycl::half*)q->data, (const ::sycl::half*)emb_rel_k->data, (::sycl::half*)dst->data, d_k, T, n_head, W, scale);
    } else {
        return false;
    }
    return true;
}

bool ggml_sycl_op_relative_pe_keys_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    return ggml_sycl_op_relative_pe_keys(backend, node);
}

} // namespace sycl
} // namespace ggml_ops_ext
