#include "ops/ops.h"
#include "ops_sycl.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "common.hpp"

namespace ggml_ops_ext {
namespace sycl {

class RelativePeKeysSYCLKernelF32;

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

        q_queue->submit([&](::sycl::handler &cgh) {
            cgh.parallel_for<RelativePeKeysSYCLKernelF32>(
                ::sycl::range<3>(n_head, T, T),
                [=](::sycl::id<3> id) {
                    int64_t h = id[0];
                    int64_t i = id[1]; // query
                    int64_t j = id[2]; // key

                    int64_t k_idx = j - i;
                    float val = 0.0f;
                    if (k_idx >= -W && k_idx <= W) {
                        int64_t r_idx = k_idx + W;
                        int64_t r_len = 2 * W + 1;
                        const float* q_vec = q_d + h * T * d_k + i * d_k;
                        const float* r_vec = r_d + h * r_len * d_k + r_idx * d_k;
                        float sum = 0.0f;
                        for (int64_t d = 0; d < d_k; ++d) {
                            sum += q_vec[d] * r_vec[d];
                        }
                        val = sum * scale;
                    }
                    dst_d[h * T * T + i * T + j] = val;
                }
            );
        });
    } else {
        return false;
    }
    q_queue->wait();
    return true;
}

bool ggml_sycl_op_relative_pe_keys_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    return ggml_sycl_op_relative_pe_keys(backend, node);
}

} // namespace sycl
} // namespace ggml_ops_ext
