#include "ops/ops.h"
#include <cmath>
#include <omp.h>

namespace ggml_ops_ext {
namespace cpu {

bool ops_cpu_op_relative_pe_keys(ggml_backend_t backend, struct ggml_tensor* node) {
    (void)backend;
    if ((int)node->op != GGML_OP_OPS_VIRT_RELATIVE_PE_KEYS) return false;

    ops_relative_pe_keys_params params;
    if (!ops_extract_relative_pe_keys_params(node, params)) return false;

    struct ggml_tensor* q = params.q;
    struct ggml_tensor* emb_rel_k = params.emb_rel_k;
    struct ggml_tensor* dst = node;

    float scale = params.scale;
    int32_t W = params.window_size;

    const float* q_d = (const float*)q->data;
    const float* r_d = (const float*)emb_rel_k->data;
    float* dst_d = (float*)dst->data;

    int64_t d_k = q->ne[0];
    int64_t T = q->ne[1];
    int64_t n_head = q->ne[2];
    int64_t r_len = 2 * W + 1;

    #pragma omp parallel for collapse(3)
    for (int64_t h = 0; h < n_head; ++h) {
        for (int64_t i = 0; i < T; ++i) {
            for (int64_t j = 0; j < T; ++j) {
                int64_t k_idx = j - i;
                float val = 0.0f;
                if (k_idx >= -W && k_idx <= W) {
                    int64_t r_idx = k_idx + W;
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
        }
    }

    return true;
}

} // namespace cpu
} // namespace ggml_ops_ext
