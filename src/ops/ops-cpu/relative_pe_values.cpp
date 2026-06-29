#include "ops/ops.h"
#include <cmath>
#include <omp.h>
#include <algorithm>

namespace ggml_ops_ext {
namespace cpu {

bool ops_cpu_op_relative_pe_values(ggml_backend_t backend, struct ggml_tensor* node) {
    (void)backend;
    if ((int)node->op != GGML_OP_OPS_VIRT_RELATIVE_PE_VALUES) return false;

    ops_relative_pe_values_params params;
    if (!ops_extract_relative_pe_values_params(node, params)) return false;

    struct ggml_tensor* attn_w = params.attn_w;
    struct ggml_tensor* emb_rel_v = params.emb_rel_v;
    struct ggml_tensor* dst = node;

    int32_t W = params.window_size;

    const float* w_d = (const float*)attn_w->data;
    const float* r_d = (const float*)emb_rel_v->data;
    float* dst_d = (float*)dst->data;

    int64_t T = attn_w->ne[1];
    int64_t d_k = emb_rel_v->ne[0];
    int64_t n_head = attn_w->ne[2];
    int64_t r_len = 2 * W + 1;

    #pragma omp parallel for collapse(3)
    for (int64_t i = 0; i < T; ++i) {
        for (int64_t h = 0; h < n_head; ++h) {
            for (int64_t d = 0; d < d_k; ++d) {
                float sum = 0.0f;
                int64_t j_start = std::max((int64_t)0, i - W);
                int64_t j_end = std::min(T - 1, i + W);
                for (int64_t j = j_start; j <= j_end; ++j) {
                    int64_t r_idx = (j - i) + W;
                    float w_val = w_d[h * T * T + i * T + j];
                    float r_val = r_d[h * r_len * d_k + r_idx * d_k + d];
                    sum += w_val * r_val;
                }
                dst_d[i * n_head * d_k + h * d_k + d] = sum;
            }
        }
    }

    return true;
}

} // namespace cpu
} // namespace ggml_ops_ext
