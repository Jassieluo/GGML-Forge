#include "ops/ops.h"
#include "ops_cpu_common.h"
#include <cmath>
#include <omp.h>
#include <algorithm>
#include <vector>

namespace ggml_ops_ext {
namespace cpu {

template <typename Tw, typename Td>
static void compute_relative_values(
    const Tw* w_d, const float* r_d, Td* dst_d,
    int64_t T, int64_t d_k, int64_t n_head, int64_t r_len, int32_t W
) {
    #pragma omp parallel for collapse(3)
    for (int64_t i = 0; i < T; ++i) {
        for (int64_t h = 0; h < n_head; ++h) {
            for (int64_t d = 0; d < d_k; ++d) {
                float sum = 0.0f;
                int64_t j_start = std::max((int64_t)0, i - (int64_t)W);
                int64_t j_end = std::min(T - 1, i + (int64_t)W);
                for (int64_t j = j_start; j <= j_end; ++j) {
                    int64_t r_idx = (j - i) + W;
                    float w_val = read_val(&w_d[h * T * T + i * T + j]);
                    float r_val = r_d[h * r_len * d_k + r_idx * d_k + d];
                    sum += w_val * r_val;
                }
                write_val(&dst_d[i * n_head * d_k + h * d_k + d], sum);
            }
        }
    }
}

bool ops_cpu_op_relative_pe_values(ggml_backend_t backend, struct ggml_tensor* node) {
    (void)backend;
    if ((int)node->op != GGML_OP_OPS_VIRT_RELATIVE_PE_VALUES) return false;

    ops_relative_pe_values_params params;
    if (!ops_extract_relative_pe_values_params(node, params)) return false;

    struct ggml_tensor* attn_w = params.attn_w;
    struct ggml_tensor* emb_rel_v = params.emb_rel_v;
    struct ggml_tensor* dst = node;

    int32_t W = params.window_size;

    int64_t T = attn_w->ne[1];
    int64_t d_k = emb_rel_v->ne[0];
    int64_t n_head = attn_w->ne[2];
    int64_t r_len = 2 * W + 1;

    std::vector<float> r_f32;
    if (emb_rel_v->type == GGML_TYPE_F16) {
        int64_t r_elems = d_k * r_len * n_head;
        r_f32.resize(r_elems);
        const ggml_fp16_t* p = (const ggml_fp16_t*)emb_rel_v->data;
        for (int64_t idx = 0; idx < r_elems; ++idx) {
            r_f32[idx] = ggml_fp16_to_fp32(p[idx]);
        }
    }
    const float* r_ptr = emb_rel_v->type == GGML_TYPE_F32 ? (const float*)emb_rel_v->data : r_f32.data();

    if (attn_w->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        compute_relative_values((const float*)attn_w->data, r_ptr, (float*)dst->data, T, d_k, n_head, r_len, W);
    } else if (attn_w->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
        compute_relative_values((const ggml_fp16_t*)attn_w->data, r_ptr, (float*)dst->data, T, d_k, n_head, r_len, W);
    } else if (attn_w->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        compute_relative_values((const float*)attn_w->data, r_ptr, (ggml_fp16_t*)dst->data, T, d_k, n_head, r_len, W);
    } else if (attn_w->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        compute_relative_values((const ggml_fp16_t*)attn_w->data, r_ptr, (ggml_fp16_t*)dst->data, T, d_k, n_head, r_len, W);
    } else {
        return false;
    }

    return true;
}

} // namespace cpu
} // namespace ggml_ops_ext
