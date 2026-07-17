#include "ops/ops.h"
#include "ops/cpu.h"
#include "ops_cpu_common.h"
#include <cmath>
#include <omp.h>
#include <vector>

namespace ggml_ops_ext {
namespace cpu {

template <typename Tq, typename Td>
static void compute_relative_keys(
    const Tq* q_d, const float* r_d, Td* dst_d,
    int64_t d_k, int64_t T, int64_t n_head, int64_t r_len,
    float scale, int32_t W, int omp_threads
) {
    #pragma omp parallel for collapse(3) num_threads(omp_threads)
    for (int64_t h = 0; h < n_head; ++h) {
        for (int64_t i = 0; i < T; ++i) {
            for (int64_t j = 0; j < T; ++j) {
                int64_t k_idx = j - i;
                float val = 0.0f;
                if (k_idx >= -W && k_idx <= W) {
                    int64_t r_idx = k_idx + W;
                    const Tq* q_vec = q_d + h * T * d_k + i * d_k;
                    const float* r_vec = r_d + h * r_len * d_k + r_idx * d_k;
                    float sum = 0.0f;
                    for (int64_t d = 0; d < d_k; ++d) {
                        sum += read_val(&q_vec[d]) * r_vec[d];
                    }
                    val = sum * scale;
                }
                write_val(&dst_d[h * T * T + i * T + j], val);
            }
        }
    }
}

bool ops_cpu_op_relative_pe_keys(ggml_backend_t backend, struct ggml_tensor* node) {
    const int omp_threads = backend_thread_count(backend);
    if ((int)node->op != GGML_OP_OPS_VIRT_RELATIVE_PE_KEYS) return false;

    ops_relative_pe_keys_params params;
    if (!ops_extract_relative_pe_keys_params(node, params)) return false;

    struct ggml_tensor* q = params.q;
    struct ggml_tensor* emb_rel_k = params.emb_rel_k;
    struct ggml_tensor* dst = node;

    float scale = params.scale;
    int32_t W = params.window_size;

    int64_t d_k = q->ne[0];
    int64_t T = q->ne[1];
    int64_t n_head = q->ne[2];
    int64_t r_len = 2 * W + 1;

    std::vector<float> r_f32;
    if (emb_rel_k->type == GGML_TYPE_F16) {
        int64_t r_elems = d_k * r_len * n_head;
        r_f32.resize(r_elems);
        const ggml_fp16_t* p = (const ggml_fp16_t*)emb_rel_k->data;
        for (int64_t idx = 0; idx < r_elems; ++idx) {
            r_f32[idx] = ggml_fp16_to_fp32(p[idx]);
        }
    }
    const float* r_ptr = emb_rel_k->type == GGML_TYPE_F32 ? (const float*)emb_rel_k->data : r_f32.data();

    if (q->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        compute_relative_keys((const float*)q->data, r_ptr, (float*)dst->data, d_k, T, n_head, r_len, scale, W, omp_threads);
    } else if (q->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
        compute_relative_keys((const ggml_fp16_t*)q->data, r_ptr, (float*)dst->data, d_k, T, n_head, r_len, scale, W, omp_threads);
    } else if (q->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        compute_relative_keys((const float*)q->data, r_ptr, (ggml_fp16_t*)dst->data, d_k, T, n_head, r_len, scale, W, omp_threads);
    } else if (q->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        compute_relative_keys((const ggml_fp16_t*)q->data, r_ptr, (ggml_fp16_t*)dst->data, d_k, T, n_head, r_len, scale, W, omp_threads);
    } else {
        return false;
    }

    return true;
}

} // namespace cpu
} // namespace ggml_ops_ext
