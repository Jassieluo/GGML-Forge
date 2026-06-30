#include "ops/ops.h"
#include "ops_cpu_common.h"
#include "ggml.h"
#include "matmul_f32.h"
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <cmath>
#include <immintrin.h>

namespace ggml_ops_ext {
namespace cpu {

template <typename Tq, typename Tk, typename Tv, typename Td, typename Tw>
static void compute_attention_impl(
    const Tq* q_d, const Tk* k_d, const Tv* v_d, const float* bias_d, Tw* w_d, Td* dst_d,
    int64_t head_dim, int64_t n_heads_q, int64_t seq_len_q, int64_t batch,
    int64_t n_heads_kv, int64_t seq_len_kv, int64_t group_size, float scale,
    size_t nb_q1, size_t nb_q2, size_t nb_q3,
    size_t nb_k1, size_t nb_k2, size_t nb_k3,
    size_t nb_v1, size_t nb_v2, size_t nb_v3,
    size_t nb_dst1, size_t nb_dst2, size_t nb_dst3,
    size_t nb_bias0, size_t nb_bias1, size_t nb_bias2, size_t nb_bias3,
    size_t nb_w0, size_t nb_w1, size_t nb_w2, size_t nb_w3
) {
    #pragma omp parallel for collapse(3)
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h_q = 0; h_q < n_heads_q; ++h_q) {
            for (int64_t iq = 0; iq < seq_len_q; ++iq) {
                int64_t h_kv = h_q / group_size;

                const Tq* vec_q = (const Tq*)((const char*)q_d + b * nb_q3 + iq * nb_q2 + h_q * nb_q1);
                Td* vec_dst = (Td*)((char*)dst_d + b * nb_dst3 + iq * nb_dst2 + h_q * nb_dst1);

                float scores_stack[1024];
                float* scores = scores_stack;
                std::vector<float> scores_heap;
                if (seq_len_kv > 1024) {
                    scores_heap.resize(seq_len_kv);
                    scores = scores_heap.data();
                }

                // 1. Compute dot products: Q @ K^T
                if constexpr (std::is_same_v<Tq, float> && std::is_same_v<Tk, float>) {
                    for (int64_t ik = 0; ik < seq_len_kv; ++ik) {
                        const float* vec_k = (const float*)((const char*)k_d + b * nb_k3 + ik * nb_k2 + h_kv * nb_k1);
                        scores[ik] = ops_vec_dot_f32((int)head_dim, (const float*)vec_q, vec_k) * scale;
                    }
                } else {
                    for (int64_t ik = 0; ik < seq_len_kv; ++ik) {
                        const Tk* vec_k = (const Tk*)((const char*)k_d + b * nb_k3 + ik * nb_k2 + h_kv * nb_k1);
                        float sum = 0.0f;
                        for (int64_t d = 0; d < head_dim; ++d) {
                            sum += read_val(&vec_q[d]) * read_val(&vec_k[d]);
                        }
                        scores[ik] = sum * scale;
                    }
                }

                // 2. Add optional bias
                if (bias_d) {
                    for (int64_t ik = 0; ik < seq_len_kv; ++ik) {
                        float b_val = *(const float*)((const char*)bias_d + b * nb_bias3 + h_q * nb_bias2 + iq * nb_bias1 + ik * nb_bias0);
                        scores[ik] += b_val;
                    }
                }

                // 3. Softmax
                float max_score = scores[0];
                for (int64_t ik = 1; ik < seq_len_kv; ++ik) {
                    if (scores[ik] > max_score) max_score = scores[ik];
                }

                float sum_exp = 0.0f;
                for (int64_t ik = 0; ik < seq_len_kv; ++ik) {
                    scores[ik] = std::exp(scores[ik] - max_score);
                    sum_exp += scores[ik];
                }

                float inv_sum = 1.0f / sum_exp;
                for (int64_t ik = 0; ik < seq_len_kv; ++ik) {
                    scores[ik] *= inv_sum;
                }

                // 4. Write back weights if requested
                if (w_d) {
                    for (int64_t ik = 0; ik < seq_len_kv; ++ik) {
                        Tw* w_ptr = (Tw*)((char*)w_d + b * nb_w3 + h_q * nb_w2 + iq * nb_w1 + ik * nb_w0);
                        write_val(w_ptr, scores[ik]);
                    }
                }

                // 5. Multiply by V (Weighted sum of values)
                if constexpr (std::is_same_v<Tv, float> && std::is_same_v<Td, float>) {
                    std::memset(vec_dst, 0, head_dim * sizeof(float));
                    for (int64_t ik = 0; ik < seq_len_kv; ++ik) {
                        float s = scores[ik];
                        const float* vec_v = (const float*)((const char*)v_d + b * nb_v3 + ik * nb_v2 + h_kv * nb_v1);

                        int64_t ic = 0;
#if defined(__AVX2__)
                        __m256 vs = _mm256_set1_ps(s);
                        for (; ic <= head_dim - 8; ic += 8) {
                            __m256 vd = _mm256_loadu_ps((float*)vec_dst + ic);
                            __m256 vv = _mm256_loadu_ps(vec_v + ic);
                            _mm256_storeu_ps((float*)vec_dst + ic, _mm256_fmadd_ps(vs, vv, vd));
                        }
#endif
                        for (; ic < head_dim; ++ic) {
                            ((float*)vec_dst)[ic] += s * vec_v[ic];
                        }
                    }
                } else {
                    for (int64_t d = 0; d < head_dim; ++d) {
                        write_val(&vec_dst[d], 0.0f);
                    }
                    for (int64_t ik = 0; ik < seq_len_kv; ++ik) {
                        float s = scores[ik];
                        const Tv* vec_v = (const Tv*)((const char*)v_d + b * nb_v3 + ik * nb_v2 + h_kv * nb_v1);
                        for (int64_t d = 0; d < head_dim; ++d) {
                            float cur = read_val(&vec_dst[d]);
                            write_val(&vec_dst[d], cur + s * read_val(&vec_v[d]));
                        }
                    }
                }
            }
        }
    }
}

bool ops_cpu_op_attention(ggml_backend_t backend, struct ggml_tensor* node) {
    (void)backend;

    ops_attention_params params;
    if (!ops_extract_attention_params(node, params)) return false;

    const struct ggml_tensor * q      = params.q;
    const struct ggml_tensor * k      = params.k;
    const struct ggml_tensor * v      = params.v;
    const struct ggml_tensor * bias   = params.bias;
    struct ggml_tensor *       attn_w = params.attn_w;
    struct ggml_tensor *       dst    = node;

    float scale = params.scale;

    const int64_t head_dim   = q->ne[0];
    const int64_t n_heads_q  = q->ne[1];
    const int64_t seq_len_q  = q->ne[2];
    const int64_t batch      = q->ne[3] > 0 ? q->ne[3] : 1;

    const int64_t n_heads_kv = k->ne[1];
    const int64_t seq_len_kv = k->ne[2];

    const int64_t group_size = n_heads_q / n_heads_kv;

    // Strides
    const size_t nb_q1 = q->nb[1];
    const size_t nb_q2 = q->nb[2];
    const size_t nb_q3 = q->nb[3];

    const size_t nb_k1 = k->nb[1];
    const size_t nb_k2 = k->nb[2];
    const size_t nb_k3 = k->nb[3];

    const size_t nb_v1 = v->nb[1];
    const size_t nb_v2 = v->nb[2];
    const size_t nb_v3 = v->nb[3];

    const size_t nb_dst1 = dst->nb[1];
    const size_t nb_dst2 = dst->nb[2];
    const size_t nb_dst3 = dst->nb[3];

    const size_t nb_bias0 = bias ? bias->nb[0] : 0;
    const size_t nb_bias1 = bias ? bias->nb[1] : 0;
    const size_t nb_bias2 = bias ? bias->nb[2] : 0;
    const size_t nb_bias3 = bias ? bias->nb[3] : 0;

    const size_t nb_w0 = attn_w ? attn_w->nb[0] : 0;
    const size_t nb_w1 = attn_w ? attn_w->nb[1] : 0;
    const size_t nb_w2 = attn_w ? attn_w->nb[2] : 0;
    const size_t nb_w3 = attn_w ? attn_w->nb[3] : 0;

    // Convert bias to F32 if it is F16
    std::vector<float> bias_f32;
    if (bias && bias->type == GGML_TYPE_F16) {
        int64_t bias_elems = ggml_nelements(bias);
        bias_f32.resize(bias_elems);
        const ggml_fp16_t* p = (const ggml_fp16_t*)bias->data;
        for (int64_t idx = 0; idx < bias_elems; ++idx) {
            bias_f32[idx] = ggml_fp16_to_fp32(p[idx]);
        }
    }
    const float* bias_ptr = bias ? (bias->type == GGML_TYPE_F32 ? (const float*)bias->data : bias_f32.data()) : nullptr;

    // Dispatch based on types
    #define DISPATCH_ATTN(Tq, Tk, Tv, Td, Tw) \
        compute_attention_impl<Tq, Tk, Tv, Td, Tw>( \
            (const Tq*)q->data, (const Tk*)k->data, (const Tv*)v->data, bias_ptr, (Tw*)(attn_w ? attn_w->data : nullptr), (Td*)dst->data, \
            head_dim, n_heads_q, seq_len_q, batch, n_heads_kv, seq_len_kv, group_size, scale, \
            nb_q1, nb_q2, nb_q3, nb_k1, nb_k2, nb_k3, nb_v1, nb_v2, nb_v3, nb_dst1, nb_dst2, nb_dst3, \
            nb_bias0, nb_bias1, nb_bias2, nb_bias3, nb_w0, nb_w1, nb_w2, nb_w3)

    if (q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32 && v->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        if (!attn_w || attn_w->type == GGML_TYPE_F32) {
            DISPATCH_ATTN(float, float, float, float, float);
        } else {
            DISPATCH_ATTN(float, float, float, float, ggml_fp16_t);
        }
    } else {
        // Fallback to F16 path
        if (!attn_w || attn_w->type == GGML_TYPE_F32) {
            DISPATCH_ATTN(ggml_fp16_t, ggml_fp16_t, ggml_fp16_t, ggml_fp16_t, float);
        } else {
            DISPATCH_ATTN(ggml_fp16_t, ggml_fp16_t, ggml_fp16_t, ggml_fp16_t, ggml_fp16_t);
        }
    }

    #undef DISPATCH_ATTN

    return true;
}

} // namespace cpu
} // namespace ggml_ops_ext
