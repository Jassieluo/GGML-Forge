#include "ops/ops.h"
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

    const float * q_d = (const float *)q->data;
    const float * k_d = (const float *)k->data;
    const float * v_d = (const float *)v->data;
    float *       dst_d = (float *)dst->data;

    // Strides
    const size_t nb_q0 = q->nb[0];
    const size_t nb_q1 = q->nb[1];
    const size_t nb_q2 = q->nb[2];
    const size_t nb_q3 = q->nb[3];

    const size_t nb_k0 = k->nb[0];
    const size_t nb_k1 = k->nb[1];
    const size_t nb_k2 = k->nb[2];
    const size_t nb_k3 = k->nb[3];

    const size_t nb_v0 = v->nb[0];
    const size_t nb_v1 = v->nb[1];
    const size_t nb_v2 = v->nb[2];
    const size_t nb_v3 = v->nb[3];

    const size_t nb_dst0 = dst->nb[0];
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

    #pragma omp parallel for collapse(3)
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h_q = 0; h_q < n_heads_q; ++h_q) {
            for (int64_t iq = 0; iq < seq_len_q; ++iq) {
                int64_t h_kv = h_q / group_size;

                const float * vec_q = (const float *)((const char *)q_d + b * nb_q3 + iq * nb_q2 + h_q * nb_q1);
                float * vec_dst = (float *)((char *)dst_d + b * nb_dst3 + iq * nb_dst2 + h_q * nb_dst1);

                std::vector<float> scores(seq_len_kv);

                // 1. Compute dot products: Q @ K^T
                for (int64_t ik = 0; ik < seq_len_kv; ++ik) {
                    const float * vec_k = (const float *)((const char *)k_d + b * nb_k3 + ik * nb_k2 + h_kv * nb_k1);
                    scores[ik] = ops_vec_dot_f32((int)head_dim, vec_q, vec_k) * scale;
                }

                // 2. Add optional bias
                if (bias) {
                    for (int64_t ik = 0; ik < seq_len_kv; ++ik) {
                        float b_val = *(const float *)((const char *)bias->data + b * nb_bias3 + h_q * nb_bias2 + iq * nb_bias1 + ik * nb_bias0);
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
                if (attn_w) {
                    for (int64_t ik = 0; ik < seq_len_kv; ++ik) {
                        float * w_ptr = (float *)((char *)attn_w->data + b * nb_w3 + h_q * nb_w2 + iq * nb_w1 + ik * nb_w0);
                        *w_ptr = scores[ik];
                    }
                }

                // 5. Multiply by V (Weighted sum of values)
                std::memset(vec_dst, 0, head_dim * sizeof(float));
                for (int64_t ik = 0; ik < seq_len_kv; ++ik) {
                    float s = scores[ik];
                    const float * vec_v = (const float *)((const char *)v_d + b * nb_v3 + ik * nb_v2 + h_kv * nb_v1);

                    int64_t ic = 0;
#if defined(__AVX2__)
                    __m256 vs = _mm256_set1_ps(s);
                    for (; ic <= head_dim - 8; ic += 8) {
                        __m256 vd = _mm256_loadu_ps(vec_dst + ic);
                        __m256 vv = _mm256_loadu_ps(vec_v + ic);
                        _mm256_storeu_ps(vec_dst + ic, _mm256_fmadd_ps(vs, vv, vd));
                    }
#endif
                    for (; ic < head_dim; ++ic) {
                        vec_dst[ic] += s * vec_v[ic];
                    }
                }
            }
        }
    }

    return true;
}

} // namespace cpu
} // namespace ggml_ops_ext
