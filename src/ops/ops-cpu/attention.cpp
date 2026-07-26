#include "ops/ops.h"
#include "ops/cpu.h"
#include "ops_cpu_common.h"
#define GGML_COMMON_DECL_CPP
#include "ggml.h"
#include "ggml-common.h"
#include "matmul_f32.h"
#include "quantized_block_avx2.h"
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <immintrin.h>

namespace ggml_ops_ext {
namespace cpu {

static inline float attention_row_value(ggml_type type, const void* row, int64_t index) {
    if (type == GGML_TYPE_F32) return static_cast<const float*>(row)[index];
    if (type == GGML_TYPE_F16) return ggml_fp16_to_fp32(static_cast<const ggml_fp16_t*>(row)[index]);
    if (type == GGML_TYPE_Q8_0) {
        const block_q8_0& block = static_cast<const block_q8_0*>(row)[index / QK8_0];
        return ggml_fp16_to_fp32(block.d) * block.qs[index % QK8_0];
    }
    const block_q4_0& block = static_cast<const block_q4_0*>(row)[index / QK4_0];
    const int within = static_cast<int>(index % QK4_0);
    const uint8_t packed = block.qs[within % (QK4_0 / 2)];
    const int value = ((within < QK4_0 / 2 ? packed : packed >> 4) & 0x0f) - 8;
    return ggml_fp16_to_fp32(block.d) * value;
}

static void compute_attention_streaming(
    const ggml_tensor* q, const ggml_tensor* k, const ggml_tensor* v,
    const ggml_tensor* bias, ggml_tensor* dst, float scale, int omp_threads
) {
    const int64_t head_dim = q->ne[0];
    const int64_t seq_len_q = q->ne[1];
    const int64_t n_heads_q = q->ne[2];
    const int64_t batch = q->ne[3];
    const int64_t seq_len_kv = k->ne[1];
    const int64_t n_heads_kv = k->ne[2];
    const int64_t group_size = n_heads_q / n_heads_kv;

    #pragma omp parallel for collapse(3) num_threads(omp_threads)
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h_q = 0; h_q < n_heads_q; ++h_q) {
            for (int64_t iq = 0; iq < seq_len_q; ++iq) {
                const int64_t h_kv = h_q / group_size;
                const char* q_row = static_cast<const char*>(q->data) + b * q->nb[3] + h_q * q->nb[2] + iq * q->nb[1];
                char* dst_row = static_cast<char*>(dst->data) + b * dst->nb[3] + h_q * dst->nb[2] + iq * dst->nb[1];
                float accumulator_stack[256];
                std::vector<float> accumulator_heap;
                float* accumulator = accumulator_stack;
                if (head_dim > 256) {
                    accumulator_heap.resize(static_cast<size_t>(head_dim));
                    accumulator = accumulator_heap.data();
                }
                std::fill_n(accumulator, head_dim, 0.0f);
                float running_max = -std::numeric_limits<float>::infinity();
                float running_sum = 0.0f;

                for (int64_t ik = 0; ik < seq_len_kv; ++ik) {
                    const char* k_row = static_cast<const char*>(k->data) + b * k->nb[3] + h_kv * k->nb[2] + ik * k->nb[1];
                    float dot = 0.0f;
                    if (q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32) {
                        dot = ops_vec_dot_f32(static_cast<int>(head_dim), reinterpret_cast<const float*>(q_row),
                                              reinterpret_cast<const float*>(k_row));
                    } else if (q->type == GGML_TYPE_F32 &&
                               (k->type == GGML_TYPE_Q4_0 || k->type == GGML_TYPE_Q8_0)) {
                        dot = dot_quantized_row_avx2(k->type, k_row, reinterpret_cast<const float*>(q_row), head_dim);
                    } else {
                        for (int64_t d = 0; d < head_dim; ++d) {
                            dot += attention_row_value(q->type, q_row, d) * attention_row_value(k->type, k_row, d);
                        }
                    }

                    float score = dot * scale;
                    if (bias) {
                        const char* bias_value = static_cast<const char*>(bias->data) + b * bias->nb[3] +
                            h_q * bias->nb[2] + iq * bias->nb[1] + ik * bias->nb[0];
                        score += attention_row_value(bias->type, bias_value, 0);
                    }
                    if (std::isinf(score) && score < 0.0f) continue;

                    const float next_max = std::max(running_max, score);
                    const float previous_scale = running_sum == 0.0f ? 0.0f : std::exp(running_max - next_max);
                    const float current_scale = std::exp(score - next_max);
                    running_sum = running_sum * previous_scale + current_scale;
                    const char* v_row = static_cast<const char*>(v->data) + b * v->nb[3] + h_kv * v->nb[2] + ik * v->nb[1];

                    int64_t d = 0;
#if defined(__AVX2__)
                    const __m256 previous = _mm256_set1_ps(previous_scale);
                    for (; d <= head_dim - 8; d += 8) {
                        _mm256_storeu_ps(accumulator + d, _mm256_mul_ps(_mm256_loadu_ps(accumulator + d), previous));
                    }
#endif
                    for (; d < head_dim; ++d) accumulator[d] *= previous_scale;
                    if (v->type == GGML_TYPE_Q4_0 || v->type == GGML_TYPE_Q8_0) {
                        const int64_t block_size = ggml_blck_size(v->type);
                        const size_t type_size = ggml_type_size(v->type);
                        for (int64_t offset = 0; offset < head_dim; offset += block_size) {
                            quantized_32_microtile tile;
                            decode_quantized_32_avx2(v->type, v_row + (offset / block_size) * type_size, 0, tile);
                            axpy_quantized_32_microtile_avx2(v->type, tile, current_scale, accumulator + offset);
                        }
                    } else if (v->type == GGML_TYPE_F32) {
                        int64_t i = 0;
#if defined(__AVX2__)
                        const __m256 current = _mm256_set1_ps(current_scale);
                        const float* values = reinterpret_cast<const float*>(v_row);
                        for (; i <= head_dim - 8; i += 8) {
                            _mm256_storeu_ps(accumulator + i, _mm256_fmadd_ps(
                                current, _mm256_loadu_ps(values + i), _mm256_loadu_ps(accumulator + i)));
                        }
#endif
                        for (; i < head_dim; ++i) {
                            accumulator[i] += current_scale * reinterpret_cast<const float*>(v_row)[i];
                        }
                    } else {
                        for (int64_t i = 0; i < head_dim; ++i) {
                            accumulator[i] += current_scale * attention_row_value(v->type, v_row, i);
                        }
                    }
                    running_max = next_max;
                }

                const float inv_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
                for (int64_t d = 0; d < head_dim; ++d) {
                    if (dst->type == GGML_TYPE_F32) reinterpret_cast<float*>(dst_row)[d] = accumulator[d] * inv_sum;
                    else reinterpret_cast<ggml_fp16_t*>(dst_row)[d] = ggml_fp32_to_fp16(accumulator[d] * inv_sum);
                }
            }
        }
    }
}

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
    size_t nb_w0, size_t nb_w1, size_t nb_w2, size_t nb_w3,
    int omp_threads
) {
    #pragma omp parallel for collapse(3) num_threads(omp_threads)
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h_q = 0; h_q < n_heads_q; ++h_q) {
            for (int64_t iq = 0; iq < seq_len_q; ++iq) {
                int64_t h_kv = h_q / group_size;

                const Tq* vec_q = (const Tq*)((const char*)q_d + b * nb_q3 + h_q * nb_q2 + iq * nb_q1);
                Td* vec_dst = (Td*)((char*)dst_d + b * nb_dst3 + h_q * nb_dst2 + iq * nb_dst1);

                if (!w_d) {
                    float accumulator_stack[256];
                    float* accumulator = accumulator_stack;
                    std::vector<float> accumulator_heap;
                    if (head_dim > 256) {
                        accumulator_heap.resize(static_cast<size_t>(head_dim));
                        accumulator = accumulator_heap.data();
                    }
                    std::fill_n(accumulator, head_dim, 0.0f);

                    float running_max = -std::numeric_limits<float>::infinity();
                    float running_sum = 0.0f;
                    for (int64_t ik = 0; ik < seq_len_kv; ++ik) {
                        const Tk* vec_k = (const Tk*)((const char*)k_d + b * nb_k3 + h_kv * nb_k2 + ik * nb_k1);
                        float score = 0.0f;
                        if constexpr (std::is_same_v<Tq, float> && std::is_same_v<Tk, float>) {
                            score = ops_vec_dot_f32((int)head_dim, (const float*)vec_q, (const float*)vec_k);
                        } else {
                            for (int64_t d = 0; d < head_dim; ++d) score += read_val(&vec_q[d]) * read_val(&vec_k[d]);
                        }
                        score *= scale;
                        if (bias_d) {
                            score += *(const float*)((const char*)bias_d + b * nb_bias3 + h_q * nb_bias2 +
                                                      iq * nb_bias1 + ik * nb_bias0);
                        }
                        if (std::isinf(score) && score < 0.0f) continue;

                        const float next_max = std::max(running_max, score);
                        const float previous_scale = running_sum == 0.0f ? 0.0f : std::exp(running_max - next_max);
                        const float current_scale = std::exp(score - next_max);
                        running_sum = running_sum * previous_scale + current_scale;

                        const Tv* vec_v = (const Tv*)((const char*)v_d + b * nb_v3 + h_kv * nb_v2 + ik * nb_v1);
                        for (int64_t d = 0; d < head_dim; ++d) {
                            accumulator[d] = accumulator[d] * previous_scale + current_scale * read_val(&vec_v[d]);
                        }
                        running_max = next_max;
                    }

                    const float inv_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
                    for (int64_t d = 0; d < head_dim; ++d) write_val(&vec_dst[d], accumulator[d] * inv_sum);
                    continue;
                }

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
                        const float* vec_k = (const float*)((const char*)k_d + b * nb_k3 + h_kv * nb_k2 + ik * nb_k1);
                        scores[ik] = ops_vec_dot_f32((int)head_dim, (const float*)vec_q, vec_k) * scale;
                    }
                } else {
                    for (int64_t ik = 0; ik < seq_len_kv; ++ik) {
                        const Tk* vec_k = (const Tk*)((const char*)k_d + b * nb_k3 + h_kv * nb_k2 + ik * nb_k1);
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
                if (std::isinf(max_score) && max_score < 0.0f) {
                    // Fully masked row: exp(-inf - -inf) would be NaN. Emit zeros.
                    std::fill_n(scores, seq_len_kv, 0.0f);
                } else {
                    for (int64_t ik = 0; ik < seq_len_kv; ++ik) {
                        scores[ik] = std::exp(scores[ik] - max_score);
                        sum_exp += scores[ik];
                    }
                }

                const float inv_sum = sum_exp > 0.0f ? 1.0f / sum_exp : 0.0f;
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
                        const float* vec_v = (const float*)((const char*)v_d + b * nb_v3 + h_kv * nb_v2 + ik * nb_v1);

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
                        const Tv* vec_v = (const Tv*)((const char*)v_d + b * nb_v3 + h_kv * nb_v2 + ik * nb_v1);
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
    const int omp_threads = backend_thread_count(backend);

    ops_attention_params params;
    if (!ops_extract_attention_params(node, params)) return false;

    const struct ggml_tensor * q      = params.q;
    const struct ggml_tensor * k      = params.k;
    const struct ggml_tensor * v      = params.v;
    const struct ggml_tensor * bias   = params.bias;
    struct ggml_tensor *       attn_w = params.attn_w;
    struct ggml_tensor *       dst    = node;

    ggml_tensor active_k;
    ggml_tensor active_v;
    if (params.valid_length) {
        const int32_t active_length = *static_cast<const int32_t*>(params.valid_length->data);
        if (active_length <= 0 || active_length > k->ne[1]) return false;
        active_k = *k;
        active_v = *v;
        active_k.ne[1] = active_length;
        active_v.ne[1] = active_length;
        k = &active_k;
        v = &active_v;
    }

    float scale = params.scale;

    const int64_t head_dim   = q->ne[0];
    const int64_t n_heads_q  = q->ne[2];
    const int64_t seq_len_q  = q->ne[1];
    const int64_t batch      = q->ne[3] > 0 ? q->ne[3] : 1;

    const int64_t n_heads_kv = k->ne[2];
    const int64_t seq_len_kv = k->ne[1];

    const int64_t group_size = n_heads_q / n_heads_kv;

    const bool compressed_cache = k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32;
    if (!attn_w && compressed_cache && seq_len_q > 8) {
        const int64_t cache_rows = batch * n_heads_kv * seq_len_kv;
        std::vector<float> k_f32(static_cast<size_t>(cache_rows * head_dim));
        std::vector<float> v_f32(static_cast<size_t>(cache_rows * head_dim));
        const ggml_type_traits* k_traits = ggml_get_type_traits(k->type);
        const ggml_type_traits* v_traits = ggml_get_type_traits(v->type);
        #pragma omp parallel for collapse(3) num_threads(omp_threads)
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t h = 0; h < n_heads_kv; ++h) {
                for (int64_t ik = 0; ik < seq_len_kv; ++ik) {
                    const int64_t row = (b * n_heads_kv + h) * seq_len_kv + ik;
                    const char* k_row = static_cast<const char*>(k->data) + b * k->nb[3] + h * k->nb[2] + ik * k->nb[1];
                    const char* v_row = static_cast<const char*>(v->data) + b * v->nb[3] + h * v->nb[2] + ik * v->nb[1];
                    if (k->type == GGML_TYPE_F32) std::memcpy(k_f32.data() + row * head_dim, k_row, head_dim * sizeof(float));
                    else k_traits->to_float(k_row, k_f32.data() + row * head_dim, head_dim);
                    if (v->type == GGML_TYPE_F32) std::memcpy(v_f32.data() + row * head_dim, v_row, head_dim * sizeof(float));
                    else v_traits->to_float(v_row, v_f32.data() + row * head_dim, head_dim);
                }
            }
        }
        ggml_tensor k_view = *k;
        ggml_tensor v_view = *v;
        k_view.type = GGML_TYPE_F32;
        v_view.type = GGML_TYPE_F32;
        k_view.data = k_f32.data();
        v_view.data = v_f32.data();
        k_view.nb[0] = v_view.nb[0] = sizeof(float);
        k_view.nb[1] = v_view.nb[1] = head_dim * sizeof(float);
        k_view.nb[2] = v_view.nb[2] = seq_len_kv * k_view.nb[1];
        k_view.nb[3] = v_view.nb[3] = n_heads_kv * k_view.nb[2];
        compute_attention_streaming(q, &k_view, &v_view, bias, dst, scale, omp_threads);
        return true;
    }
    // The materialized path below only has uniform-type instantiations; mixed
    // float widths (e.g. F16 q over an F32 cache) must take the streaming path,
    // which reads every element through its actual type.
    const bool uniform_f32 = q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32 &&
                             v->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;
    const bool uniform_f16 = q->type == GGML_TYPE_F16 && k->type == GGML_TYPE_F16 &&
                             v->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16;
    if (!attn_w && (seq_len_q <= 8 || compressed_cache || !(uniform_f32 || uniform_f16))) {
        compute_attention_streaming(q, k, v, bias, dst, scale, omp_threads);
        return true;
    }
    if (!(uniform_f32 || uniform_f16)) {
        return false;
    }

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

    size_t nb_bias0 = bias ? bias->nb[0] : 0;
    size_t nb_bias1 = bias ? bias->nb[1] : 0;
    size_t nb_bias2 = bias ? bias->nb[2] : 0;
    size_t nb_bias3 = bias ? bias->nb[3] : 0;

    const size_t nb_w0 = attn_w ? attn_w->nb[0] : 0;
    const size_t nb_w1 = attn_w ? attn_w->nb[1] : 0;
    const size_t nb_w2 = attn_w ? attn_w->nb[2] : 0;
    const size_t nb_w3 = attn_w ? attn_w->nb[3] : 0;

    // Convert bias to a dense F32 copy if it is F16. The copy walks the source
    // through its byte strides, and the impl must then index it with dense F32
    // strides instead of the original tensor's F16 strides.
    std::vector<float> bias_f32;
    if (bias && bias->type == GGML_TYPE_F16) {
        bias_f32.resize(static_cast<size_t>(ggml_nelements(bias)));
        float* out = bias_f32.data();
        for (int64_t i3 = 0; i3 < bias->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < bias->ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < bias->ne[1]; ++i1) {
                    const char* row = static_cast<const char*>(bias->data) +
                        i3 * bias->nb[3] + i2 * bias->nb[2] + i1 * bias->nb[1];
                    for (int64_t i0 = 0; i0 < bias->ne[0]; ++i0) {
                        *out++ = ggml_fp16_to_fp32(
                            *reinterpret_cast<const ggml_fp16_t*>(row + i0 * bias->nb[0]));
                    }
                }
            }
        }
        nb_bias0 = sizeof(float);
        nb_bias1 = bias->ne[0] * nb_bias0;
        nb_bias2 = bias->ne[1] * nb_bias1;
        nb_bias3 = bias->ne[2] * nb_bias2;
    }
    const float* bias_ptr = bias ? (bias->type == GGML_TYPE_F32 ? (const float*)bias->data : bias_f32.data()) : nullptr;

    // Dispatch based on types
    #define DISPATCH_ATTN(Tq, Tk, Tv, Td, Tw) \
        compute_attention_impl<Tq, Tk, Tv, Td, Tw>( \
            (const Tq*)q->data, (const Tk*)k->data, (const Tv*)v->data, bias_ptr, (Tw*)(attn_w ? attn_w->data : nullptr), (Td*)dst->data, \
            head_dim, n_heads_q, seq_len_q, batch, n_heads_kv, seq_len_kv, group_size, scale, \
            nb_q1, nb_q2, nb_q3, nb_k1, nb_k2, nb_k3, nb_v1, nb_v2, nb_v3, nb_dst1, nb_dst2, nb_dst3, \
            nb_bias0, nb_bias1, nb_bias2, nb_bias3, nb_w0, nb_w1, nb_w2, nb_w3, omp_threads)

    if (q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32 && v->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        if (!attn_w || attn_w->type == GGML_TYPE_F32) {
            DISPATCH_ATTN(float, float, float, float, float);
        } else {
            DISPATCH_ATTN(float, float, float, float, ggml_fp16_t);
        }
    } else {
        // uniform_f16 — guaranteed by the gate above
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
