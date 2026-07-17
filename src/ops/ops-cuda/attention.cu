#include "ops_cuda_common.cuh"
#include "quantized_conv.cuh"

namespace ggml_ops_ext {
namespace cuda {

template <ggml_type Type>
__device__ inline float attention_load(const char* row, int64_t index, size_t nb0, int64_t row_elements) {
    if constexpr (Type == GGML_TYPE_F32) {
        return *reinterpret_cast<const float*>(row + index * nb0);
    } else if constexpr (Type == GGML_TYPE_F16) {
        return __half2float(*reinterpret_cast<const half*>(row + index * nb0));
    } else {
        return load_quantized_row_value<Type>(row, 0, index, row_elements);
    }
}

template <ggml_type QType, ggml_type KType, ggml_type VType>
__global__ void attention_streaming_kernel(
    const void* q,
    const void* k,
    const void* v,
    const float* bias,
    void* dst,
    int64_t head_dim,
    int64_t seq_len_q,
    int64_t seq_len_kv,
    int64_t n_heads_q,
    int64_t n_heads_kv,
    float scale,
    size_t nb_q0, size_t nb_q1, size_t nb_q2, size_t nb_q3,
    size_t nb_k0, size_t nb_k1, size_t nb_k2, size_t nb_k3,
    size_t nb_v0, size_t nb_v1, size_t nb_v2, size_t nb_v3,
    size_t nb_dst0, size_t nb_dst1, size_t nb_dst2, size_t nb_dst3,
    size_t nb_bias0, size_t nb_bias1, size_t nb_bias2, size_t nb_bias3
) {
    const int lane = threadIdx.x;
    const int64_t row = blockIdx.x;
    const int64_t iq = row % seq_len_q;
    const int64_t tmp = row / seq_len_q;
    const int64_t h_q = tmp % n_heads_q;
    const int64_t batch = tmp / n_heads_q;
    const int64_t h_kv = h_q / (n_heads_q / n_heads_kv);

    const char* q_row = reinterpret_cast<const char*>(q) + batch * nb_q3 + h_q * nb_q2 + iq * nb_q1;
    char* dst_row = reinterpret_cast<char*>(dst) + batch * nb_dst3 + h_q * nb_dst2 + iq * nb_dst1;

    float accumulator[8] = {};
    float running_max = -3.402823466e+38F;
    float running_sum = 0.0f;

    for (int64_t ik = 0; ik < seq_len_kv; ++ik) {
        const char* k_row = reinterpret_cast<const char*>(k) + batch * nb_k3 + h_kv * nb_k2 + ik * nb_k1;
        float dot = 0.0f;
        for (int64_t d = lane; d < head_dim; d += 32) {
            dot += attention_load<QType>(q_row, d, nb_q0, head_dim) *
                   attention_load<KType>(k_row, d, nb_k0, head_dim);
        }
        for (int offset = 16; offset > 0; offset >>= 1) dot += __shfl_down_sync(0xffffffff, dot, offset);

        float score = __shfl_sync(0xffffffff, dot, 0) * scale;
        if (lane == 0 && bias) {
            const char* bias_value = reinterpret_cast<const char*>(bias) + batch * nb_bias3 +
                                     h_q * nb_bias2 + iq * nb_bias1 + ik * nb_bias0;
            score += *reinterpret_cast<const float*>(bias_value);
        }
        score = __shfl_sync(0xffffffff, score, 0);
        if (isinf(score) && score < 0.0f) continue;

        const float next_max = fmaxf(running_max, score);
        const float previous_scale = running_sum == 0.0f ? 0.0f : expf(running_max - next_max);
        const float current_scale = expf(score - next_max);
        running_sum = running_sum * previous_scale + current_scale;

        const char* v_row = reinterpret_cast<const char*>(v) + batch * nb_v3 + h_kv * nb_v2 + ik * nb_v1;
#pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            const int64_t d = lane + slot * 32;
            if (d < head_dim) {
                const float value = attention_load<VType>(v_row, d, nb_v0, head_dim);
                accumulator[slot] = accumulator[slot] * previous_scale + current_scale * value;
            }
        }
        running_max = next_max;
    }

    const float inv_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
#pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
        const int64_t d = lane + slot * 32;
        if (d < head_dim) {
            if constexpr (QType == GGML_TYPE_F32) {
                *reinterpret_cast<float*>(dst_row + d * nb_dst0) = accumulator[slot] * inv_sum;
            } else {
                *reinterpret_cast<half*>(dst_row + d * nb_dst0) = __float2half(accumulator[slot] * inv_sum);
            }
        }
    }
}

template <ggml_type QType, ggml_type KType, ggml_type VType>
__global__ void attention_tiled_quantized_prefill_kernel(
    const void* q, const void* k, const void* v, const float* bias, void* dst,
    int64_t head_dim, int64_t seq_len_q, int64_t seq_len_kv,
    int64_t n_heads_q, int64_t n_heads_kv, float scale,
    size_t nb_q0, size_t nb_q1, size_t nb_q2, size_t nb_q3,
    size_t nb_k0, size_t nb_k1, size_t nb_k2, size_t nb_k3,
    size_t nb_v0, size_t nb_v1, size_t nb_v2, size_t nb_v3,
    size_t nb_dst0, size_t nb_dst1, size_t nb_dst2, size_t nb_dst3,
    size_t nb_bias0, size_t nb_bias1, size_t nb_bias2, size_t nb_bias3
) {
    constexpr int warps_per_block = 8;
    constexpr int key_tile = 32;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int64_t query_tiles = (seq_len_q + warps_per_block - 1) / warps_per_block;
    const int64_t query_tile = blockIdx.x % query_tiles;
    const int64_t tmp = blockIdx.x / query_tiles;
    const int64_t h_q = tmp % n_heads_q;
    const int64_t batch = tmp / n_heads_q;
    const int64_t h_kv = h_q / (n_heads_q / n_heads_kv);
    const int64_t iq = query_tile * warps_per_block + warp;
    const bool active = iq < seq_len_q;

    const char* q_row = active ? reinterpret_cast<const char*>(q) + batch * nb_q3 +
        h_q * nb_q2 + iq * nb_q1 : nullptr;
    char* dst_row = active ? reinterpret_cast<char*>(dst) + batch * nb_dst3 +
        h_q * nb_dst2 + iq * nb_dst1 : nullptr;
    extern __shared__ float shared[];
    float* shared_k = shared;
    float* shared_v = shared + key_tile * head_dim;
    float accumulator[4] = {};
    float running_max = -3.402823466e+38F;
    float running_sum = 0.0f;

    for (int64_t key_start = 0; key_start < seq_len_kv; key_start += key_tile) {
        const int tile_elements = key_tile * static_cast<int>(head_dim);
        for (int linear = threadIdx.x; linear < tile_elements; linear += blockDim.x) {
            const int key_offset = linear / head_dim;
            const int dim = linear - key_offset * head_dim;
            const int64_t ik = key_start + key_offset;
            float key_value = 0.0f;
            float value_value = 0.0f;
            if (ik < seq_len_kv) {
                const char* k_row = reinterpret_cast<const char*>(k) + batch * nb_k3 +
                    h_kv * nb_k2 + ik * nb_k1;
                const char* v_row = reinterpret_cast<const char*>(v) + batch * nb_v3 +
                    h_kv * nb_v2 + ik * nb_v1;
                key_value = attention_load<KType>(k_row, dim, nb_k0, head_dim);
                value_value = attention_load<VType>(v_row, dim, nb_v0, head_dim);
            }
            shared_k[linear] = key_value;
            shared_v[linear] = value_value;
        }
        __syncthreads();

        if (active) {
            const int valid_keys = min(key_tile, static_cast<int>(seq_len_kv - key_start));
            for (int key_offset = 0; key_offset < valid_keys; ++key_offset) {
                float dot = 0.0f;
                for (int64_t d = lane; d < head_dim; d += 32) {
                    dot += attention_load<QType>(q_row, d, nb_q0, head_dim) *
                           shared_k[key_offset * head_dim + d];
                }
                for (int offset = 16; offset > 0; offset >>= 1) {
                    dot += __shfl_down_sync(0xffffffff, dot, offset);
                }
                float score = __shfl_sync(0xffffffff, dot, 0) * scale;
                if (lane == 0 && bias) {
                    const char* bias_value = reinterpret_cast<const char*>(bias) + batch * nb_bias3 +
                        h_q * nb_bias2 + iq * nb_bias1 + (key_start + key_offset) * nb_bias0;
                    score += *reinterpret_cast<const float*>(bias_value);
                }
                score = __shfl_sync(0xffffffff, score, 0);
                if (isinf(score) && score < 0.0f) continue;
                const float next_max = fmaxf(running_max, score);
                const float previous_scale = running_sum == 0.0f ? 0.0f : expf(running_max - next_max);
                const float current_scale = expf(score - next_max);
                running_sum = running_sum * previous_scale + current_scale;
#pragma unroll
                for (int slot = 0; slot < 4; ++slot) {
                    const int64_t d = lane + slot * 32;
                    if (d < head_dim) {
                        accumulator[slot] = accumulator[slot] * previous_scale +
                            current_scale * shared_v[key_offset * head_dim + d];
                    }
                }
                running_max = next_max;
            }
        }
        __syncthreads();
    }

    if (active) {
        const float inv_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            const int64_t d = lane + slot * 32;
            if (d < head_dim) {
                if constexpr (QType == GGML_TYPE_F32) {
                    *reinterpret_cast<float*>(dst_row + d * nb_dst0) = accumulator[slot] * inv_sum;
                } else {
                    *reinterpret_cast<half*>(dst_row + d * nb_dst0) = __float2half(accumulator[slot] * inv_sum);
                }
            }
        }
    }
}

template <ggml_type QType, ggml_type KType, ggml_type VType>
static void launch_attention_streaming(
    cudaStream_t stream, const ggml_tensor* q, const ggml_tensor* k, const ggml_tensor* v,
    const ggml_tensor* bias, ggml_tensor* dst, int64_t total_queries, float scale
) {
    constexpr bool quantized_cache = KType == GGML_TYPE_Q4_0 || KType == GGML_TYPE_Q8_0 ||
                                     VType == GGML_TYPE_Q4_0 || VType == GGML_TYPE_Q8_0;
    if constexpr (quantized_cache) {
        if (q->ne[1] > 8 && q->ne[0] <= 128) {
            constexpr int warps_per_block = 8;
            const int64_t query_tiles = (q->ne[1] + warps_per_block - 1) / warps_per_block;
            const int64_t blocks = q->ne[3] * q->ne[2] * query_tiles;
            const size_t shared_bytes = 2 * 32 * q->ne[0] * sizeof(float);
            attention_tiled_quantized_prefill_kernel<QType, KType, VType>
                <<<blocks, warps_per_block * 32, shared_bytes, stream>>>(
                    q->data, k->data, v->data,
                    bias ? static_cast<const float*>(bias->data) : nullptr, dst->data,
                    q->ne[0], q->ne[1], k->ne[1], q->ne[2], k->ne[2], scale,
                    q->nb[0], q->nb[1], q->nb[2], q->nb[3],
                    k->nb[0], k->nb[1], k->nb[2], k->nb[3],
                    v->nb[0], v->nb[1], v->nb[2], v->nb[3],
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                    bias ? bias->nb[0] : 0, bias ? bias->nb[1] : 0,
                    bias ? bias->nb[2] : 0, bias ? bias->nb[3] : 0);
            return;
        }
    }
    attention_streaming_kernel<QType, KType, VType><<<total_queries, 32, 0, stream>>>(
        q->data, k->data, v->data, bias ? static_cast<const float*>(bias->data) : nullptr, dst->data,
        q->ne[0], q->ne[1], k->ne[1], q->ne[2], k->ne[2], scale,
        q->nb[0], q->nb[1], q->nb[2], q->nb[3],
        k->nb[0], k->nb[1], k->nb[2], k->nb[3],
        v->nb[0], v->nb[1], v->nb[2], v->nb[3],
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
        bias ? bias->nb[0] : 0, bias ? bias->nb[1] : 0,
        bias ? bias->nb[2] : 0, bias ? bias->nb[3] : 0);
}

template <ggml_type QType, ggml_type KType>
static bool launch_attention_streaming_v(
    cudaStream_t stream, const ggml_tensor* q, const ggml_tensor* k, const ggml_tensor* v,
    const ggml_tensor* bias, ggml_tensor* dst, int64_t total_queries, float scale
) {
    switch (v->type) {
        case GGML_TYPE_F32:  launch_attention_streaming<QType, KType, GGML_TYPE_F32 >(stream, q, k, v, bias, dst, total_queries, scale); return true;
        case GGML_TYPE_F16:  launch_attention_streaming<QType, KType, GGML_TYPE_F16 >(stream, q, k, v, bias, dst, total_queries, scale); return true;
        case GGML_TYPE_Q8_0: launch_attention_streaming<QType, KType, GGML_TYPE_Q8_0>(stream, q, k, v, bias, dst, total_queries, scale); return true;
        case GGML_TYPE_Q4_0: launch_attention_streaming<QType, KType, GGML_TYPE_Q4_0>(stream, q, k, v, bias, dst, total_queries, scale); return true;
        default: return false;
    }
}

template <ggml_type QType>
static bool launch_attention_streaming_kv(
    cudaStream_t stream, const ggml_tensor* q, const ggml_tensor* k, const ggml_tensor* v,
    const ggml_tensor* bias, ggml_tensor* dst, int64_t total_queries, float scale
) {
    switch (k->type) {
        case GGML_TYPE_F32:  return launch_attention_streaming_v<QType, GGML_TYPE_F32 >(stream, q, k, v, bias, dst, total_queries, scale);
        case GGML_TYPE_F16:  return launch_attention_streaming_v<QType, GGML_TYPE_F16 >(stream, q, k, v, bias, dst, total_queries, scale);
        case GGML_TYPE_Q8_0: return launch_attention_streaming_v<QType, GGML_TYPE_Q8_0>(stream, q, k, v, bias, dst, total_queries, scale);
        case GGML_TYPE_Q4_0: return launch_attention_streaming_v<QType, GGML_TYPE_Q4_0>(stream, q, k, v, bias, dst, total_queries, scale);
        default: return false;
    }
}

__global__ void attention_softmax_bias_kernel(
    float* scores,               // [batch, n_heads_q, seq_len_q, seq_len_kv]
    const float* bias,           // [seq_len_kv, seq_len_q, n_heads_q, batch] (optional)
    float* attn_w,               // [seq_len_kv, seq_len_q, n_heads_q, batch] (optional)
    int64_t seq_len_kv,
    int64_t seq_len_q,
    int64_t n_heads_q,
    size_t nb_bias0, size_t nb_bias1, size_t nb_bias2, size_t nb_bias3,
    size_t nb_w0, size_t nb_w1, size_t nb_w2, size_t nb_w3
) {
    // Each block handles one query of one head of one batch
    int64_t b_h_iq = blockIdx.x; // index in range [0, batch * n_heads_q * seq_len_q - 1]
    if (b_h_iq >= gridDim.x) return;

    int64_t iq = b_h_iq % seq_len_q;
    int64_t tmp = b_h_iq / seq_len_q;
    int64_t h_q = tmp % n_heads_q;
    int64_t b = tmp / n_heads_q;

    float* score_row = scores + b_h_iq * seq_len_kv;
    const char* bias_row = bias ? ((const char*)bias + b * nb_bias3 + h_q * nb_bias2 + iq * nb_bias1) : nullptr;
    char* w_row = attn_w ? ((char*)attn_w + b * nb_w3 + h_q * nb_w2 + iq * nb_w1) : nullptr;

    // 1. Add optional bias & Find max_score
    float local_max = -1e20f;
    for (int64_t ik = threadIdx.x; ik < seq_len_kv; ik += blockDim.x) {
        float s = score_row[ik];
        if (bias) {
            const float* b_ptr = (const float*)(bias_row + ik * nb_bias0);
            s += *b_ptr;
            score_row[ik] = s; // write back updated score
        }
        if (s > local_max) local_max = s;
    }

    // Block-level reduction for max_score using shared memory
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    sdata[tid] = local_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float max_score = sdata[0];

    // 2. Compute sum of exponentials
    float local_sum = 0.0f;
    for (int64_t ik = threadIdx.x; ik < seq_len_kv; ik += blockDim.x) {
        float val = expf(score_row[ik] - max_score);
        score_row[ik] = val;
        local_sum += val;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float sum_exp = sdata[0];
    float inv_sum = 1.0f / (sum_exp + 1e-9f);

    // 3. Normalize & Write to attn_w if needed
    for (int64_t ik = threadIdx.x; ik < seq_len_kv; ik += blockDim.x) {
        float soft_val = score_row[ik] * inv_sum;
        score_row[ik] = soft_val;

        if (attn_w) {
            float* w_ptr = (float*)(w_row + ik * nb_w0);
            *w_ptr = soft_val;
        }
    }
}

bool ggml_cuda_op_attention(
    ggml_backend_t backend,
    struct ggml_tensor* node
) {
    ops_attention_params params;
    if (!ops_extract_attention_params(node, params)) return false;

    const struct ggml_tensor* q      = params.q;
    const struct ggml_tensor* k      = params.k;
    const struct ggml_tensor* v      = params.v;
    const struct ggml_tensor* bias   = params.bias;
    struct ggml_tensor*       attn_w = params.attn_w;
    struct ggml_tensor*       dst    = node;

    float scale = params.scale;

    const int64_t head_dim   = q->ne[0];
    const int64_t n_heads_q  = q->ne[2];
    const int64_t seq_len_q  = q->ne[1];
    const int64_t batch      = q->ne[3] > 0 ? q->ne[3] : 1;

    const int64_t n_heads_kv = k->ne[2];
    const int64_t seq_len_kv = k->ne[1];

    const int64_t group_size = n_heads_q / n_heads_kv;

    int device = ggml_ops_ext_bridge_cuda_get_device(backend);
    cudaStream_t stream = (cudaStream_t)ggml_ops_ext_bridge_cuda_get_stream(backend);
    cublasHandle_t cublas = (cublasHandle_t)ggml_ops_ext_bridge_cuda_get_cublas(backend);

    CUDA_CHECK(cudaSetDevice(device));
    CUBLAS_CHECK(cublasSetStream(cublas, stream));

    const float* q_d = (const float*)q->data;
    const float* k_d = (const float*)k->data;
    const float* v_d = (const float*)v->data;
    float*       dst_d = (float*)dst->data;

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

    const bool float_gemm_path = q->type == GGML_TYPE_F32 &&
                                 k->type == GGML_TYPE_F32 && v->type == GGML_TYPE_F32;
    if (!attn_w && !float_gemm_path && head_dim <= 256) {
        const int64_t total_queries = batch * n_heads_q * seq_len_q;
        const bool launched = q->type == GGML_TYPE_F32
            ? launch_attention_streaming_kv<GGML_TYPE_F32>(stream, q, k, v, bias, dst, total_queries, scale)
            : launch_attention_streaming_kv<GGML_TYPE_F16>(stream, q, k, v, bias, dst, total_queries, scale);
        if (!launched) return false;
        CUDA_CHECK(cudaGetLastError());
        return true;
    }

    // Stream-ordered lifetime keeps concurrent sessions isolated without a
    // device-wide synchronization.
    size_t scores_size = batch * n_heads_q * seq_len_q * seq_len_kv;
    ops_cuda_alloc<float> scores_alloc(stream);
    scores_alloc.alloc(scores_size);
    float* scores_d = scores_alloc.get();

    // 2. Compute dot products: scores = scale * K^T @ Q
    if (group_size == 1) {
        // Fast path: use a single batched Sgemm call
        float alpha = scale;
        float beta = 0.0f;

        CUBLAS_CHECK(cublasSgemmStridedBatched(cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            seq_len_kv, seq_len_q, head_dim,
            &alpha,
            k_d, nb_k1 / 4, nb_k2 / 4,
            q_d, nb_q1 / 4, nb_q2 / 4,
            &beta,
            scores_d, seq_len_kv, seq_len_q * seq_len_kv,
            batch * n_heads_q));
    } else {
        // Fallback path: loop over heads and batches
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t h_q = 0; h_q < n_heads_q; ++h_q) {
                int64_t h_kv = h_q / group_size;

                const float* ptr_q = (const float*)((const char*)q_d + b * nb_q3 + h_q * nb_q2);
                const float* ptr_k = (const float*)((const char*)k_d + b * nb_k3 + h_kv * nb_k2);
                float* ptr_scores = scores_d + (b * n_heads_q + h_q) * seq_len_q * seq_len_kv;

                float alpha = scale;
                float beta = 0.0f;

                CUBLAS_CHECK(cublasSgemm(cublas,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    seq_len_kv, seq_len_q, head_dim,
                    &alpha,
                    ptr_k, nb_k1 / 4,
                    ptr_q, nb_q1 / 4,
                    &beta,
                    ptr_scores, seq_len_kv));
            }
        }
    }

    // 3. Compute Softmax and Add Bias on GPU
    int64_t total_queries = batch * n_heads_q * seq_len_q;
    int block_size = 256;
    int shared_mem = block_size * sizeof(float);

    attention_softmax_bias_kernel<<<total_queries, block_size, shared_mem, stream>>>(
        scores_d,
        bias ? (const float*)bias->data : nullptr,
        attn_w ? (float*)attn_w->data : nullptr,
        seq_len_kv, seq_len_q, n_heads_q,
        nb_bias0, nb_bias1, nb_bias2, nb_bias3,
        nb_w0, nb_w1, nb_w2, nb_w3
    );
    CUDA_CHECK(cudaGetLastError());

    // 4. Compute weighted sum: dst = V @ scores
    if (group_size == 1) {
        // Fast path: use a single batched Sgemm call
        float alpha = 1.0f;
        float beta = 0.0f;

        CUBLAS_CHECK(cublasSgemmStridedBatched(cublas,
            CUBLAS_OP_N, CUBLAS_OP_N,
            head_dim, seq_len_q, seq_len_kv,
            &alpha,
            v_d, nb_v1 / 4, nb_v2 / 4,
            scores_d, seq_len_kv, seq_len_q * seq_len_kv,
            &beta,
            dst_d, nb_dst1 / 4, nb_dst2 / 4,
            batch * n_heads_q));
    } else {
        // Fallback path: loop over heads and batches
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t h_q = 0; h_q < n_heads_q; ++h_q) {
                int64_t h_kv = h_q / group_size;

                const float* ptr_v = (const float*)((const char*)v_d + b * nb_v3 + h_kv * nb_v2);
                float* ptr_scores = scores_d + (b * n_heads_q + h_q) * seq_len_q * seq_len_kv;
                float* ptr_dst = (float*)((char*)dst_d + b * nb_dst3 + h_q * nb_dst2);

                float alpha = 1.0f;
                float beta = 0.0f;

                CUBLAS_CHECK(cublasSgemm(cublas,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    head_dim, seq_len_q, seq_len_kv,
                    &alpha,
                    ptr_v, nb_v1 / 4,
                    ptr_scores, seq_len_kv,
                    &beta,
                    ptr_dst, nb_dst1 / 4));
            }
        }
    }

    return true;
}

bool ggml_cuda_op_attention_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    return ggml_cuda_op_attention(backend, node);
}

} // namespace cuda
} // namespace ggml_ops_ext
