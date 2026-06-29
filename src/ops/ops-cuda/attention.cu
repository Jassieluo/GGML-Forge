#include "ops_cuda_common.cuh"

namespace ggml_ops_ext {
namespace cuda {

struct CudaAttentionWorkspace {
    float* ptr = nullptr;
    size_t size = 0;
    int device = -1;

    ~CudaAttentionWorkspace() {
        if (ptr) {
            cudaSetDevice(device);
            cudaFree(ptr);
        }
    }

    float* get(int dev, size_t req_size) {
        if (device != dev || size < req_size) {
            if (ptr) {
                cudaSetDevice(device);
                cudaFree(ptr);
                ptr = nullptr;
            }
            device = dev;
            cudaSetDevice(device);
            CUDA_CHECK(cudaMalloc(&ptr, req_size * sizeof(float)));
            size = req_size;
        }
        return ptr;
    }
};

static thread_local CudaAttentionWorkspace g_attn_workspace;

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

    // 1. Add optional bias & Find max_score
    float local_max = -1e20f;
    for (int64_t ik = threadIdx.x; ik < seq_len_kv; ik += blockDim.x) {
        float s = score_row[ik];
        if (bias) {
            const float* b_ptr = (const float*)((const char*)bias + b * nb_bias3 + h_q * nb_bias2 + iq * nb_bias1 + ik * nb_bias0);
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
            float* w_ptr = (float*)((char*)attn_w + b * nb_w3 + h_q * nb_w2 + iq * nb_w1 + ik * nb_w0);
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
    const int64_t n_heads_q  = q->ne[1];
    const int64_t seq_len_q  = q->ne[2];
    const int64_t batch      = q->ne[3] > 0 ? q->ne[3] : 1;

    const int64_t n_heads_kv = k->ne[1];
    const int64_t seq_len_kv = k->ne[2];

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

    // 1. Allocate / fetch persistent workspace
    size_t scores_size = batch * n_heads_q * seq_len_q * seq_len_kv;
    float* scores_d = g_attn_workspace.get(device, scores_size);

    // 2. Compute dot products: scores = scale * K^T @ Q
    if (group_size == 1) {
        // Fast path: use a single batched Sgemm call
        float alpha = scale;
        float beta = 0.0f;

        CUBLAS_CHECK(cublasSgemmStridedBatched(cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            seq_len_kv, seq_len_q, head_dim,
            &alpha,
            k_d, nb_k2 / 4, nb_k1 / 4,
            q_d, nb_q2 / 4, nb_q1 / 4,
            &beta,
            scores_d, seq_len_kv, seq_len_q * seq_len_kv,
            batch * n_heads_q));
    } else {
        // Fallback path: loop over heads and batches
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t h_q = 0; h_q < n_heads_q; ++h_q) {
                int64_t h_kv = h_q / group_size;

                const float* ptr_q = (const float*)((const char*)q_d + b * nb_q3 + h_q * nb_q1);
                const float* ptr_k = (const float*)((const char*)k_d + b * nb_k3 + h_kv * nb_k1);
                float* ptr_scores = scores_d + (b * n_heads_q + h_q) * seq_len_q * seq_len_kv;

                float alpha = scale;
                float beta = 0.0f;

                CUBLAS_CHECK(cublasSgemm(cublas,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    seq_len_kv, seq_len_q, head_dim,
                    &alpha,
                    ptr_k, nb_k2 / 4,
                    ptr_q, nb_q2 / 4,
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
            v_d, nb_v2 / 4, nb_v1 / 4,
            scores_d, seq_len_kv, seq_len_q * seq_len_kv,
            &beta,
            dst_d, nb_dst2 / 4, nb_dst1 / 4,
            batch * n_heads_q));
    } else {
        // Fallback path: loop over heads and batches
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t h_q = 0; h_q < n_heads_q; ++h_q) {
                int64_t h_kv = h_q / group_size;

                const float* ptr_v = (const float*)((const char*)v_d + b * nb_v3 + h_kv * nb_v1);
                float* ptr_scores = scores_d + (b * n_heads_q + h_q) * seq_len_q * seq_len_kv;
                float* ptr_dst = (float*)((char*)dst_d + b * nb_dst3 + h_q * nb_dst1);

                float alpha = 1.0f;
                float beta = 0.0f;

                CUBLAS_CHECK(cublasSgemm(cublas,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    head_dim, seq_len_q, seq_len_kv,
                    &alpha,
                    ptr_v, nb_v2 / 4,
                    ptr_scores, seq_len_kv,
                    &beta,
                    ptr_dst, nb_dst2 / 4));
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
