#include "ops/ops.h"
#include "ops_sycl.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "common.hpp"
#include <oneapi/mkl/blas.hpp>

namespace ggml_ops_ext {
namespace sycl {

struct SyclAttentionWorkspace {
    float* ptr = nullptr;
    size_t size = 0;
    
    const float** q_ptrs = nullptr;
    const float** k_ptrs = nullptr;
    const float** v_ptrs = nullptr;
    float** scores_ptrs = nullptr;
    float** dst_ptrs = nullptr;
    size_t max_ptrs_count = 0;
    
    ::sycl::queue* last_queue = nullptr;

    ~SyclAttentionWorkspace() {
        if (last_queue) {
            if (ptr) ::sycl::free(ptr, *last_queue);
            if (q_ptrs) ::sycl::free((void*)q_ptrs, *last_queue);
            if (k_ptrs) ::sycl::free((void*)k_ptrs, *last_queue);
            if (v_ptrs) ::sycl::free((void*)v_ptrs, *last_queue);
            if (scores_ptrs) ::sycl::free(scores_ptrs, *last_queue);
            if (dst_ptrs) ::sycl::free(dst_ptrs, *last_queue);
        }
    }

    void allocate(::sycl::queue* q, size_t req_size, size_t ptrs_count) {
        if (last_queue != q || size < req_size || max_ptrs_count < ptrs_count) {
            if (last_queue) {
                if (ptr) ::sycl::free(ptr, *last_queue);
                if (q_ptrs) ::sycl::free((void*)q_ptrs, *last_queue);
                if (k_ptrs) ::sycl::free((void*)k_ptrs, *last_queue);
                if (v_ptrs) ::sycl::free((void*)v_ptrs, *last_queue);
                if (scores_ptrs) ::sycl::free(scores_ptrs, *last_queue);
                if (dst_ptrs) ::sycl::free(dst_ptrs, *last_queue);
                ptr = nullptr;
                q_ptrs = nullptr;
                k_ptrs = nullptr;
                v_ptrs = nullptr;
                scores_ptrs = nullptr;
                dst_ptrs = nullptr;
            }
            last_queue = q;
            
            if (req_size > 0) {
                ptr = ::sycl::malloc_device<float>(req_size, *q);
                size = req_size;
            }
            if (ptrs_count > 0) {
                q_ptrs = ::sycl::malloc_shared<const float*>(ptrs_count, *q);
                k_ptrs = ::sycl::malloc_shared<const float*>(ptrs_count, *q);
                v_ptrs = ::sycl::malloc_shared<const float*>(ptrs_count, *q);
                scores_ptrs = ::sycl::malloc_shared<float*>(ptrs_count, *q);
                dst_ptrs = ::sycl::malloc_shared<float*>(ptrs_count, *q);
                max_ptrs_count = ptrs_count;
            }
        }
    }
};

static thread_local SyclAttentionWorkspace g_attn_workspace;

class AttentionSoftmaxBiasSYCLKernel;

bool ggml_sycl_op_attention(
    ggml_backend_t backend,
    struct ggml_tensor* node
) {
    try {
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

        ::sycl::queue* q_sycl = (::sycl::queue*)ggml_ops_ext_bridge_sycl_get_queue(backend);
        if (!q_sycl) return false;

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

        const float* bias_d = bias ? (const float*)bias->data : nullptr;
        float* attn_w_d = attn_w ? (float*)attn_w->data : nullptr;

        size_t scores_size = batch * n_heads_q * seq_len_q * seq_len_kv;
        size_t ptrs_count = batch * n_heads_q;

        // Allocate workspace and USM shared memory pointer arrays
        g_attn_workspace.allocate(q_sycl, scores_size, ptrs_count);

        // Fill pointers to the batch elements
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t h_q = 0; h_q < n_heads_q; ++h_q) {
                int64_t h_kv = h_q / group_size;
                int64_t idx = b * n_heads_q + h_q;
                
                g_attn_workspace.q_ptrs[idx] = (const float*)((const char*)q_d + b * nb_q3 + h_q * nb_q2);
                g_attn_workspace.k_ptrs[idx] = (const float*)((const char*)k_d + b * nb_k3 + h_kv * nb_k2);
                g_attn_workspace.v_ptrs[idx] = (const float*)((const char*)v_d + b * nb_v3 + h_kv * nb_v2);
                g_attn_workspace.scores_ptrs[idx] = g_attn_workspace.ptr + idx * seq_len_q * seq_len_kv;
                g_attn_workspace.dst_ptrs[idx] = (float*)((char*)dst_d + b * nb_dst3 + h_q * nb_dst2);
            }
        }

        // 1. Compute dot products: scores = scale * K^T @ Q using pointer-based gemm_batch
        oneapi::mkl::transpose transa1 = oneapi::mkl::transpose::trans;
        oneapi::mkl::transpose transb1 = oneapi::mkl::transpose::nontrans;
        int64_t m1 = seq_len_kv;
        int64_t n1 = seq_len_q;
        int64_t k1 = head_dim;
        int64_t lda1 = nb_k1 / 4;
        int64_t ldb1 = nb_q1 / 4;
        int64_t ldc1 = seq_len_kv;
        float alpha1 = scale;
        float beta1 = 0.0f;
        int64_t gsize1 = ptrs_count;

        oneapi::mkl::blas::column_major::gemm_batch(
            *q_sycl,
            &transa1, &transb1,
            &m1, &n1, &k1,
            &alpha1, g_attn_workspace.k_ptrs, &lda1,
            g_attn_workspace.q_ptrs, &ldb1,
            &beta1, g_attn_workspace.scores_ptrs, &ldc1,
            1, &gsize1
        );

        // 2. Compute Softmax and Add Bias on SYCL Device
        int64_t total_queries = batch * n_heads_q * seq_len_q;
        constexpr int block_size = 256;
        float* scores_base_ptr = g_attn_workspace.ptr;

        q_sycl->submit([&](::sycl::handler &cgh) {
            ::sycl::local_accessor<float, 1> sdata(::sycl::range<1>(block_size), cgh);

            cgh.parallel_for<AttentionSoftmaxBiasSYCLKernel>(
                ::sycl::nd_range<1>(
                    ::sycl::range<1>(total_queries * block_size),
                    ::sycl::range<1>(block_size)
                ),
                [=](::sycl::nd_item<1> item) {
                    int64_t b_h_iq = item.get_group(0);
                    int tid = item.get_local_id(0);

                    int64_t iq = b_h_iq % seq_len_q;
                    int64_t tmp = b_h_iq / seq_len_q;
                    int64_t h_q = tmp % n_heads_q;
                    int64_t b = tmp / n_heads_q;

                    float* score_row = scores_base_ptr + b_h_iq * seq_len_kv;
                    const char* bias_row = bias_d ? ((const char*)bias_d + b * nb_bias3 + h_q * nb_bias2 + iq * nb_bias1) : nullptr;
                    char* w_row = attn_w_d ? ((char*)attn_w_d + b * nb_w3 + h_q * nb_w2 + iq * nb_w1) : nullptr;

                    // 1. Add optional bias & Find max_score
                    float local_max = -1e20f;
                    for (int64_t ik = tid; ik < seq_len_kv; ik += block_size) {
                        float s = score_row[ik];
                        if (bias_row) {
                            const float* b_ptr = (const float*)(bias_row + ik * nb_bias0);
                            s += *b_ptr;
                            score_row[ik] = s;
                        }
                        if (s > local_max) local_max = s;
                    }

                    sdata[tid] = local_max;
                    item.barrier(::sycl::access::fence_space::local_space);

                    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
                        if (tid < s) {
                            sdata[tid] = sdata[tid] > sdata[tid + s] ? sdata[tid] : sdata[tid + s];
                        }
                        item.barrier(::sycl::access::fence_space::local_space);
                    }
                    float max_score = sdata[0];

                    // 2. Compute sum of exponentials
                    float local_sum = 0.0f;
                    for (int64_t ik = tid; ik < seq_len_kv; ik += block_size) {
                        float val = ::sycl::exp(score_row[ik] - max_score);
                        score_row[ik] = val;
                        local_sum += val;
                    }

                    sdata[tid] = local_sum;
                    item.barrier(::sycl::access::fence_space::local_space);

                    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
                        if (tid < s) {
                            sdata[tid] += sdata[tid + s];
                        }
                        item.barrier(::sycl::access::fence_space::local_space);
                    }
                    float sum_exp = sdata[0];
                    float inv_sum = 1.0f / (sum_exp + 1e-9f);

                    // 3. Normalize & Write to attn_w if needed
                    for (int64_t ik = tid; ik < seq_len_kv; ik += block_size) {
                        float soft_val = score_row[ik] * inv_sum;
                        score_row[ik] = soft_val;

                        if (w_row) {
                            float* w_ptr = (float*)(w_row + ik * nb_w0);
                            *w_ptr = soft_val;
                        }
                    }
                }
            );
        });

        // 3. Compute weighted sum: dst = V @ scores using pointer-based gemm_batch
        oneapi::mkl::transpose transa2 = oneapi::mkl::transpose::nontrans;
        oneapi::mkl::transpose transb2 = oneapi::mkl::transpose::nontrans;
        int64_t m2 = head_dim;
        int64_t n2 = seq_len_q;
        int64_t k2 = seq_len_kv;
        int64_t lda2 = nb_v1 / 4;
        int64_t ldb2 = seq_len_kv;
        int64_t ldc2 = nb_dst1 / 4;
        float alpha2 = 1.0f;
        float beta2 = 0.0f;
        int64_t gsize2 = ptrs_count;

        oneapi::mkl::blas::column_major::gemm_batch(
            *q_sycl,
            &transa2, &transb2,
            &m2, &n2, &k2,
            &alpha2, g_attn_workspace.v_ptrs, &lda2,
            (const float**)g_attn_workspace.scores_ptrs, &ldb2,
            &beta2, g_attn_workspace.dst_ptrs, &ldc2,
            1, &gsize2
        );
        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "SYCL Fused Attention Exception: %s\n", e.what());
        return false;
    } catch (...) {
        fprintf(stderr, "SYCL Fused Attention Exception: Unknown\n");
        return false;
    }
}

bool ggml_sycl_op_attention_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    return ggml_sycl_op_attention(backend, node);
}

} // namespace sycl
} // namespace ggml_ops_ext
