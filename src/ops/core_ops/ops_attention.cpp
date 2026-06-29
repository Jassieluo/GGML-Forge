#include "ops/ops.h"
#include "ggml-impl.h"
#include <cmath>
#include <cstring>

struct ggml_tensor* ggml_ops_attention(
    struct ggml_context* ctx,
    struct ggml_tensor* q,
    struct ggml_tensor* k,
    struct ggml_tensor* v,
    struct ggml_tensor* bias,
    struct ggml_tensor* attn_w,
    float scale,
    int32_t window_size,
    ggml_backend_t backend
) {
    // 1. Check if the backend registers a custom builder
    ggml_ops_ext::ops_op_builder_t builder = ggml_ops_ext::find_ops_builder(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_FUSED_ATTN);
    if (builder) {
        struct ggml_tensor* srcs[] = { q, k, v, bias, attn_w };
        union {
            float f;
            int32_t i;
        } u_scale;
        u_scale.f = scale;
        int32_t params[] = { u_scale.i, window_size };
        return builder(ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_FUSED_ATTN, srcs, 5, params, 2, backend);
    }

    // 2. Otherwise check if it supports direct handler execution (virtual node)
    if (ggml_ops_backend_supports_op(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_FUSED_ATTN)) {
        struct ggml_tensor* srcs[] = { q, k, v, bias, attn_w };
        struct ggml_tensor* result = ggml_ops_ext::ops_new_virtual_node(
            ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_FUSED_ATTN, q->type, ggml_n_dims(q), q->ne, 5, srcs);
        
        union {
            float f;
            int32_t i;
        } u_scale;
        u_scale.f = scale;
        int32_t params[] = { u_scale.i, window_size };
        ggml_set_op_params(result, params, sizeof(params));
        return result;
    }

    // 3. Fallback: construct standard GGUF/GGML subgraph
    // This allows complete functionality on any backend without custom handler registration.
    
    // Q: [head_dim, n_heads_q, seq_len_q, batch]
    // K: [head_dim, n_heads_kv, seq_len_kv, batch]
    // V: [head_dim, n_heads_kv, seq_len_kv, batch]
    
    // Q_perm: [head_dim, seq_len_q, n_heads_q, batch]
    struct ggml_tensor* q_perm = ggml_permute(ctx, q, 0, 2, 1, 3);
    struct ggml_tensor* q_cont = ggml_cont(ctx, q_perm);
    
    // K_perm: [head_dim, seq_len_kv, n_heads_kv, batch]
    struct ggml_tensor* k_perm = ggml_permute(ctx, k, 0, 2, 1, 3);
    struct ggml_tensor* k_cont = ggml_cont(ctx, k_perm);
    
    // Compute Q K^T: [seq_len_q, seq_len_kv, n_heads, batch]
    // ggml_mul_mat takes A [K, M] and B [K, N], yields C [N, M]
    // So k_cont [head_dim, seq_len_kv] (A) and q_cont [head_dim, seq_len_q] (B)
    // yields kq [seq_len_q, seq_len_kv, n_heads_q, batch]
    struct ggml_tensor* kq = ggml_mul_mat(ctx, k_cont, q_cont);
    
    // Scale scores
    kq = ggml_scale(ctx, kq, scale);
    
    // Transpose to [seq_len_kv, seq_len_q, n_heads_q, batch] so that seq_len_kv is ne[0] (innermost)
    // for correct softmax calculation and bias addition.
    kq = ggml_transpose(ctx, kq);
    kq = ggml_cont(ctx, kq);

    // Add bias / mask (shape: [seq_len_kv, seq_len_q, n_heads_q, batch])
    if (bias) {
        kq = ggml_add(ctx, kq, bias);
    }
    
    // Softmax along ne[0] (seq_len_kv)
    struct ggml_tensor* kq_soft = ggml_soft_max(ctx, kq);
    
    // Write out weights if requested
    if (attn_w) {
        kq_soft = ggml_cpy(ctx, kq_soft, attn_w);
    }
    
    // V_perm: [seq_len_kv, head_dim, n_heads_kv, batch]
    struct ggml_tensor* v_perm = ggml_permute(ctx, v, 1, 2, 0, 3);
    struct ggml_tensor* v_cont = ggml_cont(ctx, v_perm);
    
    // Multiply by V: [seq_len_q, head_dim, n_heads, batch]
    struct ggml_tensor* kqv = ggml_mul_mat(ctx, v_cont, kq_soft);
    
    // Permute back to [head_dim, n_heads, seq_len_q, batch]
    kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    struct ggml_tensor* out = ggml_cont(ctx, kqv);
    
    return out;
}
