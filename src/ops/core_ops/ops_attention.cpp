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
    ggml_backend_t backend,
    struct ggml_tensor* valid_length,
    struct ggml_tensor* dependency
) {
    const bool compressed_cache = k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32;
    const bool quantized_cache = k->type == GGML_TYPE_Q4_0 || k->type == GGML_TYPE_Q8_0 ||
                                 v->type == GGML_TYPE_Q4_0 || v->type == GGML_TYPE_Q8_0;
    const char* backend_name = backend ? ggml_backend_name(backend) : nullptr;
    const bool is_sycl = backend_name && std::strstr(backend_name, "SYCL") != nullptr;
    const bool is_cuda = backend_name && std::strstr(backend_name, "CUDA") != nullptr;
    const bool gpu_f16_cache = (is_cuda || is_sycl) && compressed_cache && !quantized_cache;
    const bool sycl_quantized_prefill = is_sycl && quantized_cache && q->ne[1] > 8;
    if (backend && !attn_w && !valid_length && q->type == GGML_TYPE_F32 &&
        (gpu_f16_cache || sycl_quantized_prefill)) {
        struct ggml_tensor* k_f32 = ggml_cont(ctx, ggml_cast(ctx, k, GGML_TYPE_F32));
        struct ggml_tensor* v_f32 = ggml_cont(ctx, ggml_cast(ctx, v, GGML_TYPE_F32));
        return ggml_ops_attention(ctx, q, k_f32, v_f32, bias, nullptr, scale, window_size, backend);
    }
    if (backend && !attn_w && is_cuda && q->type == GGML_TYPE_F32 &&
        k->type == GGML_TYPE_F32 && v->type == GGML_TYPE_F32 && q->ne[1] > 8) {
        return ggml_ops_attention(ctx, q, k, v, bias, nullptr, scale, window_size, nullptr);
    }

    struct ggml_tensor* srcs[] = { q, k, v, bias, attn_w, valid_length, dependency };
    union { float f; int32_t i; } u_scale;
    u_scale.f = scale;
    int32_t params[] = { u_scale.i, window_size };
    // 2. Otherwise check if it supports direct handler execution (virtual node)
    if (ggml_ops_backend_supports_op(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_FUSED_ATTN,
                                     srcs, 7, params, sizeof(params))) {
        struct ggml_tensor* result = ggml_ops_ext::ops_new_virtual_node(
            ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_FUSED_ATTN, q->type, ggml_n_dims(q), q->ne, 7, srcs);
        
        ggml_set_op_params(result, params, sizeof(params));
        return result;
    }

    // Runtime-length attention requires a backend kernel because a fallback
    // graph would bake the active cache length into tensor views.
    if (valid_length || dependency) return nullptr;

    // 3. Fallback: construct standard GGUF/GGML subgraph
    // This allows complete functionality on any backend without custom handler registration.
    
    // Q: [head_dim, n_heads_q, seq_len_q, batch]
    // K: [head_dim, n_heads_kv, seq_len_kv, batch]
    // V: [head_dim, n_heads_kv, seq_len_kv, batch]
    
    // Q, K, V are expected in layout: [head_dim, seq_len, n_heads, batch]
    struct ggml_tensor* q_cont = ggml_cont(ctx, q);
    struct ggml_tensor* k_cont = ggml_cont(ctx, k);
    
    // Compute Q K^T: [seq_len_q, seq_len_kv, n_heads, batch]
    // ggml_mul_mat takes A [K, M] and B [K, N], yields C [N, M]
    // So k_cont [head_dim, seq_len_kv] (A) and q_cont [head_dim, seq_len_q] (B)
    // yields kq [seq_len_q, seq_len_kv, n_heads_q, batch]
    struct ggml_tensor* kq = ggml_mul_mat(ctx, k_cont, q_cont);
    
    // Scale scores
    kq = ggml_scale(ctx, kq, scale);
    
    // ggml_mul_mat already returns [seq_len_kv, seq_len_q, heads, batch].
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
    // V is [head_dim, seq_len_kv, n_heads_kv, batch], permuting 0 and 1 yields [seq_len_kv, head_dim, n_heads_kv, batch]
    struct ggml_tensor* v_perm = ggml_permute(ctx, v, 1, 0, 2, 3);
    struct ggml_tensor* v_cont = ggml_cont(ctx, v_perm);
    
    // Multiply by V: [head_dim, seq_len_q, n_heads, batch]
    struct ggml_tensor* kqv = ggml_mul_mat(ctx, v_cont, kq_soft);
    
    struct ggml_tensor* out = ggml_cont(ctx, kqv);
    
    return out;
}
