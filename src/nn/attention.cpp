#include "nn/nn.h"

namespace nn {

// MultiHeadAttention
MultiHeadAttention::MultiHeadAttention(
    struct ggml_tensor* qw, struct ggml_tensor* qb,
    struct ggml_tensor* kw, struct ggml_tensor* kb,
    struct ggml_tensor* vw, struct ggml_tensor* vb,
    struct ggml_tensor* ow, struct ggml_tensor* ob,
    int n_heads, int head_dim
) : q_proj(qw, qb), k_proj(kw, kb), v_proj(vw, vb), out_proj(ow, ob),
    n_heads(n_heads), head_dim(head_dim) {}

struct ggml_tensor* MultiHeadAttention::forward(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* mask,
    ggml_backend_t backend
) {
    struct ggml_tensor* q = q_proj.forward(ctx, x);
    struct ggml_tensor* k = k_proj.forward(ctx, x);
    struct ggml_tensor* v = v_proj.forward(ctx, x);

    q = ggml_cont(ctx, ggml_reshape_3d(ctx, q, head_dim, n_heads, q->ne[1]));
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    q = ggml_reshape_4d(ctx, q, head_dim, q->ne[1], n_heads, 1);

    k = ggml_cont(ctx, ggml_reshape_3d(ctx, k, head_dim, n_heads, k->ne[1]));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    k = ggml_reshape_4d(ctx, k, head_dim, k->ne[1], n_heads, 1);

    v = ggml_cont(ctx, ggml_reshape_3d(ctx, v, head_dim, n_heads, v->ne[1]));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
    v = ggml_reshape_4d(ctx, v, head_dim, v->ne[1], n_heads, 1);

    float scale = 1.0f / std::sqrt((float)head_dim);
    struct ggml_tensor* attn_out = ggml_ops_attention(ctx, q, k, v, mask, nullptr, scale, -1, backend);

    attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));
    attn_out = ggml_reshape_2d(ctx, attn_out, n_heads * head_dim, attn_out->ne[2]);

    return out_proj.forward(ctx, attn_out);
}

// KVHeadAttention
KVHeadAttention::KVHeadAttention(
    struct ggml_tensor* qw, struct ggml_tensor* qb,
    struct ggml_tensor* kw, struct ggml_tensor* kb,
    struct ggml_tensor* vw, struct ggml_tensor* vb,
    struct ggml_tensor* ow, struct ggml_tensor* ob,
    int n_heads, int head_dim, int layer_idx
) : q_proj(qw, qb), k_proj(kw, kb), v_proj(vw, vb), out_proj(ow, ob),
    n_heads(n_heads), head_dim(head_dim), layer_idx(layer_idx) {}

struct ggml_tensor* KVHeadAttention::forward(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* kv_k,
    struct ggml_tensor* kv_v,
    int q_len,
    int total_len,
    struct ggml_tensor* mask,
    struct ggml_cgraph* cgraph,
    ggml_backend_t backend
) {
    struct ggml_tensor* Q = q_proj.forward(ctx, x);
    struct ggml_tensor* K = k_proj.forward(ctx, x);
    struct ggml_tensor* V = v_proj.forward(ctx, x);

    Q = ggml_reshape_3d(ctx, Q, head_dim, n_heads, q_len);
    K = ggml_reshape_3d(ctx, K, head_dim, n_heads, q_len);
    V = ggml_reshape_3d(ctx, V, head_dim, n_heads, q_len);

    // Permute K and V to [head_dim, q_len, n_heads]
    struct ggml_tensor* K_perm = ggml_permute(ctx, K, 0, 2, 1, 3);
    struct ggml_tensor* K_cont = ggml_cont(ctx, K_perm);
    struct ggml_tensor* V_perm = ggml_permute(ctx, V, 0, 2, 1, 3);
    struct ggml_tensor* V_cont = ggml_cont(ctx, V_perm);

    struct ggml_tensor* K_dest = nullptr;
    struct ggml_tensor* V_dest = nullptr;

    if (q_len > 1) { // First step (prompt phase)
        int64_t offset_bytes = layer_idx * kv_k->nb[3];
        K_dest = ggml_view_3d(ctx, kv_k, head_dim, total_len, n_heads,
            kv_k->nb[1], kv_k->nb[2], offset_bytes);
        V_dest = ggml_view_3d(ctx, kv_v, head_dim, total_len, n_heads,
            kv_v->nb[1], kv_v->nb[2], offset_bytes);
    } else { // Auto-regressive decoding step
        int pos_idx = total_len - 1;
        int64_t offset_bytes = layer_idx * kv_k->nb[3] + pos_idx * kv_k->nb[1];
        K_dest = ggml_view_3d(ctx, kv_k, head_dim, 1, n_heads,
            kv_k->nb[1], kv_k->nb[2], offset_bytes);
        V_dest = ggml_view_3d(ctx, kv_v, head_dim, 1, n_heads,
            kv_v->nb[1], kv_v->nb[2], offset_bytes);
    }

    struct ggml_tensor* K_cpy = ggml_cpy(ctx, K_cont, K_dest);
    struct ggml_tensor* V_cpy = ggml_cpy(ctx, V_cont, V_dest);
    if (cgraph) {
        ggml_build_forward_expand(cgraph, K_cpy);
        ggml_build_forward_expand(cgraph, V_cpy);
    }

    // Active views from KV Cache
    struct ggml_tensor* K_cached = ggml_view_3d(ctx, kv_k, head_dim, total_len, n_heads,
        kv_k->nb[1], kv_k->nb[2], layer_idx * kv_k->nb[3]);
    struct ggml_tensor* V_cached = ggml_view_3d(ctx, kv_v, head_dim, total_len, n_heads,
        kv_v->nb[1], kv_v->nb[2], layer_idx * kv_v->nb[3]);

    struct ggml_tensor* Q_perm = ggml_permute(ctx, Q, 0, 2, 1, 3);
    struct ggml_tensor* K_cached_perm = K_cached;
    struct ggml_tensor* V_cached_perm = ggml_permute(ctx, V_cached, 1, 0, 2, 3);

    struct ggml_tensor* Q_cont = ggml_cont(ctx, Q_perm);
    struct ggml_tensor* K_cont_cached = ggml_cont(ctx, K_cached_perm);

    // Perform attention matrix multiplication
    struct ggml_tensor* r = ggml_mul_mat(ctx, Q_cont, K_cont_cached);
    ggml_mul_mat_set_prec(r, GGML_PREC_DEFAULT);

    struct ggml_tensor* kq = ggml_transpose(ctx, r);
    kq = ggml_cont(ctx, kq);
    struct ggml_tensor* kq_scaled = ggml_scale(ctx, kq, 1.0f / std::sqrt((float)head_dim));
    struct ggml_tensor* kq_masked = mask ? ggml_add(ctx, kq_scaled, mask) : kq_scaled;
    struct ggml_tensor* kq_soft = ggml_soft_max(ctx, kq_masked);

    struct ggml_tensor* V_cont_cached = ggml_cont(ctx, V_cached_perm);
    struct ggml_tensor* kqv = ggml_mul_mat(ctx, V_cont_cached, kq_soft);
    kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    kqv = ggml_cont(ctx, kqv);
    kqv = ggml_reshape_2d(ctx, kqv, n_heads * head_dim, q_len);

    return out_proj.forward(ctx, kqv);
}

} // namespace nn
