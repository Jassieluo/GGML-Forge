#include "nn/nn.h"

namespace nn {

struct ggml_tensor* functional::attention(
    struct ggml_context* ctx,
    struct ggml_tensor* q,
    struct ggml_tensor* k,
    struct ggml_tensor* v,
    struct ggml_tensor* mask,
    struct ggml_tensor* weights,
    float scale,
    int sliding_window,
    ggml_backend_t backend
) {
    return ggml_ops_attention(ctx, q, k, v, mask, weights, scale, sliding_window, backend);
}

struct ggml_tensor* functional::relative_position_keys(
    struct ggml_context* ctx,
    struct ggml_tensor* q,
    struct ggml_tensor* embedding,
    float scale,
    int window,
    ggml_backend_t backend
) {
    return ggml_ops_relative_pe_keys(ctx, q, embedding, scale, window, backend);
}

struct ggml_tensor* functional::relative_position_values(
    struct ggml_context* ctx,
    struct ggml_tensor* weights,
    struct ggml_tensor* embedding,
    struct ggml_tensor* attention_output,
    int window,
    ggml_backend_t backend
) {
    return ggml_ops_relative_pe_values(
        ctx, weights, embedding, attention_output, window, backend);
}

struct ggml_tensor* MultiHeadAttention::forward(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* mask,
    ggml_backend_t backend,
    struct ggml_tensor* pos_tensor
) {
    ggml_backend_t b = backend ? backend : this->backend;
    struct ggml_tensor* q = q_proj.forward(ctx, x);
    struct ggml_tensor* k = k_proj.forward(ctx, x);
    struct ggml_tensor* v = v_proj.forward(ctx, x);

    int64_t T = x->ne[1];

    if (pos_tensor != nullptr) {
        auto apply_python_rope = [&](struct ggml_tensor* tensor) {
            const size_t element_size = ggml_element_size(tensor);
            struct ggml_tensor* first_head = ggml_view_3d(
                ctx, tensor, head_dim, 1, T,
                head_dim * element_size, tensor->nb[1], 0);
            first_head = ggml_rope_ext(
                ctx, first_head, pos_tensor, nullptr, head_dim, GGML_ROPE_TYPE_NORMAL,
                32768, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            first_head = ggml_reshape_2d(ctx, first_head, head_dim, T);

            struct ggml_tensor* remaining = ggml_view_2d(
                ctx, tensor, head_dim * (n_heads - 1), T,
                tensor->nb[1], head_dim * element_size);
            return ggml_cont(ctx, ggml_concat(ctx, first_head, remaining, 0));
        };

        q = apply_python_rope(q);
        k = apply_python_rope(k);
    }

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
    struct ggml_tensor* attn_out = ggml_ops_attention(ctx, q, k, v, mask, nullptr, scale, -1, b);

    attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));
    attn_out = ggml_reshape_2d(ctx, attn_out, n_heads * head_dim, attn_out->ne[2]);

    return out_proj.forward(ctx, attn_out);
}

// KVHeadAttention

struct ggml_tensor* KVHeadAttention::decode(
    Context& context,
    struct ggml_tensor* x,
    KVCache& cache,
    struct ggml_tensor* position,
    struct ggml_tensor* valid_length,
    ggml_backend_t backend
) {
    ggml_context* ctx = context.native_handle();
    ggml_backend_t b = backend ? backend : this->backend;
    ggml_tensor* Q = q_proj.forward(ctx, x);
    ggml_tensor* K = k_proj.forward(ctx, x);
    ggml_tensor* V = v_proj.forward(ctx, x);
    Q = ggml_permute(ctx, ggml_reshape_3d(ctx, Q, head_dim, n_heads, 1), 0, 2, 1, 3);
    K = ggml_permute(ctx, ggml_reshape_3d(ctx, K, head_dim, n_heads, 1), 0, 2, 1, 3);
    V = ggml_permute(ctx, ggml_reshape_3d(ctx, V, head_dim, n_heads, 1), 0, 2, 1, 3);
    Q = ggml_cont(ctx, Q);
    if (Q->type != GGML_TYPE_F32) Q = ggml_cont(ctx, ggml_cast(ctx, Q, GGML_TYPE_F32));
    ggml_tensor* attended = cache.decode_attention(
        context, layer_idx, Q, K, V, position, valid_length,
        1.0f / std::sqrt(static_cast<float>(head_dim)), b);
    attended = ggml_cont(ctx, ggml_permute(ctx, attended, 0, 2, 1, 3));
    attended = ggml_reshape_2d(ctx, attended, n_heads * head_dim, 1);
    return out_proj.forward(ctx, attended);
}

struct ggml_tensor* KVHeadAttention::prefill(
    Context& context,
    struct ggml_tensor* x,
    KVCache& cache,
    struct ggml_tensor* mask,
    struct ggml_cgraph* cgraph,
    ggml_backend_t backend
) {
    ggml_context* ctx = context.native_handle();
    ggml_backend_t b = backend ? backend : this->backend;
    const int q_len = static_cast<int>(x->ne[1]);
    struct ggml_tensor* Q = q_proj.forward(ctx, x);
    struct ggml_tensor* K = k_proj.forward(ctx, x);
    struct ggml_tensor* V = v_proj.forward(ctx, x);

    Q = ggml_reshape_3d(ctx, Q, head_dim, n_heads, q_len);
    K = ggml_reshape_3d(ctx, K, head_dim, n_heads, q_len);
    V = ggml_reshape_3d(ctx, V, head_dim, n_heads, q_len);

    // Permute K and V to [head_dim, q_len, n_heads]
    struct ggml_tensor* K_perm = ggml_permute(ctx, K, 0, 2, 1, 3);
    struct ggml_tensor* V_perm = ggml_permute(ctx, V, 0, 2, 1, 3);

    struct ggml_tensor* Q_perm = ggml_permute(ctx, Q, 0, 2, 1, 3);
    struct ggml_tensor* Q_cont = ggml_cont(ctx, Q_perm);
    if (Q_cont->type != GGML_TYPE_F32) {
        Q_cont = ggml_cont(ctx, ggml_cast(ctx, Q_cont, GGML_TYPE_F32));
    }
    struct ggml_tensor* kqv = cache.prefill_attention(
        context, layer_idx, Q_cont, K_perm, V_perm,
        1.0f / std::sqrt((float)head_dim), b, mask, cgraph);
    kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    kqv = ggml_cont(ctx, kqv);
    kqv = ggml_reshape_2d(ctx, kqv, n_heads * head_dim, q_len);

    struct ggml_tensor* attn_out = out_proj.forward(ctx, kqv);

    return attn_out;
}

} // namespace nn
