#include "ops/ops.h"
#include "ggml-impl.h"
#include <algorithm>

struct ggml_tensor* ggml_ops_relative_pe_keys(
    struct ggml_context* ctx,
    struct ggml_tensor* q,
    struct ggml_tensor* emb_rel_k,
    float scale,
    int32_t window_size,
    ggml_backend_t backend
) {
    // 1. Check if backend registers a custom builder
    ggml_ops_ext::ops_op_builder_t builder = ggml_ops_ext::find_ops_builder(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_RELATIVE_PE_KEYS);
    if (builder) {
        struct ggml_tensor* srcs[] = { q, emb_rel_k };
        union {
            float f;
            int32_t i;
        } u_scale;
        u_scale.f = scale;
        int32_t params[] = { u_scale.i, window_size };
        return builder(ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_RELATIVE_PE_KEYS, srcs, 2, params, 2, backend);
    }

    // 2. Check if backend registers virtual node support
    if (ggml_ops_backend_supports_op(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_RELATIVE_PE_KEYS)) {
        struct ggml_tensor* srcs[] = { q, emb_rel_k };
        // Output scores shape: [T, T, n_head]
        int64_t T = q->ne[1];
        int64_t n_head = q->ne[2];
        int64_t ne[GGML_MAX_DIMS] = { T, T, n_head, 1 };
        struct ggml_tensor* result = ggml_ops_ext::ops_new_virtual_node(
            ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_RELATIVE_PE_KEYS, q->type, 3, ne, 2, srcs);

        union {
            float f;
            int32_t i;
        } u_scale;
        u_scale.f = scale;
        int32_t params[] = { u_scale.i, window_size };
        ggml_set_op_params(result, params, sizeof(params));
        return result;
    }

    // 3. Fallback: Coordinate Shift logic for Keys
    int64_t d_k = q->ne[0];
    int64_t T = q->ne[1];
    int64_t n_head = q->ne[2];
    int limit = window_size + 1;

    int pad_length = std::max((int)T - limit, 0);
    struct ggml_tensor* padded_emb_k = emb_rel_k;
    if (pad_length > 0) {
        struct ggml_tensor* zeros_pad = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_k, pad_length, emb_rel_k->ne[2]);
        zeros_pad = ggml_fill(ctx, zeros_pad, 0.0f);
        struct ggml_tensor* temp = ggml_concat(ctx, zeros_pad, emb_rel_k, 1);
        padded_emb_k = ggml_concat(ctx, temp, zeros_pad, 1);
    }

    int slice_start = std::max(limit - (int)T, 0);
    int slice_len = 2 * T - 1;
    size_t offset_k = slice_start * padded_emb_k->nb[1];
    struct ggml_tensor* rel_emb_k = ggml_view_3d(ctx, padded_emb_k, d_k, slice_len, emb_rel_k->ne[2], padded_emb_k->nb[1], padded_emb_k->nb[2], offset_k);

    struct ggml_tensor* q_scaled = ggml_scale(ctx, q, scale);
    struct ggml_tensor* rel_emb_k_f32 = (rel_emb_k->type == GGML_TYPE_F32) ? rel_emb_k : ggml_cast(ctx, rel_emb_k, GGML_TYPE_F32);
    struct ggml_tensor* q_scaled_f32 = (q_scaled->type == GGML_TYPE_F32) ? q_scaled : ggml_cast(ctx, q_scaled, GGML_TYPE_F32);
    struct ggml_tensor* rel_logits = ggml_mul_mat(ctx, rel_emb_k_f32, q_scaled_f32);

    struct ggml_tensor* zeros_col = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, T, n_head);
    zeros_col = ggml_fill(ctx, zeros_col, 0.0f);
    struct ggml_tensor* x_padded = ggml_concat(ctx, rel_logits, zeros_col, 0); // [2*T, T, n_head]

    struct ggml_tensor* x_flat = ggml_reshape_2d(ctx, x_padded, T * 2 * T, n_head);

    struct ggml_tensor* zeros_flat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T - 1, n_head);
    zeros_flat = ggml_fill(ctx, zeros_flat, 0.0f);
    struct ggml_tensor* x_flat_padded = ggml_concat(ctx, x_flat, zeros_flat, 0);

    struct ggml_tensor* x_final = ggml_reshape_3d(ctx, x_flat_padded, 2 * T - 1, T + 1, n_head);

    size_t view_offset = (T - 1) * sizeof(float);
    struct ggml_tensor* scores_local = ggml_view_3d(ctx, x_final, T, T, n_head, x_final->nb[1], x_final->nb[2], view_offset);
    return ggml_cont(ctx, scores_local);
}
