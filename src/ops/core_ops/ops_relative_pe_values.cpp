#include "ops/ops.h"
#include "ggml-impl.h"
#include <algorithm>

struct ggml_tensor* ggml_ops_relative_pe_values(
    struct ggml_context* ctx,
    struct ggml_tensor* attn_w,
    struct ggml_tensor* emb_rel_v,
    int32_t window_size,
    ggml_backend_t backend
) {
    // 1. Check if backend registers a custom builder
    ggml_ops_ext::ops_op_builder_t builder = ggml_ops_ext::find_ops_builder(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_RELATIVE_PE_VALUES);
    if (builder) {
        struct ggml_tensor* srcs[] = { attn_w, emb_rel_v };
        int32_t params[] = { window_size };
        return builder(ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_RELATIVE_PE_VALUES, srcs, 2, params, 1, backend);
    }

    // 2. Check if backend registers virtual node support
    if (ggml_ops_backend_supports_op(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_RELATIVE_PE_VALUES)) {
        struct ggml_tensor* srcs[] = { attn_w, emb_rel_v };
        // Output bias shape: [d_k * n_head, T]
        int64_t T = attn_w->ne[1];
        int64_t d_k = emb_rel_v->ne[0];
        int64_t n_head = attn_w->ne[2];
        int64_t ne[GGML_MAX_DIMS] = { d_k * n_head, T, 1, 1 };
        struct ggml_tensor* result = ggml_ops_ext::ops_new_virtual_node(
            ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_RELATIVE_PE_VALUES, attn_w->type, 2, ne, 2, srcs);

        int32_t params[] = { window_size };
        ggml_set_op_params(result, params, sizeof(params));
        return result;
    }

    // 3. Fallback: Coordinate Shift logic for Values
    struct ggml_tensor* attn_w_f32 = attn_w->type == GGML_TYPE_F32 ? attn_w : ggml_cast(ctx, attn_w, GGML_TYPE_F32);
    struct ggml_tensor* emb_rel_v_f32 = emb_rel_v->type == GGML_TYPE_F32 ? emb_rel_v : ggml_cast(ctx, emb_rel_v, GGML_TYPE_F32);

    int64_t T = attn_w_f32->ne[1];
    int64_t d_k = emb_rel_v_f32->ne[0];
    int64_t n_head = attn_w_f32->ne[2];
    int limit = window_size + 1;

    int pad_length = std::max((int)T - limit, 0);
    struct ggml_tensor* padded_emb_v = emb_rel_v_f32;
    if (pad_length > 0) {
        struct ggml_tensor* zeros_pad = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_k, pad_length, emb_rel_v_f32->ne[2]);
        zeros_pad = ggml_fill(ctx, zeros_pad, 0.0f);
        struct ggml_tensor* temp = ggml_concat(ctx, zeros_pad, emb_rel_v_f32, 1);
        padded_emb_v = ggml_concat(ctx, temp, zeros_pad, 1);
    }

    int slice_start = std::max(limit - (int)T, 0);
    int slice_len = 2 * T - 1;
    size_t offset_v = slice_start * padded_emb_v->nb[1];
    struct ggml_tensor* rel_emb_v = ggml_view_3d(ctx, padded_emb_v, d_k, slice_len, emb_rel_v_f32->ne[2], padded_emb_v->nb[1], padded_emb_v->nb[2], offset_v);

    struct ggml_tensor* zeros_cols = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T - 1, T, n_head);
    zeros_cols = ggml_fill(ctx, zeros_cols, 0.0f);
    struct ggml_tensor* x_padded = ggml_concat(ctx, attn_w_f32, zeros_cols, 0); // [2*T-1, T, n_head]

    struct ggml_tensor* x_flat = ggml_reshape_2d(ctx, x_padded, T * (2 * T - 1), n_head);

    struct ggml_tensor* zeros_beg = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, n_head);
    zeros_beg = ggml_fill(ctx, zeros_beg, 0.0f);
    struct ggml_tensor* x_flat_padded = ggml_concat(ctx, zeros_beg, x_flat, 0);

    struct ggml_tensor* x_final = ggml_reshape_3d(ctx, x_flat_padded, 2 * T, T, n_head);

    size_t offset_w = 1 * sizeof(float);
    struct ggml_tensor* rel_weights = ggml_cont(ctx, ggml_view_3d(ctx, x_final, 2 * T - 1, T, n_head, x_final->nb[1], x_final->nb[2], offset_w));

    struct ggml_tensor* rel_emb_v_t = ggml_cont(ctx, ggml_transpose(ctx, rel_emb_v));

    struct ggml_tensor* dummy = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, slice_len, d_k, n_head);
    struct ggml_tensor* rel_emb_v_t_repeated = ggml_cont(ctx, ggml_repeat(ctx, rel_emb_v_t, dummy));
    struct ggml_tensor* rel_out_bias = ggml_mul_mat(ctx, rel_weights, rel_emb_v_t_repeated);

    rel_out_bias = ggml_cont(ctx, ggml_permute(ctx, rel_out_bias, 2, 0, 1, 3));
    struct ggml_tensor* rel_out_bias_flat = ggml_reshape_2d(ctx, rel_out_bias, d_k * n_head, T);

    struct ggml_tensor* res = rel_out_bias_flat;
    if (res->type != attn_w->type) {
        res = ggml_cast(ctx, res, attn_w->type);
    }
    return res;
}
