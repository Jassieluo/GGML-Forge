#include "ops/ops.h"
#include "ggml-impl.h"
#include <cstring>

namespace {

// Optional srcs can only be omitted from the tail (contract). Appends the
// optional tensors in order and rejects a present tensor after an absent one
// (e.g. h0 without biases — bind zero biases instead).
int collect_optional_srcs(struct ggml_tensor** srcs, int base,
                          struct ggml_tensor* const* optionals, int count) {
    int n = base;
    bool stopped = false;
    for (int i = 0; i < count; ++i) {
        if (!optionals[i]) {
            stopped = true;
            continue;
        }
        if (stopped) return -1;
        srcs[n++] = optionals[i];
    }
    return n;
}

struct ggml_tensor* build_recurrent(
    struct ggml_context* ctx, ggml_ops_ext::ops_virt_op_type op,
    struct ggml_tensor* x, struct ggml_tensor* weight_ih, struct ggml_tensor* weight_hh,
    struct ggml_tensor* const* optionals, int optional_count, bool reverse,
    ggml_backend_t backend
) {
    if (!x || !weight_ih || !weight_hh) return nullptr;
    ggml_ops_ext::ops_recurrent_params params;
    params.reverse = reverse ? 1 : 0;

    struct ggml_tensor* srcs[7] = { x, weight_ih, weight_hh };
    const int n_srcs = collect_optional_srcs(srcs, 3, optionals, optional_count);
    if (n_srcs < 0) return nullptr;

    if (!ggml_ops_backend_supports_op(backend, op, srcs, n_srcs, &params, sizeof(params))) {
        // Kernel-required: unrolling the recurrence into native nodes would
        // create thousands of graph nodes — the exact cost this op removes.
        return nullptr;
    }

    const int64_t hidden = weight_hh->ne[0];
    const int64_t ne[2] = { hidden, x->ne[1] };
    struct ggml_tensor* result = ggml_ops_ext::ops_new_virtual_node(
        ctx, op, GGML_TYPE_F32, 2, ne, n_srcs, srcs);
    ggml_set_op_params(result, &params, sizeof(params));
    return result;
}

} // namespace

struct ggml_tensor* ggml_ops_gru(
    struct ggml_context* ctx, struct ggml_tensor* x,
    struct ggml_tensor* weight_ih, struct ggml_tensor* weight_hh,
    struct ggml_tensor* bias_ih, struct ggml_tensor* bias_hh,
    struct ggml_tensor* h0, bool reverse, ggml_backend_t backend
) {
    struct ggml_tensor* optionals[] = { bias_ih, bias_hh, h0 };
    return build_recurrent(ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_GRU, x, weight_ih, weight_hh,
                           optionals, 3, reverse, backend);
}

struct ggml_tensor* ggml_ops_lstm(
    struct ggml_context* ctx, struct ggml_tensor* x,
    struct ggml_tensor* weight_ih, struct ggml_tensor* weight_hh,
    struct ggml_tensor* bias_ih, struct ggml_tensor* bias_hh,
    struct ggml_tensor* h0, struct ggml_tensor* c0, bool reverse, ggml_backend_t backend
) {
    struct ggml_tensor* optionals[] = { bias_ih, bias_hh, h0, c0 };
    return build_recurrent(ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_LSTM, x, weight_ih, weight_hh,
                           optionals, 4, reverse, backend);
}
