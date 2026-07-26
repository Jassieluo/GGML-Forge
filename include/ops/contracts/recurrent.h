#pragma once

#include "ops/types.h"
#include <cstdint>
#include <cstring>

namespace ggml_ops_ext {

// ---- Sequence-level fused recurrent cells -----------------------------------
// One node consumes the whole sequence; kernels loop timesteps internally so
// a 500-step GRU is one graph node, not thousands of unrolled ones. Gate
// order, formulas, and weight layout follow torch.nn.GRU / torch.nn.LSTM
// exactly, so converted PyTorch weights bind without reshuffling:
//
// GRU  (gates r, z, n; G = 3):
//   r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
//   z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
//   n = tanh   (W_in x + b_in + r * (W_hn h + b_hn))
//   h' = (1 - z) * n + z * h
// LSTM (gates i, f, g, o; G = 4):
//   i, f, o = sigmoid(...), g = tanh(...)
//   c' = f * c + i * g;  h' = o * tanh(c')
//
// srcs (all F32, contiguous):
//   [0] x         [input_dim, steps]      single sequence (batch = 1 for now)
//   [1] weight_ih [input_dim, G * hidden] row g*hidden+j maps input -> gate g
//   [2] weight_hh [hidden, G * hidden]
//   [3] bias_ih   [G * hidden]            (optional; pass with bias_hh or not at all)
//   [4] bias_hh   [G * hidden]
//   [5] h0        [hidden]                (optional initial state, zeros if absent)
//   [6] c0        [hidden]                (LSTM only, optional, zeros if absent)
// Optional srcs may only be omitted from the tail.
// Output: [hidden, steps] — every hidden state; the final state is the last
// column. LSTM cell state is internal (re-run with saved h/c columns for
// streaming once that lands).
//
// op_params: { int32 reverse } — process timesteps last-to-first while
// writing outputs at their original positions; a bidirectional layer is the
// forward op plus a reverse op concatenated on dim 0.

struct ops_recurrent_params {
    int32_t reverse = 0;
};
static_assert(sizeof(ops_recurrent_params) == sizeof(int32_t));

inline bool ops_validate_recurrent(const ops_request& request, bool lstm) {
    const int gates = lstm ? 4 : 3;
    if (!request.srcs || request.n_srcs < 3 || !request.params ||
        request.params_size < sizeof(ops_recurrent_params)) return false;
    const int max_srcs = lstm ? 7 : 6;
    // Execute-time probes pass the node's fixed-size src array with trailing
    // nulls: count the leading non-null run, then require nothing after it.
    int n_srcs = 0;
    while (n_srcs < request.n_srcs && n_srcs < max_srcs && request.srcs[n_srcs]) ++n_srcs;
    for (int i = n_srcs; i < request.n_srcs; ++i) {
        if (request.srcs[i]) return false;
    }
    if (n_srcs < 3) return false;
    const ggml_tensor* x = request.srcs[0];
    const ggml_tensor* w_ih = request.srcs[1];
    const ggml_tensor* w_hh = request.srcs[2];
    for (int i = 0; i < n_srcs; ++i) {
        const ggml_tensor* t = request.srcs[i];
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) return false;
    }
    const int64_t input_dim = x->ne[0];
    const int64_t steps = x->ne[1];
    if (input_dim < 1 || steps < 1 || x->ne[2] != 1 || x->ne[3] != 1) return false;
    const int64_t hidden = w_hh->ne[0];
    if (hidden < 1 || w_hh->ne[1] != gates * hidden) return false;
    if (w_ih->ne[0] != input_dim || w_ih->ne[1] != gates * hidden) return false;
    if (n_srcs > 3 &&
        (n_srcs < 5 || ggml_nelements(request.srcs[3]) != gates * hidden)) return false;
    if (n_srcs > 4 && ggml_nelements(request.srcs[4]) != gates * hidden) return false;
    if (n_srcs > 5 && ggml_nelements(request.srcs[5]) != hidden) return false;
    if (n_srcs > 6 && ggml_nelements(request.srcs[6]) != hidden) return false;
    if (!request.output) return true;
    return request.output->type == GGML_TYPE_F32 &&
           request.output->ne[0] == hidden && request.output->ne[1] == steps &&
           request.output->ne[2] == 1 && request.output->ne[3] == 1;
}

inline bool ops_validate_gru(const ops_request& request) {
    return ops_validate_recurrent(request, false);
}

inline bool ops_validate_lstm(const ops_request& request) {
    return ops_validate_recurrent(request, true);
}

} // namespace ggml_ops_ext
