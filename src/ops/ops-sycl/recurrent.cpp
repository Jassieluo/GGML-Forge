#include "common.hpp"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "ops/ops.h"
#include "ops_sycl.h"

#include <cstdio>
#include <cstring>

namespace ggml_ops_ext {
namespace sycl {

class RecurrentProjectInputsSYCLKernel;
class GruRecurrenceSYCLKernel;
class LstmRecurrenceSYCLKernel;

// Sequence-level fused GRU / LSTM (torch.nn gate order; see
// contracts/recurrent.h). Mirrors the CUDA kernels: phase 1 projects the
// whole sequence through weight_ih into a pool-allocated scratch buffer
// (embarrassingly parallel), phase 2 is a single work-group that runs the
// sequential recurrence with h (and c for LSTM) plus the per-step gate
// scratch in local memory, barrier between steps. The in-order queue
// serializes the two phases. Probe caps hidden at
// ops_sycl_recurrent_max_hidden so the local arrays fit in SLM.
// No fp64 anywhere: the target iGPU has no double support.

static inline float recurrent_sigmoidf(float value) {
    return 1.0f / (1.0f + ::sycl::exp(-value));
}

// gates_x[t * gate_rows + row] = b_ih[row] + sum_i w_ih[row * in + i] * x[t * in + i]
static void launch_project_inputs(::sycl::queue* queue, const float* x, const float* w_ih,
                                  const float* b_ih, float* gates_x, int64_t input_dim,
                                  int64_t steps, int64_t gate_rows) {
    const size_t count = static_cast<size_t>(steps * gate_rows);
    queue->submit([&](::sycl::handler& handler) {
        handler.parallel_for<RecurrentProjectInputsSYCLKernel>(
            ::sycl::range<1>(count), [=](::sycl::id<1> id) {
                const int64_t idx = static_cast<int64_t>(id[0]);
                const int64_t t = idx / gate_rows;
                const int64_t row = idx % gate_rows;
                const float* x_t = x + t * input_dim;
                const float* w_row = w_ih + row * input_dim;
                float sum = b_ih ? b_ih[row] : 0.0f;
                for (int64_t i = 0; i < input_dim; ++i) sum += w_row[i] * x_t[i];
                gates_x[idx] = sum;
            });
    });
}

// Single work-group; local memory = h[hidden] + rh[3 * hidden].
static void launch_gru_recurrence(::sycl::queue* queue, const float* gates_x, const float* w_hh,
                                  const float* b_hh, const float* h0, float* out, int64_t hidden,
                                  int64_t steps, int32_t reverse) {
    constexpr size_t local_size = 256;
    queue->submit([&](::sycl::handler& handler) {
        ::sycl::local_accessor<float, 1> h(::sycl::range<1>(static_cast<size_t>(hidden)), handler);
        ::sycl::local_accessor<float, 1> rh(::sycl::range<1>(static_cast<size_t>(3 * hidden)),
                                            handler);
        handler.parallel_for<GruRecurrenceSYCLKernel>(
            ::sycl::nd_range<1>(local_size, local_size), [=](::sycl::nd_item<1> item) {
                const int64_t lid = static_cast<int64_t>(item.get_local_linear_id());
                const int64_t stride = static_cast<int64_t>(local_size);
                const int64_t gate_rows = 3 * hidden;
                for (int64_t j = lid; j < hidden; j += stride) {
                    h[j] = h0 ? h0[j] : 0.0f;
                }
                item.barrier(::sycl::access::fence_space::local_space);
                for (int64_t step = 0; step < steps; ++step) {
                    const int64_t t = reverse ? steps - 1 - step : step;
                    // rh[row] = b_hh[row] + w_hh[row, :] . h
                    for (int64_t row = lid; row < gate_rows; row += stride) {
                        const float* w_row = w_hh + row * hidden;
                        float sum = b_hh ? b_hh[row] : 0.0f;
                        for (int64_t j = 0; j < hidden; ++j) sum += w_row[j] * h[j];
                        rh[row] = sum;
                    }
                    // all reads of h done before it is overwritten
                    item.barrier(::sycl::access::fence_space::local_space);
                    const float* gx = gates_x + t * gate_rows;
                    float* out_t = out + t * hidden;
                    for (int64_t j = lid; j < hidden; j += stride) {
                        const float r = recurrent_sigmoidf(gx[j] + rh[j]);
                        const float z = recurrent_sigmoidf(gx[hidden + j] + rh[hidden + j]);
                        const float n =
                            ::sycl::tanh(gx[2 * hidden + j] + r * rh[2 * hidden + j]);
                        h[j] = (1.0f - z) * n + z * h[j];
                        out_t[j] = h[j];
                    }
                    // h fully updated before the next step's matvec
                    item.barrier(::sycl::access::fence_space::local_space);
                }
            });
    });
}

// Single work-group; local memory = h[hidden] + c[hidden] + rh[4 * hidden].
static void launch_lstm_recurrence(::sycl::queue* queue, const float* gates_x, const float* w_hh,
                                   const float* b_hh, const float* h0, const float* c0,
                                   float* out, int64_t hidden, int64_t steps, int32_t reverse) {
    constexpr size_t local_size = 256;
    queue->submit([&](::sycl::handler& handler) {
        ::sycl::local_accessor<float, 1> h(::sycl::range<1>(static_cast<size_t>(hidden)), handler);
        ::sycl::local_accessor<float, 1> c(::sycl::range<1>(static_cast<size_t>(hidden)), handler);
        ::sycl::local_accessor<float, 1> rh(::sycl::range<1>(static_cast<size_t>(4 * hidden)),
                                            handler);
        handler.parallel_for<LstmRecurrenceSYCLKernel>(
            ::sycl::nd_range<1>(local_size, local_size), [=](::sycl::nd_item<1> item) {
                const int64_t lid = static_cast<int64_t>(item.get_local_linear_id());
                const int64_t stride = static_cast<int64_t>(local_size);
                const int64_t gate_rows = 4 * hidden;
                for (int64_t j = lid; j < hidden; j += stride) {
                    h[j] = h0 ? h0[j] : 0.0f;
                    c[j] = c0 ? c0[j] : 0.0f;
                }
                item.barrier(::sycl::access::fence_space::local_space);
                for (int64_t step = 0; step < steps; ++step) {
                    const int64_t t = reverse ? steps - 1 - step : step;
                    for (int64_t row = lid; row < gate_rows; row += stride) {
                        const float* w_row = w_hh + row * hidden;
                        float sum = b_hh ? b_hh[row] : 0.0f;
                        for (int64_t j = 0; j < hidden; ++j) sum += w_row[j] * h[j];
                        rh[row] = sum;
                    }
                    item.barrier(::sycl::access::fence_space::local_space);
                    const float* gx = gates_x + t * gate_rows;
                    float* out_t = out + t * hidden;
                    for (int64_t j = lid; j < hidden; j += stride) {
                        const float i = recurrent_sigmoidf(gx[j] + rh[j]);
                        const float f = recurrent_sigmoidf(gx[hidden + j] + rh[hidden + j]);
                        const float g = ::sycl::tanh(gx[2 * hidden + j] + rh[2 * hidden + j]);
                        const float o =
                            recurrent_sigmoidf(gx[3 * hidden + j] + rh[3 * hidden + j]);
                        c[j] = f * c[j] + i * g;
                        h[j] = o * ::sycl::tanh(c[j]);
                        out_t[j] = h[j];
                    }
                    item.barrier(::sycl::access::fence_space::local_space);
                }
            });
    });
}

static bool launch_recurrent(ggml_backend_t backend, ggml_tensor* node, bool lstm) {
    ::sycl::queue* queue =
        static_cast<::sycl::queue*>(ggml_ops_ext_bridge_sycl_get_queue(backend));
    if (!queue) {
        return false;
    }

    const ggml_tensor* x = node->src[0];
    const ggml_tensor* w_ih = node->src[1];
    const ggml_tensor* w_hh = node->src[2];
    const ggml_tensor* b_ih = node->src[3];
    const ggml_tensor* b_hh = node->src[4];
    const ggml_tensor* h0 = node->src[5];
    const ggml_tensor* c0 = lstm ? node->src[6] : nullptr;
    ops_recurrent_params params;
    std::memcpy(&params, node->op_params, sizeof(params));

    const int64_t input_dim = x->ne[0];
    const int64_t steps = x->ne[1];
    const int64_t hidden = w_hh->ne[0];
    const int64_t gate_rows = (lstm ? 4 : 3) * hidden;

    ops_sycl_pool_alloc<float> gates_x(backend);
    if (!gates_x.alloc(static_cast<size_t>(steps * gate_rows))) {
        return false;
    }

    try {
        launch_project_inputs(queue, static_cast<const float*>(x->data),
                              static_cast<const float*>(w_ih->data),
                              b_ih ? static_cast<const float*>(b_ih->data) : nullptr,
                              gates_x.get(), input_dim, steps, gate_rows);
        if (lstm) {
            launch_lstm_recurrence(queue, gates_x.get(), static_cast<const float*>(w_hh->data),
                                   b_hh ? static_cast<const float*>(b_hh->data) : nullptr,
                                   h0 ? static_cast<const float*>(h0->data) : nullptr,
                                   c0 ? static_cast<const float*>(c0->data) : nullptr,
                                   static_cast<float*>(node->data), hidden, steps,
                                   params.reverse);
        } else {
            launch_gru_recurrence(queue, gates_x.get(), static_cast<const float*>(w_hh->data),
                                  b_hh ? static_cast<const float*>(b_hh->data) : nullptr,
                                  h0 ? static_cast<const float*>(h0->data) : nullptr,
                                  static_cast<float*>(node->data), hidden, steps,
                                  params.reverse);
        }
        // The scratch buffer returns to the pool at scope exit; the in-order
        // queue keeps any future reuse ordered after these kernels.
    } catch (const ::sycl::exception& error) {
        std::fprintf(stderr, "[ops-sycl] %s failed: %s\n", lstm ? "lstm" : "gru", error.what());
        return false;
    }
    return true;
}

bool ggml_sycl_op_gru_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    return launch_recurrent(backend, node, false);
}

bool ggml_sycl_op_lstm_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    return launch_recurrent(backend, node, true);
}

} // namespace sycl
} // namespace ggml_ops_ext
