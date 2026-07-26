#include "ops/cpu.h"
#include "ops/ops.h"
#include "ops_cpu_common.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace ggml_ops_ext::cpu {
namespace {

inline float sigmoidf(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

// gates_x[t * G*H + row] = bias_ih[row] + sum_i w_ih[row * in + i] * x[t * in + i]
// The whole-sequence input projection is embarrassingly parallel; only the
// recurrence below is sequential.
void project_inputs(const float* x, const float* w_ih, const float* b_ih,
                    float* gates_x, int64_t input_dim, int64_t steps, int64_t gate_rows,
                    int threads) {
#pragma omp parallel for collapse(2) num_threads(threads) schedule(static)
    for (int64_t t = 0; t < steps; ++t) {
        for (int64_t row = 0; row < gate_rows; ++row) {
            const float* x_t = x + t * input_dim;
            const float* w_row = w_ih + row * input_dim;
            float sum = b_ih ? b_ih[row] : 0.0f;
            for (int64_t i = 0; i < input_dim; ++i) sum += w_row[i] * x_t[i];
            gates_x[t * gate_rows + row] = sum;
        }
    }
}

// rh[row] = bias_hh[row] + sum_j w_hh[row * hidden + j] * h[j]
void project_hidden(const float* h, const float* w_hh, const float* b_hh, float* rh,
                    int64_t hidden, int64_t gate_rows, int threads) {
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t row = 0; row < gate_rows; ++row) {
        const float* w_row = w_hh + row * hidden;
        float sum = b_hh ? b_hh[row] : 0.0f;
        for (int64_t j = 0; j < hidden; ++j) sum += w_row[j] * h[j];
        rh[row] = sum;
    }
}

} // namespace

bool ops_cpu_op_gru(ggml_backend_t backend, ggml_tensor* node) {
    const ggml_tensor* x = node->src[0];
    const ggml_tensor* w_ih = node->src[1];
    const ggml_tensor* w_hh = node->src[2];
    const ggml_tensor* b_ih = node->src[3];
    const ggml_tensor* b_hh = node->src[4];
    const ggml_tensor* h0 = node->src[5];
    ops_recurrent_params params;
    std::memcpy(&params, node->op_params, sizeof(params));

    const int64_t input_dim = x->ne[0];
    const int64_t steps = x->ne[1];
    const int64_t hidden = w_hh->ne[0];
    const int64_t gate_rows = 3 * hidden;
    const int threads = backend_thread_count(backend);
    float* out = static_cast<float*>(node->data);

    std::vector<float> gates_x(static_cast<size_t>(steps * gate_rows));
    project_inputs(static_cast<const float*>(x->data), static_cast<const float*>(w_ih->data),
                   b_ih ? static_cast<const float*>(b_ih->data) : nullptr,
                   gates_x.data(), input_dim, steps, gate_rows, threads);

    std::vector<float> h(static_cast<size_t>(hidden), 0.0f);
    if (h0) std::memcpy(h.data(), h0->data, static_cast<size_t>(hidden) * sizeof(float));
    std::vector<float> rh(static_cast<size_t>(gate_rows));

    for (int64_t step = 0; step < steps; ++step) {
        const int64_t t = params.reverse ? steps - 1 - step : step;
        project_hidden(h.data(), static_cast<const float*>(w_hh->data),
                       b_hh ? static_cast<const float*>(b_hh->data) : nullptr,
                       rh.data(), hidden, gate_rows, threads);
        const float* gx = gates_x.data() + t * gate_rows;
        float* out_t = out + t * hidden;
        for (int64_t j = 0; j < hidden; ++j) {
            const float r = sigmoidf(gx[j] + rh[j]);
            const float z = sigmoidf(gx[hidden + j] + rh[hidden + j]);
            const float n = std::tanh(gx[2 * hidden + j] + r * rh[2 * hidden + j]);
            h[j] = (1.0f - z) * n + z * h[j];
            out_t[j] = h[j];
        }
    }
    return true;
}

bool ops_cpu_op_lstm(ggml_backend_t backend, ggml_tensor* node) {
    const ggml_tensor* x = node->src[0];
    const ggml_tensor* w_ih = node->src[1];
    const ggml_tensor* w_hh = node->src[2];
    const ggml_tensor* b_ih = node->src[3];
    const ggml_tensor* b_hh = node->src[4];
    const ggml_tensor* h0 = node->src[5];
    const ggml_tensor* c0 = node->src[6];
    ops_recurrent_params params;
    std::memcpy(&params, node->op_params, sizeof(params));

    const int64_t input_dim = x->ne[0];
    const int64_t steps = x->ne[1];
    const int64_t hidden = w_hh->ne[0];
    const int64_t gate_rows = 4 * hidden;
    const int threads = backend_thread_count(backend);
    float* out = static_cast<float*>(node->data);

    std::vector<float> gates_x(static_cast<size_t>(steps * gate_rows));
    project_inputs(static_cast<const float*>(x->data), static_cast<const float*>(w_ih->data),
                   b_ih ? static_cast<const float*>(b_ih->data) : nullptr,
                   gates_x.data(), input_dim, steps, gate_rows, threads);

    std::vector<float> h(static_cast<size_t>(hidden), 0.0f);
    std::vector<float> c(static_cast<size_t>(hidden), 0.0f);
    if (h0) std::memcpy(h.data(), h0->data, static_cast<size_t>(hidden) * sizeof(float));
    if (c0) std::memcpy(c.data(), c0->data, static_cast<size_t>(hidden) * sizeof(float));
    std::vector<float> rh(static_cast<size_t>(gate_rows));

    for (int64_t step = 0; step < steps; ++step) {
        const int64_t t = params.reverse ? steps - 1 - step : step;
        project_hidden(h.data(), static_cast<const float*>(w_hh->data),
                       b_hh ? static_cast<const float*>(b_hh->data) : nullptr,
                       rh.data(), hidden, gate_rows, threads);
        const float* gx = gates_x.data() + t * gate_rows;
        float* out_t = out + t * hidden;
        for (int64_t j = 0; j < hidden; ++j) {
            const float i = sigmoidf(gx[j] + rh[j]);
            const float f = sigmoidf(gx[hidden + j] + rh[hidden + j]);
            const float g = std::tanh(gx[2 * hidden + j] + rh[2 * hidden + j]);
            const float o = sigmoidf(gx[3 * hidden + j] + rh[3 * hidden + j]);
            c[j] = f * c[j] + i * g;
            h[j] = o * std::tanh(c[j]);
            out_t[j] = h[j];
        }
    }
    return true;
}

} // namespace ggml_ops_ext::cpu
