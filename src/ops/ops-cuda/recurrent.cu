#include "ops_cuda_common.cuh"

#include <algorithm>
#include <cstring>

namespace ggml_ops_ext {
namespace cuda {

// Sequence-level fused GRU / LSTM (torch.nn gate order; see
// contracts/recurrent.h). Two phases, matching the CPU kernel:
//   1. recurrent_project_inputs_kernel — whole-sequence input projection
//      gates_x[t, row] = b_ih[row] + w_ih[row, :] . x[t, :], embarrassingly
//      parallel grid-stride over steps * gate_rows into a scratch buffer.
//   2. a single resident block runs the sequential recurrence with h (and c
//      for LSTM) plus the per-step gate scratch in shared memory; threads
//      parallelize over gate rows / hidden units, barrier between steps.
// Probe caps hidden at 1024 so the shared arrays fit (LSTM worst case
// (1 + 1 + 4) * hidden floats = 24 KiB).

__device__ inline float recurrent_sigmoidf(float value) {
    return 1.0f / (1.0f + expf(-value));
}

// gates_x[t * gate_rows + row] = b_ih[row] + sum_i w_ih[row * in + i] * x[t * in + i]
__global__ void recurrent_project_inputs_kernel(const float* x, const float* w_ih,
                                                const float* b_ih, float* gates_x,
                                                int64_t input_dim, int64_t steps,
                                                int64_t gate_rows) {
    const int64_t count = steps * gate_rows;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < count;
         idx += (int64_t)gridDim.x * blockDim.x) {
        const int64_t t = idx / gate_rows;
        const int64_t row = idx % gate_rows;
        const float* x_t = x + t * input_dim;
        const float* w_row = w_ih + row * input_dim;
        float sum = b_ih ? b_ih[row] : 0.0f;
        for (int64_t i = 0; i < input_dim; ++i) sum += w_row[i] * x_t[i];
        gates_x[idx] = sum;
    }
}

// Single resident block; shared = h[hidden] + rh[3 * hidden].
__global__ void gru_recurrence_kernel(const float* gates_x, const float* w_hh, const float* b_hh,
                                      const float* h0, float* out, int64_t hidden, int64_t steps,
                                      int32_t reverse) {
    extern __shared__ float shared[];
    float* h = shared;           // hidden
    float* rh = shared + hidden; // 3 * hidden
    const int64_t gate_rows = 3 * hidden;
    for (int64_t j = threadIdx.x; j < hidden; j += blockDim.x) {
        h[j] = h0 ? h0[j] : 0.0f;
    }
    __syncthreads();
    for (int64_t step = 0; step < steps; ++step) {
        const int64_t t = reverse ? steps - 1 - step : step;
        // rh[row] = b_hh[row] + w_hh[row, :] . h
        for (int64_t row = threadIdx.x; row < gate_rows; row += blockDim.x) {
            const float* w_row = w_hh + row * hidden;
            float sum = b_hh ? b_hh[row] : 0.0f;
            for (int64_t j = 0; j < hidden; ++j) sum += w_row[j] * h[j];
            rh[row] = sum;
        }
        __syncthreads(); // all reads of h done before it is overwritten
        const float* gx = gates_x + t * gate_rows;
        float* out_t = out + t * hidden;
        for (int64_t j = threadIdx.x; j < hidden; j += blockDim.x) {
            const float r = recurrent_sigmoidf(gx[j] + rh[j]);
            const float z = recurrent_sigmoidf(gx[hidden + j] + rh[hidden + j]);
            const float n = tanhf(gx[2 * hidden + j] + r * rh[2 * hidden + j]);
            h[j] = (1.0f - z) * n + z * h[j];
            out_t[j] = h[j];
        }
        __syncthreads(); // h fully updated before the next step's matvec
    }
}

// Single resident block; shared = h[hidden] + c[hidden] + rh[4 * hidden].
__global__ void lstm_recurrence_kernel(const float* gates_x, const float* w_hh, const float* b_hh,
                                       const float* h0, const float* c0, float* out,
                                       int64_t hidden, int64_t steps, int32_t reverse) {
    extern __shared__ float shared[];
    float* h = shared;                // hidden
    float* c = shared + hidden;       // hidden
    float* rh = shared + 2 * hidden;  // 4 * hidden
    const int64_t gate_rows = 4 * hidden;
    for (int64_t j = threadIdx.x; j < hidden; j += blockDim.x) {
        h[j] = h0 ? h0[j] : 0.0f;
        c[j] = c0 ? c0[j] : 0.0f;
    }
    __syncthreads();
    for (int64_t step = 0; step < steps; ++step) {
        const int64_t t = reverse ? steps - 1 - step : step;
        for (int64_t row = threadIdx.x; row < gate_rows; row += blockDim.x) {
            const float* w_row = w_hh + row * hidden;
            float sum = b_hh ? b_hh[row] : 0.0f;
            for (int64_t j = 0; j < hidden; ++j) sum += w_row[j] * h[j];
            rh[row] = sum;
        }
        __syncthreads();
        const float* gx = gates_x + t * gate_rows;
        float* out_t = out + t * hidden;
        for (int64_t j = threadIdx.x; j < hidden; j += blockDim.x) {
            const float i = recurrent_sigmoidf(gx[j] + rh[j]);
            const float f = recurrent_sigmoidf(gx[hidden + j] + rh[hidden + j]);
            const float g = tanhf(gx[2 * hidden + j] + rh[2 * hidden + j]);
            const float o = recurrent_sigmoidf(gx[3 * hidden + j] + rh[3 * hidden + j]);
            c[j] = f * c[j] + i * g;
            h[j] = o * tanhf(c[j]);
            out_t[j] = h[j];
        }
        __syncthreads();
    }
}

static bool launch_recurrent(ggml_backend_t backend, ggml_tensor* node, bool lstm) {
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

    int device = ggml_ops_ext_bridge_cuda_get_device(backend);
    cudaStream_t stream = ggml_ops_ext_bridge_cuda_get_stream(backend);
    CUDA_CHECK(cudaSetDevice(device));

    ops_cuda_alloc<float> gates_x(stream);
    if (!gates_x.alloc((size_t)(steps * gate_rows))) {
        return false;
    }

    constexpr int threads = 256;
    const int64_t count = steps * gate_rows;
    const int blocks = (int)std::min<int64_t>((count + threads - 1) / threads, 4096);
    recurrent_project_inputs_kernel<<<blocks, threads, 0, stream>>>(
        (const float*)x->data, (const float*)w_ih->data,
        b_ih ? (const float*)b_ih->data : nullptr, gates_x.get(), input_dim, steps, gate_rows);
    if (cudaGetLastError() != cudaSuccess) {
        return false;
    }

    const size_t shared_bytes =
        (size_t)((lstm ? 2 : 1) * hidden + gate_rows) * sizeof(float);
    if (lstm) {
        lstm_recurrence_kernel<<<1, threads, shared_bytes, stream>>>(
            gates_x.get(), (const float*)w_hh->data,
            b_hh ? (const float*)b_hh->data : nullptr,
            h0 ? (const float*)h0->data : nullptr, c0 ? (const float*)c0->data : nullptr,
            (float*)node->data, hidden, steps, params.reverse);
    } else {
        gru_recurrence_kernel<<<1, threads, shared_bytes, stream>>>(
            gates_x.get(), (const float*)w_hh->data,
            b_hh ? (const float*)b_hh->data : nullptr,
            h0 ? (const float*)h0->data : nullptr, (float*)node->data, hidden, steps,
            params.reverse);
    }
    return cudaGetLastError() == cudaSuccess;
}

bool ggml_cuda_op_gru_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    return launch_recurrent(backend, node, false);
}

bool ggml_cuda_op_lstm_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    return launch_recurrent(backend, node, true);
}

} // namespace cuda
} // namespace ggml_ops_ext
