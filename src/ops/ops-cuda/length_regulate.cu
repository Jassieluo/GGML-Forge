#include "ops_cuda_common.cuh"

#include <cstring>

namespace ggml_ops_ext {
namespace cuda {

// Duration-based frame expansion. A single resident block builds the
// (clamped) duration prefix sum in shared memory, then all threads scatter
// output frames by binary-searching their source frame. Matches the CPU
// kernel's clamp behavior exactly: durations <= 0 skip the frame, excess
// output zero-fills, excess input is dropped. Probe caps frames at
// OPS_LENGTH_REGULATE_MAX_FRAMES so the prefix array fits in shared memory.
__global__ void length_regulate_kernel(const float* x, const int32_t* dur, float* out,
                                       int64_t channels, int64_t frames, int64_t total) {
    extern __shared__ int32_t prefix[]; // frames + 1 entries
    if (threadIdx.x == 0) {
        prefix[0] = 0;
        for (int64_t t = 0; t < frames; ++t) {
            const int32_t d = dur[t] > 0 ? dur[t] : 0;
            int64_t next = (int64_t)prefix[t] + d;
            if (next > total) next = total;
            prefix[t + 1] = (int32_t)next;
        }
    }
    __syncthreads();

    const int64_t count = channels * total;
    for (int64_t idx = threadIdx.x; idx < count; idx += blockDim.x) {
        const int64_t pos = idx / channels;
        const int64_t c = idx % channels;
        int found = -1;
        int lo = 0, hi = (int)frames - 1;
        while (lo <= hi) {
            const int mid = (lo + hi) / 2;
            if ((int64_t)prefix[mid] <= pos && pos < (int64_t)prefix[mid + 1]) {
                found = mid;
                break;
            }
            if (pos < (int64_t)prefix[mid]) {
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        }
        out[pos * channels + c] = found >= 0 ? x[(int64_t)found * channels + c] : 0.0f;
    }
}

bool ggml_cuda_op_length_regulate_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    ops_length_regulate_params params;
    std::memcpy(&params, node->op_params, sizeof(params));

    const ggml_tensor* x = node->src[0];
    const ggml_tensor* durations = node->src[1];
    const int64_t channels = x->ne[0];
    const int64_t frames = x->ne[1];
    const int64_t total = params.total;

    int device = ggml_ops_ext_bridge_cuda_get_device(backend);
    cudaStream_t stream = (cudaStream_t)ggml_ops_ext_bridge_cuda_get_stream(backend);
    CUDA_CHECK(cudaSetDevice(device));

    const size_t shared_bytes = (size_t)(frames + 1) * sizeof(int32_t);
    length_regulate_kernel<<<1, 256, shared_bytes, stream>>>(
        (const float*)x->data, (const int32_t*)durations->data, (float*)node->data,
        channels, frames, total);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace cuda
} // namespace ggml_ops_ext
