#include "ops_cuda_common.cuh"

#include <cstring>

namespace ggml_ops_ext {
namespace cuda {

// y[i, t] = x[i, t] + (i even ? sin : cos)(position * base^(-pair/width))
// with pair = i - (i % 2) and position = offset + (*position_tensor) + t.
// Matches src/ops/ops-cpu/pos_encoding.cpp exactly (same float expression order).
// NOTE: the optional position tensor lives in DEVICE memory; it is read inside
// the kernel, never dereferenced on the host.
template <typename T>
__global__ void pos_encoding_kernel(const T* x, T* dst, const int32_t* position, int64_t width,
                                    int64_t tokens, int64_t total, int32_t offset, float base) {
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const int64_t start = (int64_t)offset + (position ? (int64_t)*position : 0);
    const float log_base = logf(base);

    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += stride) {
        const int64_t i = idx % width;
        const int64_t t = (idx / width) % tokens;
        const int64_t pair = i - (i % 2);
        const float pos = (float)(start + t);
        const float angle = pos * expf(-log_base * (float)pair / (float)width);
        const float pe = ((i & 1) == 0) ? sinf(angle) : cosf(angle);
        dst[idx] = (T)((float)x[idx] + pe);
    }
}

bool ggml_cuda_op_pos_encoding(ggml_backend_t backend, struct ggml_tensor* x,
                               struct ggml_tensor* position, struct ggml_tensor* dst, float base,
                               int32_t offset) {
    int device = ggml_ops_ext_bridge_cuda_get_device(backend);
    cudaStream_t stream = (cudaStream_t)ggml_ops_ext_bridge_cuda_get_stream(backend);

    CUDA_CHECK(cudaSetDevice(device));

    const int64_t width = x->ne[0];
    const int64_t tokens = x->ne[1];
    const int64_t total = ggml_nelements(x);
    if (total == 0) {
        return true;
    }

    // Grid-stride loop with int64 indexing: no element-count limit, grid.x only.
    constexpr int threads = 256;
    const int64_t want_blocks = (total + threads - 1) / threads;
    const unsigned int blocks = (unsigned int)(want_blocks > 65535 ? 65535 : want_blocks);

    const int32_t* position_d = position ? (const int32_t*)position->data : nullptr;

    if (x->type == GGML_TYPE_F32) {
        pos_encoding_kernel<float><<<blocks, threads, 0, stream>>>(
            (const float*)x->data, (float*)dst->data, position_d, width, tokens, total, offset,
            base);
    } else if (x->type == GGML_TYPE_F16) {
        pos_encoding_kernel<half><<<blocks, threads, 0, stream>>>(
            (const half*)x->data, (half*)dst->data, position_d, width, tokens, total, offset,
            base);
    } else {
        fprintf(stderr, "Unsupported data type for CUDA PosEncoding: %d\n", x->type);
        return false;
    }

    return cudaGetLastError() == cudaSuccess;
}

bool ggml_cuda_op_pos_encoding_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    ops_pos_encoding_params params;
    std::memcpy(&params, node->op_params, sizeof(params));
    return ggml_cuda_op_pos_encoding(backend, node->src[0], node->src[1], node, params.base,
                                     params.offset);
}

} // namespace cuda
} // namespace ggml_ops_ext
