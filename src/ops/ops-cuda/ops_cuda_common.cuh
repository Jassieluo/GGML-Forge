#pragma once

#include "ops/ops.h"
#include "ggml.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/convert.cuh"
#ifdef GGML_USE_CUDNN
#include <cudnn.h>
#endif
#include <mutex>
#include <iostream>

#undef CUDA_CHECK
#define CUDA_CHECK(err)                                                        \
    do {                                                                       \
        cudaError_t e = (err);                                                 \
        if (e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n",                       \
                    cudaGetErrorString(e), __FILE__, __LINE__);                \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#ifdef GGML_USE_CUDNN
#define CUDNN_CHECK(status)                                                    \
    do {                                                                       \
        cudnnStatus_t s = (status);                                            \
        if (s != CUDNN_STATUS_SUCCESS) {                                       \
            fprintf(stderr, "cuDNN Error: %s at %s:%d\n",                      \
                    cudnnGetErrorString(s), __FILE__, __LINE__);               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)
#endif

namespace ggml_ops_ext {
namespace cuda {

// Unified cuDNN handle fetcher defined in ops_cuda.cu
#ifdef GGML_USE_CUDNN
cudnnHandle_t get_cudnn_handle(int device);
#endif

// Custom kernels for weight conversion on the fly
static __global__ void convert_f16_to_f32_kernel(const half* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __half2float(src[idx]);
    }
}

static __global__ void convert_bf16_to_f32_kernel(const unsigned short* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned int val = ((unsigned int)src[idx]) << 16;
        dst[idx] = *reinterpret_cast<float*>(&val);
    }
}

// Custom stream-ordered allocator to replace ggml_cuda_pool_alloc
template <typename T>
class ops_cuda_alloc {
    T* ptr = nullptr;
    cudaStream_t stream;
public:
    ops_cuda_alloc(cudaStream_t stream) : stream(stream) {}
    ~ops_cuda_alloc() {
        if (ptr) {
            cudaFreeAsync(ptr, stream);
        }
    }
    void alloc(size_t n) {
        if (ptr) {
            cudaFreeAsync(ptr, stream);
            ptr = nullptr;
        }
        CUDA_CHECK(cudaMallocAsync(&ptr, n * sizeof(T), stream));
    }
    T* get() const { return ptr; }
};

} // namespace cuda
} // namespace ggml_ops_ext
