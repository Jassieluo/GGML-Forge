#pragma once

#include "ops/ops.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

// ────────────────────────────────────────────────────────────
// Bridge function pointer types (resolved at runtime from
// ggml-cuda.dll via GetProcAddress, since ggml-cuda is a
// MODULE_LIBRARY with no import library).
// ────────────────────────────────────────────────────────────

typedef int (*pfn_bridge_cuda_get_device_t)(ggml_backend_t);
typedef void* (*pfn_bridge_cuda_get_stream_t)(ggml_backend_t);
typedef void* (*pfn_bridge_cuda_get_cublas_t)(ggml_backend_t);
typedef bool (*pfn_bridge_cuda_dequantize_t)(ggml_backend_t, const struct ggml_tensor *, void *, enum ggml_type);

// Initialized in ops_cuda.cu :: register_backend()
extern pfn_bridge_cuda_get_device_t g_bridge_cuda_get_device;
extern pfn_bridge_cuda_get_stream_t g_bridge_cuda_get_stream;
extern pfn_bridge_cuda_get_cublas_t g_bridge_cuda_get_cublas;
extern pfn_bridge_cuda_dequantize_t g_bridge_cuda_dequantize;

// Inline wrappers — call through the function pointers.
// Safer to use than the raw pointers directly in kernel code.
inline int ggml_ops_ext_bridge_cuda_get_device(ggml_backend_t backend) {
    return g_bridge_cuda_get_device ? g_bridge_cuda_get_device(backend) : 0;
}

inline cudaStream_t ggml_ops_ext_bridge_cuda_get_stream(ggml_backend_t backend) {
    return g_bridge_cuda_get_stream
        ? (cudaStream_t)g_bridge_cuda_get_stream(backend)
        : nullptr;
}

inline cublasHandle_t ggml_ops_ext_bridge_cuda_get_cublas(ggml_backend_t backend) {
    return g_bridge_cuda_get_cublas
        ? (cublasHandle_t)g_bridge_cuda_get_cublas(backend)
        : nullptr;
}

inline bool ggml_ops_ext_bridge_cuda_dequantize(
    ggml_backend_t backend, const struct ggml_tensor * src, void * dst, enum ggml_type dst_type
) {
    return g_bridge_cuda_dequantize && g_bridge_cuda_dequantize(backend, src, dst, dst_type);
}
#include "ggml-cuda/convert.cuh"
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

#undef CUBLAS_CHECK
#define CUBLAS_CHECK(err)                                                      \
    do {                                                                       \
        cublasStatus_t e = (err);                                              \
        if (e != CUBLAS_STATUS_SUCCESS) {                                      \
            fprintf(stderr, "cuBLAS Error: %d at %s:%d\n",                     \
                    (int)e, __FILE__, __LINE__);                               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

namespace ggml_ops_ext {
namespace cuda {

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
