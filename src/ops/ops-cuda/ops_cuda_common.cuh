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

// Initialized in ops_cuda.cu :: register_backend()
extern pfn_bridge_cuda_get_device_t g_bridge_cuda_get_device;
extern pfn_bridge_cuda_get_stream_t g_bridge_cuda_get_stream;
extern pfn_bridge_cuda_get_cublas_t g_bridge_cuda_get_cublas;

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

#define CUBLAS_CHECK(err)                                                      \
    do {                                                                       \
        cublasStatus_t e = (err);                                              \
        if (e != CUBLAS_STATUS_SUCCESS) {                                      \
            fprintf(stderr, "cuBLAS Error: %d at %s:%d\n",                     \
                    (int)e, __FILE__, __LINE__);                               \
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

#ifndef GGML_CUDA_MAX_DEVICES
#define GGML_CUDA_MAX_DEVICES 16
#endif

class CudaWorkspace {
    void* ptrs[GGML_CUDA_MAX_DEVICES];
    size_t sizes[GGML_CUDA_MAX_DEVICES];
public:
    CudaWorkspace() {
        for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
            ptrs[i] = nullptr;
            sizes[i] = 0;
        }
    }
    ~CudaWorkspace() {
        for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
            if (ptrs[i]) {
                cudaSetDevice(i);
                cudaFree(ptrs[i]);
            }
        }
    }
    void* get(int device, size_t req_size, cudaStream_t stream) {
        if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) return nullptr;
        if (sizes[device] < req_size) {
            int orig_device = 0;
            cudaGetDevice(&orig_device);
            if (orig_device != device) {
                cudaSetDevice(device);
            }
            if (ptrs[device]) {
                cudaFreeAsync(ptrs[device], stream);
                ptrs[device] = nullptr;
                sizes[device] = 0;
            }
            CUDA_CHECK(cudaMallocAsync(&ptrs[device], req_size, stream));
            sizes[device] = req_size;
            if (orig_device != device) {
                cudaSetDevice(orig_device);
            }
        }
        return ptrs[device];
    }
};

inline void* get_cuda_workspace(int device, size_t req_size, cudaStream_t stream) {
    thread_local CudaWorkspace workspace;
    return workspace.get(device, req_size, stream);
}

} // namespace cuda
} // namespace ggml_ops_ext
