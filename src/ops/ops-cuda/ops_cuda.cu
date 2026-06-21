#include "ops/ops.h"
#include "ops_cuda.h"
#include "ggml.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/convert.cuh"
#include <cudnn.h>
#include <mutex>
#include <unordered_map>
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

#define CUDNN_CHECK(status)                                                    \
    do {                                                                       \
        cudnnStatus_t s = (status);                                            \
        if (s != CUDNN_STATUS_SUCCESS) {                                       \
            fprintf(stderr, "cuDNN Error: %s at %s:%d\n",                      \
                    cudnnGetErrorString(s), __FILE__, __LINE__);               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static cudnnHandle_t g_cudnn_handles[GGML_CUDA_MAX_DEVICES] = { nullptr };
static std::mutex g_cudnn_mutex;

static cudnnHandle_t get_cudnn_handle(int device) {
    std::lock_guard<std::mutex> lock(g_cudnn_mutex);
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) return nullptr;
    if (!g_cudnn_handles[device]) {
        CUDNN_CHECK(cudnnCreate(&g_cudnn_handles[device]));
    }
    return g_cudnn_handles[device];
}

// Custom kernels for weight conversion on the fly
__global__ void convert_f16_to_f32_kernel(const half* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __half2float(src[idx]);
    }
}

__global__ void convert_bf16_to_f32_kernel(const unsigned short* src, float* dst, int n) {
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

namespace tts {
namespace ops {
namespace cuda {

bool compute_conv_1d(
    ggml_backend_t backend,
    struct ggml_tensor* w,
    struct ggml_tensor* x,
    struct ggml_tensor* dst,
    int stride,
    int padding,
    int dilation
) {
    // 1. Get the GGML CUDA context
    ggml_backend_cuda_context* ctx = (ggml_backend_cuda_context*)backend->context;
    int device = ctx->device;
    
    // Get CUDA stream. Create if not initialized.
    cudaStream_t stream = ctx->streams[device][ctx->curr_stream_no];
    if (stream == nullptr) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&ctx->streams[device][ctx->curr_stream_no], cudaStreamNonBlocking));
        stream = ctx->streams[device][ctx->curr_stream_no];
    }

    // Set CUDA device
    CUDA_CHECK(cudaSetDevice(device));

    // Get or create cuDNN handle and set stream
    cudnnHandle_t cudnn = get_cudnn_handle(device);
    CUDNN_CHECK(cudnnSetStream(cudnn, stream));

    // 2. Extract shapes
    int64_t N = x->ne[2];
    int64_t C = x->ne[1];
    int64_t W = x->ne[0];

    int64_t K  = w->ne[2];
    int64_t kW = w->ne[0];

    int64_t OW = dst->ne[0];

    // Create cuDNN descriptors
    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           N, C, 1, W));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           K, C, 1, kW));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           N, K, 1, OW));

    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
                                                 0 /*pad_h*/, padding /*pad_w*/,
                                                 1 /*stride_h*/, stride /*stride_w*/,
                                                 1 /*dilation_h*/, dilation /*dilation_w*/,
                                                 CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // Algorithm selection
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    size_t workspace_size = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, x_desc, w_desc, conv_desc, y_desc, algo, &workspace_size));

    // Workspace allocation using async allocator
    ops_cuda_alloc<uint8_t> workspace_alloc(stream);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        workspace_alloc.alloc(workspace_size);
        workspace = workspace_alloc.get();
    }

    // Cast weights to FP32 if needed (cuDNN needs float weights for float convolutions)
    const float* w_d = (const float*)w->data;
    ops_cuda_alloc<float> w_f32_alloc(stream);
    if (w->type != GGML_TYPE_F32) {
        const int64_t ne_w = ggml_nelements(w);
        w_f32_alloc.alloc(ne_w);
        int block_size = 256;
        int grid_size = (ne_w + block_size - 1) / block_size;
        if (w->type == GGML_TYPE_F16) {
            convert_f16_to_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                (const half*)w->data, w_f32_alloc.get(), ne_w);
            w_d = w_f32_alloc.get();
        } else if (w->type == GGML_TYPE_BF16) {
            convert_bf16_to_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                (const unsigned short*)w->data, w_f32_alloc.get(), ne_w);
            w_d = w_f32_alloc.get();
        } else {
            fprintf(stderr, "Unsupported weight type for cuDNN convolution: %d\n", w->type);
            exit(1);
        }
    }

    const float* x_d = (const float*)x->data;
    float* dst_d = (float*)dst->data;

    // Execute convolution
    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(
        cudnn,
        &alpha, x_desc, x_d,
        w_desc, w_d,
        conv_desc, algo,
        workspace, workspace_size,
        &beta, y_desc, dst_d));

    // Cleanup
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(w_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));

    return true;
}

bool compute_conv_transpose_1d(
    ggml_backend_t backend,
    struct ggml_tensor* w,
    struct ggml_tensor* x,
    struct ggml_tensor* dst,
    int stride,
    int padding,
    int dilation
) {
    // 1. Get the GGML CUDA context
    ggml_backend_cuda_context* ctx = (ggml_backend_cuda_context*)backend->context;
    int device = ctx->device;
    
    // Get CUDA stream. Create if not initialized.
    cudaStream_t stream = ctx->streams[device][ctx->curr_stream_no];
    if (stream == nullptr) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&ctx->streams[device][ctx->curr_stream_no], cudaStreamNonBlocking));
        stream = ctx->streams[device][ctx->curr_stream_no];
    }

    // Set CUDA device
    CUDA_CHECK(cudaSetDevice(device));

    // Get or create cuDNN handle and set stream
    cudnnHandle_t cudnn = get_cudnn_handle(device);
    CUDNN_CHECK(cudnnSetStream(cudnn, stream));

    // 2. Extract shapes for transposed convolution
    const int kW = (int)w->ne[0]; // kernel_size
    const int C  = (int)w->ne[1]; // in_channels (becomes output channels)
    const int K  = (int)w->ne[2]; // out_channels (becomes input channels)
    const int W  = (int)x->ne[0]; // input seq_len
    const int N  = (int)x->ne[2]; // batch

    const int OW = (int)dst->ne[0]; // output seq_len

    // Create cuDNN descriptors
    cudnnTensorDescriptor_t dy_desc, dx_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dy_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dx_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    // dx = output
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dx_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           N, C, 1, OW));
    // dy = input
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dy_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           N, K, 1, W));
    // Filter descriptor
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           K, C, 1, kW));

    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
                                                 0 /*pad_h*/, padding /*pad_w*/,
                                                 1 /*stride_h*/, stride /*stride_w*/,
                                                 1 /*dilation_h*/, dilation /*dilation_w*/,
                                                 CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // Algorithm selection
    cudnnConvolutionBwdDataAlgo_t algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    size_t workspace_size = 0;
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn, w_desc, dy_desc, conv_desc, dx_desc, algo, &workspace_size));

    // Workspace allocation using async allocator
    ops_cuda_alloc<uint8_t> workspace_alloc(stream);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        workspace_alloc.alloc(workspace_size);
        workspace = workspace_alloc.get();
    }

    // Cast weights to FP32 if needed
    const float* w_d = (const float*)w->data;
    ops_cuda_alloc<float> w_f32_alloc(stream);
    if (w->type != GGML_TYPE_F32) {
        const int64_t ne_w = ggml_nelements(w);
        w_f32_alloc.alloc(ne_w);
        int block_size = 256;
        int grid_size = (ne_w + block_size - 1) / block_size;
        if (w->type == GGML_TYPE_F16) {
            convert_f16_to_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                (const half*)w->data, w_f32_alloc.get(), ne_w);
            w_d = w_f32_alloc.get();
        } else if (w->type == GGML_TYPE_BF16) {
            convert_bf16_to_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                (const unsigned short*)w->data, w_f32_alloc.get(), ne_w);
            w_d = w_f32_alloc.get();
        } else {
            fprintf(stderr, "Unsupported weight type for cuDNN convolution: %d\n", w->type);
            exit(1);
        }
    }

    const float* x_d = (const float*)x->data;
    float* dst_d = (float*)dst->data;

    // Execute transposed convolution
    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionBackwardData(
        cudnn,
        &alpha, w_desc, w_d,
        dy_desc, x_d,
        conv_desc, algo,
        workspace, workspace_size,
        &beta, dx_desc, dst_d));

    // Cleanup
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dy_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dx_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(w_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));

    return true;
}

void register_backend() {
    ops_backend_interface iface;
    iface.backend_name_prefix = "CUDA";
    iface.compute_conv_1d = compute_conv_1d;
    iface.compute_conv_transpose_1d = compute_conv_transpose_1d;
    register_ops_backend(iface);
}

} // namespace cuda
} // namespace ops
} // namespace tts
