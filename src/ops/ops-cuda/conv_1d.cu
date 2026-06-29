#include "ops_cuda_common.cuh"

namespace ggml_ops_ext {
namespace cuda {

template <typename T>
__device__ inline float to_float(T val) {
    return (float)val;
}

#if defined(__CUDACC__)
__device__ inline float to_float(half val) {
    return __half2float(val);
}
#endif

#ifndef GGML_USE_CUDNN
template <typename T_in, typename T_out>
__global__ void im2col_1d_kernel_chunked(
    const T_in* x, T_out* data_col,
    int64_t C, int64_t W, int64_t OW, int64_t kW,
    int stride, int padding, int dilation,
    int64_t N,
    size_t nb_x0, size_t nb_x1, size_t nb_x2,
    int64_t ow_start, int64_t cur_chunk_size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * C * kW * cur_chunk_size;
    if (idx < total) {
        int64_t ow_offset = idx % cur_chunk_size;
        int64_t tmp = idx / cur_chunk_size;
        int64_t ik = tmp % kW;
        tmp = tmp / kW;
        int64_t ic = tmp % C;
        int64_t n = tmp / C;

        int64_t ow = ow_start + ow_offset;
        int64_t iw = ow * stride - padding + ik * dilation;
        T_out val = 0.0f;
        if (iw >= 0 && iw < W) {
            const T_in* px = (const T_in*)((const char*)x + n * nb_x2 + ic * nb_x1 + iw * nb_x0);
            val = (T_out)to_float(*px);
        }
        data_col[n * (C * kW * cur_chunk_size) + (ic * kW + ik) * cur_chunk_size + ow_offset] = val;
    }
}
#endif

namespace {
__global__ void cast_half_to_float_kernel(const half* src, float* dst, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __half2float(src[idx]);
    }
}

__global__ void cast_float_to_half_kernel(const float* src, half* dst, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

void cast_tensor_cuda(const void* src, void* dst, ggml_type src_type, ggml_type dst_type, int64_t n, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    if (src_type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F32) {
        cast_half_to_float_kernel<<<grid_size, block_size, 0, stream>>>((const half*)src, (float*)dst, n);
    } else if (src_type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
        cast_float_to_half_kernel<<<grid_size, block_size, 0, stream>>>((const float*)src, (half*)dst, n);
    }
}
} // namespace

bool ggml_cuda_op_conv_1d(
    ggml_backend_t backend,
    struct ggml_tensor* w,
    struct ggml_tensor* x,
    struct ggml_tensor* node,
    int stride,
    int padding,
    int dilation
) {
    struct ggml_tensor* dst = node;
    int device = ggml_ops_ext_bridge_cuda_get_device(backend);
    cudaStream_t stream = (cudaStream_t)ggml_ops_ext_bridge_cuda_get_stream(backend);

    // Set CUDA device
    CUDA_CHECK(cudaSetDevice(device));

    // Determine weights type and pointer
    cudaDataType_t w_type = CUDA_R_32F;
    if (w->type == GGML_TYPE_F16) {
        w_type = CUDA_R_16F;
    } else if (w->type == GGML_TYPE_BF16) {
        w_type = CUDA_R_16BF;
    } else if (w->type != GGML_TYPE_F32) {
        fprintf(stderr, "Unsupported weight type for CUDA convolution: %d\n", w->type);
        exit(1);
    }
    const void* w_d = w->data;

    const void* x_d = x->data;
    void* dst_d = dst->data;

    int64_t N = x->ne[2];
    int64_t C = x->ne[1];
    int64_t W = x->ne[0];
    int64_t K = w->ne[2];
    int64_t kW = w->ne[0];
    int64_t OW = dst->ne[0];

#ifdef GGML_USE_CUDNN
    // Get or create cuDNN handle and set stream
    cudnnHandle_t cudnn = get_cudnn_handle(device);
    CUDNN_CHECK(cudnnSetStream(cudnn, stream));

    // Create cuDNN descriptors
    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    cudnnDataType_t cudnn_x_type = (x->type == GGML_TYPE_F16) ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, cudnn_x_type,
                                           N, C, 1, W));
    
    cudnnDataType_t cudnn_w_type = CUDNN_DATA_FLOAT;
    if (w->type == GGML_TYPE_F16) {
        cudnn_w_type = CUDNN_DATA_HALF;
    } else if (w->type == GGML_TYPE_BF16) {
        cudnn_w_type = CUDNN_DATA_BFLOAT16;
    }

    CUDNN_CHECK(cudnnSetFilter4dDescriptor(w_desc, cudnn_w_type, CUDNN_TENSOR_NCHW,
                                           K, C, 1, kW));
    
    cudnnDataType_t cudnn_dst_type = (dst->type == GGML_TYPE_F16) ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, cudnn_dst_type,
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
#else
    // Get cuBLAS handle from backend context and set stream
    cublasHandle_t cublas = (cublasHandle_t)ggml_ops_ext_bridge_cuda_get_cublas(backend);
    CUBLAS_CHECK(cublasSetStream(cublas, stream));

    bool is_1x1 = (kW == 1 && stride == 1 && padding == 0 && dilation == 1);
    float alpha = 1.0f;
    float beta = 0.0f;

    cudaDataType_t x_type = (x->type == GGML_TYPE_F16) ? CUDA_R_16F : CUDA_R_32F;
    cudaDataType_t dst_type = (dst->type == GGML_TYPE_F16) ? CUDA_R_16F : CUDA_R_32F;
    size_t x_elem_size = (x->type == GGML_TYPE_F16) ? sizeof(half) : sizeof(float);
    size_t dst_elem_size = (dst->type == GGML_TYPE_F16) ? sizeof(half) : sizeof(float);

    // Ensure weights have the same precision as activations (x->type) for cuBLAS
    const void* w_d_actual = w_d;
    cudaDataType_t w_type_actual = w_type;
    ops_cuda_alloc<float> w_f32_alloc(stream);
    ops_cuda_alloc<half> w_f16_alloc(stream);

    if (w->type != x->type) {
        int64_t w_len = ggml_nelements(w);
        if (x->type == GGML_TYPE_F32) {
            w_f32_alloc.alloc(w_len);
            cast_tensor_cuda(w_d, w_f32_alloc.get(), w->type, GGML_TYPE_F32, w_len, stream);
            w_d_actual = w_f32_alloc.get();
            w_type_actual = CUDA_R_32F;
        } else if (x->type == GGML_TYPE_F16) {
            w_f16_alloc.alloc(w_len);
            cast_tensor_cuda(w_d, w_f16_alloc.get(), w->type, GGML_TYPE_F16, w_len, stream);
            w_d_actual = w_f16_alloc.get();
            w_type_actual = CUDA_R_16F;
        }
    }

    if (is_1x1) {
        // 1x1 Convolution Shortcut: Bypasses im2col completely, direct GEMM write
        for (int64_t n = 0; n < N; ++n) {
            CUBLAS_CHECK(cublasGemmEx(
                cublas,
                CUBLAS_OP_N, CUBLAS_OP_N,
                OW, K, C,
                &alpha,
                (const char*)x_d + n * (C * OW * x_elem_size), x_type, OW,
                w_d_actual, w_type_actual, C,
                &beta,
                (char*)dst_d + n * (K * OW * dst_elem_size), dst_type, OW,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT
            ));
        }
    } else {
        // Chunked GEMM convolution to balance VRAM usage and speed
        const int64_t CHUNK_SIZE = 2048;
        cudaDataType_t data_col_type = (x->type == GGML_TYPE_F16) ? CUDA_R_16F : CUDA_R_32F;
        size_t data_col_elem_size = (x->type == GGML_TYPE_F16) ? sizeof(half) : sizeof(float);
        
        void* data_col = get_cuda_workspace(device, N * C * kW * CHUNK_SIZE * data_col_elem_size, stream);

        for (int64_t ow_start = 0; ow_start < OW; ow_start += CHUNK_SIZE) {
            int64_t cur_chunk_size = min(CHUNK_SIZE, OW - ow_start);

            // Launch chunked im2col kernel
            int64_t total_elements = N * C * kW * cur_chunk_size;
            int block_size = 256;
            int grid_size = (total_elements + block_size - 1) / block_size;
            
            if (x->type == GGML_TYPE_F16) {
                im2col_1d_kernel_chunked<half, half><<<grid_size, block_size, 0, stream>>>(
                    (const half*)x_d, (half*)data_col,
                    C, W, OW, kW,
                    stride, padding, dilation,
                    N,
                    x->nb[0], x->nb[1], x->nb[2],
                    ow_start, cur_chunk_size
                );
            } else {
                im2col_1d_kernel_chunked<float, float><<<grid_size, block_size, 0, stream>>>(
                    (const float*)x_d, (float*)data_col,
                    C, W, OW, kW,
                    stride, padding, dilation,
                    N,
                    x->nb[0], x->nb[1], x->nb[2],
                    ow_start, cur_chunk_size
                );
            }

            // Call cublasGemmStridedBatchedEx for all batch elements at once
            long long int strideA = C * kW * cur_chunk_size;
            long long int strideB = 0;
            long long int strideC = K * OW;

            CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                cublas,
                CUBLAS_OP_N, CUBLAS_OP_N,
                cur_chunk_size, K, C * kW,
                &alpha,
                data_col, data_col_type, cur_chunk_size, strideA,
                w_d_actual, w_type_actual, C * kW, strideB,
                &beta,
                (char*)dst_d + ow_start * dst_elem_size, dst_type, OW, strideC,
                N,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT
            ));
        }
    }
#endif

    return true;
}

bool ggml_cuda_op_conv_1d_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    ops_conv_1d_params params;
    if (!ops_extract_conv_1d_params(node, params)) {
        return false;
    }
    return ggml_cuda_op_conv_1d(backend, params.w, params.x, node, params.stride, params.padding, params.dilation);
}

} // namespace cuda
} // namespace ggml_ops_ext
