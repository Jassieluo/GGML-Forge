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

template <typename T>
__device__ inline void atomic_add(T* address, float val) {
    atomicAdd(address, val);
}

#if defined(__CUDACC__) && __CUDA_ARCH__ >= 700
__device__ inline void atomic_add(half* address, float val) {
    atomicAdd(address, __float2half(val));
}
#endif

#ifndef GGML_USE_CUDNN
template <typename T_in, typename T_out>
__global__ void col2im_1d_kernel_chunked(
    const T_in* data_col, T_out* dst,
    int64_t C, int64_t W, int64_t OW, int64_t kW,
    int stride, int padding, int dilation,
    int64_t N,
    size_t nb_dst0, size_t nb_dst1, size_t nb_dst2,
    int64_t iw_start, int64_t cur_chunk_size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * C * kW * cur_chunk_size;
    if (idx < total) {
        int64_t ik = idx % kW;
        int64_t tmp = idx / kW;
        int64_t c = tmp % C;
        tmp = tmp / C;
        int64_t iw_offset = tmp % cur_chunk_size;
        int64_t n = tmp / cur_chunk_size;

        int64_t iw = iw_start + iw_offset;
        int64_t ow = iw * stride - padding + ik * dilation;
        if (ow >= 0 && ow < OW) {
            float val = to_float(data_col[idx]);
            T_out* pdst = (T_out*)((char*)dst + n * nb_dst2 + c * nb_dst1 + ow * nb_dst0);
            atomic_add(pdst, val);
        }
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

bool ggml_cuda_op_conv_transpose_1d(
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

    const int kW = (int)w->ne[0]; // kernel_size
    const int C  = (int)w->ne[1]; // out_channels (C in weight notation)
    const int K  = (int)w->ne[2]; // in_channels (K in weight notation)
    const int W  = (int)x->ne[0]; // input seq_len
    const int N  = (int)x->ne[2]; // batch
    const int OW = (int)dst->ne[0]; // output seq_len

#ifdef GGML_USE_CUDNN
    // Get or create cuDNN handle and set stream
    cudnnHandle_t cudnn = get_cudnn_handle(device);
    CUDNN_CHECK(cudnnSetStream(cudnn, stream));

    // Create cuDNN descriptors
    cudnnTensorDescriptor_t dy_desc, dx_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dy_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dx_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    size_t x_elem_size = (x->type == GGML_TYPE_F16) ? sizeof(half) : sizeof(float);
    size_t dst_elem_size = (dst->type == GGML_TYPE_F16) ? sizeof(half) : sizeof(float);

    // dx = output
    cudnnDataType_t cudnn_dst_type = (dst->type == GGML_TYPE_F16) ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;
    int nStrideY = (int)(dst->nb[2] / dst_elem_size);
    int cStrideY = (int)(dst->nb[1] / dst_elem_size);
    int hStrideY = (int)OW;
    int wStrideY = (int)(dst->nb[0] / dst_elem_size);
    CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(dx_desc, cudnn_dst_type,
                                             N, C, 1, OW,
                                             nStrideY, cStrideY, hStrideY, wStrideY));
    // dy = input
    cudnnDataType_t cudnn_x_type = (x->type == GGML_TYPE_F16) ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;
    int nStrideX = (int)(x->nb[2] / x_elem_size);
    int cStrideX = (int)(x->nb[1] / x_elem_size);
    int hStrideX = (int)W;
    int wStrideX = (int)(x->nb[0] / x_elem_size);
    CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(dy_desc, cudnn_x_type,
                                             N, K, 1, W,
                                             nStrideX, cStrideX, hStrideX, wStrideX));
    
    // Filter descriptor
    cudnnDataType_t cudnn_w_type = CUDNN_DATA_FLOAT;
    if (w->type == GGML_TYPE_F16) {
        cudnn_w_type = CUDNN_DATA_HALF;
    } else if (w->type == GGML_TYPE_BF16) {
        cudnn_w_type = CUDNN_DATA_BFLOAT16;
    }

    CUDNN_CHECK(cudnnSetFilter4dDescriptor(w_desc, cudnn_w_type, CUDNN_TENSOR_NCHW,
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
        // 1x1 Transposed Convolution Shortcut: direct GEMM into destination, 0-Workspace
        for (int64_t n = 0; n < N; ++n) {
            CUBLAS_CHECK(cublasGemmEx(
                cublas,
                CUBLAS_OP_N, CUBLAS_OP_T,
                C, W, K,
                &alpha,
                w_d_actual, w_type_actual, C,
                (const char*)x_d + n * (K * W * x_elem_size), x_type, W,
                &beta,
                (char*)dst_d + n * (C * W * dst_elem_size), dst_type, C,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT
            ));
        }
    } else {
        // Chunked GEMM transposed convolution to balance VRAM usage and speed
        // Clear destination buffer first since we accumulate using atomicAdd
        CUDA_CHECK(cudaMemsetAsync(dst_d, 0, N * C * OW * dst_elem_size, stream));

        const int64_t CHUNK_SIZE = 2048;
        cudaDataType_t data_col_type = (x->type == GGML_TYPE_F16) ? CUDA_R_16F : CUDA_R_32F;
        size_t data_col_elem_size = (x->type == GGML_TYPE_F16) ? sizeof(half) : sizeof(float);
        
        void* data_col = get_cuda_workspace(device, N * C * kW * CHUNK_SIZE * data_col_elem_size, stream);

        for (int64_t w_start = 0; w_start < W; w_start += CHUNK_SIZE) {
            int64_t cur_chunk_size = min(CHUNK_SIZE, W - w_start);

            // Call cublasGemmStridedBatchedEx for all batch elements at once
            long long int strideA = 0;
            long long int strideB = K * W;
            long long int strideC = C * kW * cur_chunk_size;

            CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                cublas,
                CUBLAS_OP_N, CUBLAS_OP_T,
                C * kW, cur_chunk_size, K,
                &alpha,
                w_d_actual, w_type_actual, C * kW, strideA,
                (const char*)x_d + w_start * x_elem_size, x_type, W, strideB,
                &beta,
                data_col, data_col_type, C * kW, strideC,
                N,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT
            ));

            // Launch chunked col2im kernel to accumulate to dst_d
            int64_t total_elements = N * C * kW * cur_chunk_size;
            int block_size = 256;
            int grid_size = (total_elements + block_size - 1) / block_size;
            
            if (x->type == GGML_TYPE_F16) {
                col2im_1d_kernel_chunked<half, half><<<grid_size, block_size, 0, stream>>>(
                    (const half*)data_col, (half*)dst_d,
                    C, W, OW, kW,
                    stride, padding, dilation,
                    N,
                    dst->nb[0], dst->nb[1], dst->nb[2],
                    w_start, cur_chunk_size
                );
            } else {
                col2im_1d_kernel_chunked<float, float><<<grid_size, block_size, 0, stream>>>(
                    (const float*)data_col, (float*)dst_d,
                    C, W, OW, kW,
                    stride, padding, dilation,
                    N,
                    dst->nb[0], dst->nb[1], dst->nb[2],
                    w_start, cur_chunk_size
                );
            }
        }
    }
#endif

    return true;
}

bool ggml_cuda_op_conv_transpose_1d_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    ops_conv_transpose_1d_params params;
    if (!ops_extract_conv_transpose_1d_params(node, params)) {
        return false;
    }
    return ggml_cuda_op_conv_transpose_1d(backend, params.w, params.x, node, params.stride, params.padding, params.dilation);
}

} // namespace cuda
} // namespace ggml_ops_ext
