#include "ops_cuda_common.cuh"
#include "quantized_conv.cuh"

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

__device__ inline float load_val(const void* ptr, int64_t idx, int type) {
    if (type == 0) { // GGML_TYPE_F32
        return ((const float*)ptr)[idx];
    } else { // GGML_TYPE_F16
        return __half2float(((const half*)ptr)[idx]);
    }
}

template <typename T>
__global__ void add_bias_1d_kernel(
    T* dst, const void* bias, int bias_type, int64_t ne0, int64_t ne1, int64_t ne2,
    size_t nb0, size_t nb1, size_t nb2
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = ne0 * ne1 * ne2;
    if (idx < total) {
        int64_t i0 = idx % ne0;
        int64_t tmp = idx / ne0;
        int64_t i1 = tmp % ne1; // channel
        int64_t i2 = tmp / ne1; // batch

        T* pdst = (T*)((char*)dst + i2*nb2 + i1*nb1 + i0*nb0);
        float b_val = load_val(bias, i1, bias_type);
        *pdst = (T)(to_float(*pdst) + b_val);
    }
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

template <typename T>
__global__ void depthwise_conv_transpose_1d_kernel(
    const T* x, const T* w, T* dst,
    int64_t C, int64_t W, int64_t OW, int64_t kW,
    int stride, int padding, int dilation,
    int64_t N,
    size_t nb_x0, size_t nb_x1, size_t nb_x2,
    size_t nb_w0, size_t nb_w1,
    size_t nb_dst0, size_t nb_dst1, size_t nb_dst2
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * C * OW;
    if (idx < total) {
        int64_t ow = idx % OW;
        int64_t tmp = idx / OW;
        int64_t c = tmp % C;
        int64_t n = tmp / C;

        float sum = 0.0f;
        for (int64_t ik = 0; ik < kW; ++ik) {
            int64_t val = ow + padding - ik * dilation;
            if (val >= 0 && val % stride == 0) {
                int64_t iw = val / stride;
                if (iw >= 0 && iw < W) {
                    const T* px = (const T*)((const char*)x + n * nb_x2 + c * nb_x1 + iw * nb_x0);
                    const T* pw = (const T*)((const char*)w + c * nb_w1 + ik * nb_w0);
                    sum += to_float(*px) * to_float(*pw);
                }
            }
        }
        T* pdst = (T*)((char*)dst + n * nb_dst2 + c * nb_dst1 + ow * nb_dst0);
        *pdst = (T)sum;
    }
}

bool ggml_cuda_op_conv_transpose_1d(
    ggml_backend_t backend,
    struct ggml_tensor* w,
    struct ggml_tensor* x,
    struct ggml_tensor* bias,
    struct ggml_tensor* node,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    struct ggml_tensor* dst = node;
    int device = ggml_ops_ext_bridge_cuda_get_device(backend);
    cudaStream_t stream = (cudaStream_t)ggml_ops_ext_bridge_cuda_get_stream(backend);

    // Set CUDA device
    CUDA_CHECK(cudaSetDevice(device));

    ops_conv_weight_desc weight_desc = {};
    if (!ops_describe_conv_weight(GGML_OP_OPS_VIRT_CONV_TRANSPOSE_1D, w, x, groups, weight_desc)) return false;
    const int kW = (int)weight_desc.kernel;
    const int C = (int)weight_desc.output_channels_per_group;
    const int K = (int)x->ne[1];
    const int W = (int)x->ne[0];
    const int N = (int)x->ne[2];
    const int OW = (int)dst->ne[0];

    if (ggml_is_quantized(w->type)) {
        const int bias_type = bias && bias->type == GGML_TYPE_F16 ? 1 : 0;
        const int64_t total = static_cast<int64_t>(N) * C * groups * OW;
        constexpr int block_size = 256;
        const bool use_scalar = K / groups <= 8;
        const int grid_size = static_cast<int>((total + block_size - 1) / block_size);
        constexpr int time_tile = 8;
        constexpr int warps_per_block = block_size / 32;
        const int64_t time_tiles = (OW + time_tile - 1) / time_tile;
        const int64_t channel_tiles = (C * groups + warps_per_block - 1) / warps_per_block;
        const int tiled_grid_size = static_cast<int>(N * channel_tiles * time_tiles);
#define LAUNCH_DIRECT_QUANT_CONVT(weight_type, value_type) \
        do { \
            if (use_scalar) { \
                quantized_conv_transpose_1d_scalar_kernel<weight_type, value_type><<<grid_size, block_size, 0, stream>>>( \
                    w->data, static_cast<const value_type*>(x->data), bias ? bias->data : nullptr, bias_type, \
                    static_cast<value_type*>(dst->data), W, OW, K, C, kW, N, stride, padding, dilation, groups, \
                    x->nb[0], x->nb[1], x->nb[2], dst->nb[0], dst->nb[1], dst->nb[2]); \
            } else { \
                quantized_conv_transpose_1d_time_tile_kernel<weight_type, value_type, time_tile> \
                    <<<tiled_grid_size, block_size, 0, stream>>>( \
                    w->data, static_cast<const value_type*>(x->data), bias ? bias->data : nullptr, bias_type, \
                    static_cast<value_type*>(dst->data), W, OW, K, C, kW, N, stride, padding, dilation, groups, \
                    x->nb[0], x->nb[1], x->nb[2], dst->nb[0], dst->nb[1], dst->nb[2]); \
            } \
        } while (false)
        if (x->type == GGML_TYPE_F16) {
            switch (w->type) {
                case GGML_TYPE_Q4_0: LAUNCH_DIRECT_QUANT_CONVT(GGML_TYPE_Q4_0, half); break;
                case GGML_TYPE_Q4_K: LAUNCH_DIRECT_QUANT_CONVT(GGML_TYPE_Q4_K, half); break;
                case GGML_TYPE_Q8_0: LAUNCH_DIRECT_QUANT_CONVT(GGML_TYPE_Q8_0, half); break;
                default: return false;
            }
        } else {
            switch (w->type) {
                case GGML_TYPE_Q4_0: LAUNCH_DIRECT_QUANT_CONVT(GGML_TYPE_Q4_0, float); break;
                case GGML_TYPE_Q4_K: LAUNCH_DIRECT_QUANT_CONVT(GGML_TYPE_Q4_K, float); break;
                case GGML_TYPE_Q8_0: LAUNCH_DIRECT_QUANT_CONVT(GGML_TYPE_Q8_0, float); break;
                default: return false;
            }
        }
#undef LAUNCH_DIRECT_QUANT_CONVT
        return cudaGetLastError() == cudaSuccess;
    }

    ggml_type w_storage_type = w->type;
    const void* w_d = w->data;

    cudaDataType_t w_type = CUDA_R_32F;
    if (w_storage_type == GGML_TYPE_F16) {
        w_type = CUDA_R_16F;
    } else if (w_storage_type == GGML_TYPE_BF16) {
        w_type = CUDA_R_16BF;
    } else if (w_storage_type != GGML_TYPE_F32) {
        return false;
    }

    const void* x_d = x->data;
    void* dst_d = dst->data;

    cudaDataType_t x_type = (x->type == GGML_TYPE_F16) ? CUDA_R_16F : CUDA_R_32F;
    cudaDataType_t dst_type = (dst->type == GGML_TYPE_F16) ? CUDA_R_16F : CUDA_R_32F;
    size_t x_elem_size = (x->type == GGML_TYPE_F16) ? sizeof(half) : sizeof(float);
    size_t dst_elem_size = (dst->type == GGML_TYPE_F16) ? sizeof(half) : sizeof(float);

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

    // dx = output
    cudnnDataType_t cudnn_dst_type = (dst->type == GGML_TYPE_F16) ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;
    int nStrideY = (int)(dst->nb[2] / dst_elem_size);
    int cStrideY = (int)(dst->nb[1] / dst_elem_size);
    int hStrideY = (int)OW;
    int wStrideY = (int)(dst->nb[0] / dst_elem_size);
    CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(dx_desc, cudnn_dst_type,
                                             N, dst->ne[1], 1, OW,
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
    if (w_storage_type == GGML_TYPE_F16) {
        cudnn_w_type = CUDNN_DATA_HALF;
    } else if (w_storage_type == GGML_TYPE_BF16) {
        cudnn_w_type = CUDNN_DATA_BFLOAT16;
    }

    CUDNN_CHECK(cudnnSetFilter4dDescriptor(w_desc, cudnn_w_type, CUDNN_TENSOR_NCHW,
                                           K, C, 1, kW));

    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
                                                 0 /*pad_h*/, padding /*pad_w*/,
                                                 1 /*stride_h*/, stride /*stride_w*/,
                                                 1 /*dilation_h*/, dilation /*dilation_w*/,
                                                 CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
#if CUDNN_MAJOR >= 7
    CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc, groups));
#endif

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
    // Ensure weights have the same precision as activations (x->type) for cuBLAS
    const void* w_d_actual = w_d;
    cudaDataType_t w_type_actual = w_type;
    ops_cuda_alloc<float> w_f32_alloc(stream);
    ops_cuda_alloc<half> w_f16_alloc(stream);

    if (w_storage_type != x->type) {
        int64_t w_len = ggml_nelements(w);
        if (x->type == GGML_TYPE_F32) {
            w_f32_alloc.alloc(w_len);
            cast_tensor_cuda(w_d, w_f32_alloc.get(), w_storage_type, GGML_TYPE_F32, w_len, stream);
            w_d_actual = w_f32_alloc.get();
            w_type_actual = CUDA_R_32F;
        } else if (x->type == GGML_TYPE_F16) {
            w_f16_alloc.alloc(w_len);
            cast_tensor_cuda(w_d, w_f16_alloc.get(), w_storage_type, GGML_TYPE_F16, w_len, stream);
            w_d_actual = w_f16_alloc.get();
            w_type_actual = CUDA_R_16F;
        }
    }

    bool is_depthwise = (groups > 1 && C == 1 && K == groups);
    if (is_depthwise) {
        int64_t total_elements = N * groups * OW;
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;
        size_t w_actual_elem_size = (w_type_actual == CUDA_R_16F) ? sizeof(half) : sizeof(float);
        size_t w_stride0 = w_actual_elem_size;
        size_t w_stride2 = kW * w_actual_elem_size;

        if (x->type == GGML_TYPE_F16) {
            depthwise_conv_transpose_1d_kernel<<<grid_size, block_size, 0, stream>>>(
                (const half*)x_d, (const half*)w_d_actual, (half*)dst_d,
                groups, W, OW, kW,
                stride, padding, dilation,
                N,
                x->nb[0], x->nb[1], x->nb[2],
                w_stride0, w_stride2,
                dst->nb[0], dst->nb[1], dst->nb[2]
            );
        } else {
            depthwise_conv_transpose_1d_kernel<<<grid_size, block_size, 0, stream>>>(
                (const float*)x_d, (const float*)w_d_actual, (float*)dst_d,
                groups, W, OW, kW,
                stride, padding, dilation,
                N,
                x->nb[0], x->nb[1], x->nb[2],
                w_stride0, w_stride2,
                dst->nb[0], dst->nb[1], dst->nb[2]
            );
        }
        return true;
    } else {
        // Get cuBLAS handle from backend context and set stream
        cublasHandle_t cublas = (cublasHandle_t)ggml_ops_ext_bridge_cuda_get_cublas(backend);
        CUBLAS_CHECK(cublasSetStream(cublas, stream));

        bool is_1x1 = (kW == 1 && stride == 1 && padding == 0 && dilation == 1 && groups == 1);
        float alpha = 1.0f;
        float beta = 0.0f;

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
            CUDA_CHECK(cudaMemsetAsync(dst_d, 0, N * C * groups * OW * dst_elem_size, stream));

            const int64_t CHUNK_SIZE = 2048;
            cudaDataType_t data_col_type = (x->type == GGML_TYPE_F16) ? CUDA_R_16F : CUDA_R_32F;
            size_t data_col_elem_size = (x->type == GGML_TYPE_F16) ? sizeof(half) : sizeof(float);
            
            int64_t C_out_group = C;
            int64_t C_in_group = K / groups;

            ops_cuda_alloc<uint8_t> data_col_alloc(stream);
            data_col_alloc.alloc(N * C_out_group * kW * CHUNK_SIZE * data_col_elem_size);
            void* data_col = data_col_alloc.get();
            size_t w_actual_elem_size = (w_type_actual == CUDA_R_16F) ? sizeof(half) : sizeof(float);
            size_t w_channel_stride_bytes = C_out_group * kW * w_actual_elem_size;

            for (int g = 0; g < groups; ++g) {
                const void* x_d_g = (const char*)x_d + g * C_in_group * x->nb[1];
                const void* w_d_g = (const char*)w_d_actual + g * C_in_group * w_channel_stride_bytes;
                void* dst_d_g = (char*)dst_d + g * C_out_group * dst->nb[1];

                for (int64_t w_start = 0; w_start < W; w_start += CHUNK_SIZE) {
                    int64_t cur_chunk_size = min(CHUNK_SIZE, W - w_start);

                    // Call cublasGemmStridedBatchedEx for all batch elements at once
                    long long int strideA = 0;
                    long long int strideB = K * W;
                    long long int strideC = C_out_group * kW * cur_chunk_size;

                    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                        cublas,
                        CUBLAS_OP_N, CUBLAS_OP_T,
                        C_out_group * kW, cur_chunk_size, C_in_group,
                        &alpha,
                        w_d_g, w_type_actual, C_out_group * kW, strideA,
                        (const char*)x_d_g + w_start * x->nb[0], x_type, W, strideB,
                        &beta,
                        data_col, data_col_type, C_out_group * kW, strideC,
                        N,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT
                    ));

                    // Launch chunked col2im kernel to accumulate to dst_d_g
                    int64_t total_elements = N * C_out_group * kW * cur_chunk_size;
                    int block_size = 256;
                    int grid_size = (total_elements + block_size - 1) / block_size;
                    
                    if (x->type == GGML_TYPE_F16) {
                        col2im_1d_kernel_chunked<half, half><<<grid_size, block_size, 0, stream>>>(
                            (const half*)data_col, (half*)dst_d_g,
                            C_out_group, W, OW, kW,
                            stride, padding, dilation,
                            N,
                            dst->nb[0], dst->nb[1], dst->nb[2],
                            w_start, cur_chunk_size
                        );
                    } else {
                        col2im_1d_kernel_chunked<float, float><<<grid_size, block_size, 0, stream>>>(
                            (const float*)data_col, (float*)dst_d_g,
                            C_out_group, W, OW, kW,
                            stride, padding, dilation,
                            N,
                            dst->nb[0], dst->nb[1], dst->nb[2],
                            w_start, cur_chunk_size
                        );
                    }
                }
            }
        }
    }
#endif

    if (bias != nullptr) {
        int bias_type = (bias->type == GGML_TYPE_F32) ? 0 : 1;
        int64_t total = dst->ne[0] * dst->ne[1] * dst->ne[2];
        int block_size = 256;
        int grid_size = (total + block_size - 1) / block_size;
        if (dst->type == GGML_TYPE_F32) {
            add_bias_1d_kernel<float><<<grid_size, block_size, 0, stream>>>(
                (float*)dst_d, bias->data, bias_type, dst->ne[0], dst->ne[1], dst->ne[2],
                dst->nb[0], dst->nb[1], dst->nb[2]
            );
        } else if (dst->type == GGML_TYPE_F16) {
            add_bias_1d_kernel<half><<<grid_size, block_size, 0, stream>>>(
                (half*)dst_d, bias->data, bias_type, dst->ne[0], dst->ne[1], dst->ne[2],
                dst->nb[0], dst->nb[1], dst->nb[2]
            );
        }
    }

    return true;
}

bool ggml_cuda_op_conv_transpose_1d_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    ops_conv_transpose_1d_params params;
    if (!ops_extract_conv_transpose_1d_params(node, params)) {
        return false;
    }
    return ggml_cuda_op_conv_transpose_1d(backend, params.w, params.x, params.bias, node, params.stride, params.padding, params.dilation, params.groups);
}

} // namespace cuda
} // namespace ggml_ops_ext
