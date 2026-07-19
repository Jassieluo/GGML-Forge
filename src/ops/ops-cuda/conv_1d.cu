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
__global__ void depthwise_conv_1d_kernel(
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
            int64_t iw = ow * stride - padding + ik * dilation;
            if (iw >= 0 && iw < W) {
                const T* px = (const T*)((const char*)x + n * nb_x2 + c * nb_x1 + iw * nb_x0);
                const T* pw = (const T*)((const char*)w + c * nb_w1 + ik * nb_w0);
                sum += to_float(*px) * to_float(*pw);
            }
        }
        T* pdst = (T*)((char*)dst + n * nb_dst2 + c * nb_dst1 + ow * nb_dst0);
        *pdst = (T)sum;
    }
}

bool ggml_cuda_op_conv_1d(
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
    if (!ops_describe_conv_weight(GGML_OP_OPS_VIRT_CONV_1D, w, x, groups, weight_desc)) return false;
    const int64_t N = x->ne[2];
    const int64_t C = x->ne[1];
    const int64_t W = x->ne[0];
    const int64_t K = weight_desc.output_channels;
    const int64_t kW = weight_desc.kernel;
    const int64_t OW = dst->ne[0];

    if (ggml_is_quantized(w->type)) {
        const int bias_type = bias && bias->type == GGML_TYPE_F16 ? 1 : 0;
        const int64_t total = N * K * OW;
        constexpr int block_size = 256;
        const int grid_size = static_cast<int>((total * 32 + block_size - 1) / block_size);
        constexpr int time_tile = 8;
        constexpr int warps_per_block = block_size / 32;
        const bool use_cached_input = groups == 1 && C * time_tile <= 8192;
        const int64_t time_tiles = (OW + time_tile - 1) / time_tile;
        const int64_t channel_tiles = (K + warps_per_block - 1) / warps_per_block;
        const int cached_grid_size = static_cast<int>(N * channel_tiles * time_tiles);
        const size_t cached_shared_bytes = static_cast<size_t>(C * time_tile) * sizeof(float);
#define LAUNCH_DIRECT_QUANT_CONV(weight_type, value_type) \
        do { \
            if (use_cached_input) { \
                quantized_conv_1d_cached_input_kernel<weight_type, value_type, time_tile> \
                    <<<cached_grid_size, block_size, cached_shared_bytes, stream>>>( \
                        w->data, static_cast<const value_type*>(x->data), bias ? bias->data : nullptr, bias_type, \
                        static_cast<value_type*>(dst->data), W, OW, C, K, kW, N, stride, padding, dilation, \
                        x->nb[0], x->nb[1], x->nb[2], dst->nb[0], dst->nb[1], dst->nb[2]); \
            } else { \
                quantized_conv_1d_direct_kernel<weight_type, value_type><<<grid_size, block_size, 0, stream>>>( \
                    w->data, static_cast<const value_type*>(x->data), bias ? bias->data : nullptr, bias_type, \
                    static_cast<value_type*>(dst->data), W, OW, C, K, kW, N, stride, padding, dilation, groups, \
                    x->nb[0], x->nb[1], x->nb[2], dst->nb[0], dst->nb[1], dst->nb[2]); \
            } \
        } while (false)
        if (x->type == GGML_TYPE_F16) {
            switch (w->type) {
                case GGML_TYPE_Q4_0: LAUNCH_DIRECT_QUANT_CONV(GGML_TYPE_Q4_0, half); break;
                case GGML_TYPE_Q4_K: LAUNCH_DIRECT_QUANT_CONV(GGML_TYPE_Q4_K, half); break;
                case GGML_TYPE_Q8_0: LAUNCH_DIRECT_QUANT_CONV(GGML_TYPE_Q8_0, half); break;
                default: return false;
            }
        } else {
            switch (w->type) {
                case GGML_TYPE_Q4_0: LAUNCH_DIRECT_QUANT_CONV(GGML_TYPE_Q4_0, float); break;
                case GGML_TYPE_Q4_K: LAUNCH_DIRECT_QUANT_CONV(GGML_TYPE_Q4_K, float); break;
                case GGML_TYPE_Q8_0: LAUNCH_DIRECT_QUANT_CONV(GGML_TYPE_Q8_0, float); break;
                default: return false;
            }
        }
#undef LAUNCH_DIRECT_QUANT_CONV
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

    bool is_depthwise = (groups > 1 && weight_desc.input_channels_per_group == 1 && K == groups);
    if (is_depthwise) {
        int64_t total_elements = N * C * OW;
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;
        size_t w_actual_elem_size = (w_type_actual == CUDA_R_16F) ? sizeof(half) : sizeof(float);
        size_t w_stride0 = w_actual_elem_size;
        size_t w_stride2 = kW * w_actual_elem_size;

        if (x->type == GGML_TYPE_F16) {
            depthwise_conv_1d_kernel<<<grid_size, block_size, 0, stream>>>(
                (const half*)x_d, (const half*)w_d_actual, (half*)dst_d,
                C, W, OW, kW,
                stride, padding, dilation,
                N,
                x->nb[0], x->nb[1], x->nb[2],
                w_stride0, w_stride2,
                dst->nb[0], dst->nb[1], dst->nb[2]
            );
        } else {
            depthwise_conv_1d_kernel<<<grid_size, block_size, 0, stream>>>(
                (const float*)x_d, (const float*)w_d_actual, (float*)dst_d,
                C, W, OW, kW,
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
            // 1x1 Convolution Shortcut using strided batched GEMM to write directly to swapped layout
            long long int strideA = C * OW;
            long long int strideB = 0;
            long long int strideC = K * OW;

            CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                cublas,
                CUBLAS_OP_N, CUBLAS_OP_N,
                OW, K, C,
                &alpha,
                x_d, x_type, OW, strideA,
                w_d_actual, w_type_actual, C, strideB,
                &beta,
                dst_d, dst_type, OW, strideC,
                N,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT
            ));
        } else {
            // Chunked GEMM convolution to balance VRAM usage and speed
            const int64_t CHUNK_SIZE = 2048;
            cudaDataType_t data_col_type = (x->type == GGML_TYPE_F16) ? CUDA_R_16F : CUDA_R_32F;
            size_t data_col_elem_size = (x->type == GGML_TYPE_F16) ? sizeof(half) : sizeof(float);
            
            int64_t C_in_group = C / groups;
            int64_t C_out_group = K / groups;

            ops_cuda_alloc<uint8_t> data_col_alloc(stream);
            data_col_alloc.alloc(N * C_in_group * kW * CHUNK_SIZE * data_col_elem_size);
            void* data_col = data_col_alloc.get();
            size_t w_actual_elem_size = (w_type_actual == CUDA_R_16F) ? sizeof(half) : sizeof(float);
            size_t w_channel_stride_bytes = C_in_group * kW * w_actual_elem_size;

            for (int g = 0; g < groups; ++g) {
                const void* x_d_g = (const char*)x_d + g * C_in_group * x->nb[1];
                const void* w_d_g = (const char*)w_d_actual + g * C_out_group * w_channel_stride_bytes;
                void* dst_d_g = (char*)dst_d + g * C_out_group * dst->nb[1];

                for (int64_t ow_start = 0; ow_start < OW; ow_start += CHUNK_SIZE) {
                    int64_t cur_chunk_size = min(CHUNK_SIZE, OW - ow_start);

                    // Launch chunked im2col kernel
                    int64_t total_elements = N * C_in_group * kW * cur_chunk_size;
                    int block_size = 256;
                    int grid_size = (total_elements + block_size - 1) / block_size;
                    
                    if (x->type == GGML_TYPE_F16) {
                        im2col_1d_kernel_chunked<half, half><<<grid_size, block_size, 0, stream>>>(
                            (const half*)x_d_g, (half*)data_col,
                            C_in_group, W, OW, kW,
                            stride, padding, dilation,
                            N,
                            x->nb[0], x->nb[1], x->nb[2],
                            ow_start, cur_chunk_size
                        );
                    } else {
                        im2col_1d_kernel_chunked<float, float><<<grid_size, block_size, 0, stream>>>(
                            (const float*)x_d_g, (float*)data_col,
                            C_in_group, W, OW, kW,
                            stride, padding, dilation,
                            N,
                            x->nb[0], x->nb[1], x->nb[2],
                            ow_start, cur_chunk_size
                        );
                    }

                    // Call cublasGemmStridedBatchedEx for all batch elements at once, writing directly to swapped layout
                    long long int strideA = C_in_group * kW * cur_chunk_size;
                    long long int strideB = 0;
                    long long int strideC = K * OW;

                    if (getenv("GPT_SOVITS_DEBUG_CONV")) {
                        printf("[DEBUG CONV] N=%d, groups=%d, C=%d, K=%d, W=%d, OW=%d, kW=%d\n",
                               (int)N, (int)groups, (int)C, (int)K, (int)W, (int)OW, (int)kW);
                        printf("             C_in_group=%d, C_out_group=%d, cur_chunk_size=%d\n",
                               (int)C_in_group, (int)C_out_group, (int)cur_chunk_size);
                        printf("             strideA=%lld, strideB=%lld, strideC=%lld\n",
                               (long long)strideA, (long long)strideB, (long long)strideC);
                        printf("             data_col=%p, w_d_g=%p, dst_d_g=%p, offset=%lld, dst_elem_size=%d\n",
                               data_col, w_d_g, dst_d_g, (long long)(ow_start * dst_elem_size), (int)dst_elem_size);
                        printf("             x_d_g=%p, x->nb[0]=%zu, x->nb[1]=%zu, x->nb[2]=%zu\n",
                               x_d_g, x->nb[0], x->nb[1], x->nb[2]);
                        printf("             data_col_type=%d, w_type_actual=%d, dst_type=%d\n",
                               (int)data_col_type, (int)w_type_actual, (int)dst_type);
                    }

                    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                        cublas,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        cur_chunk_size, C_out_group, C_in_group * kW,
                        &alpha,
                        data_col, data_col_type, cur_chunk_size, strideA,
                        w_d_g, w_type_actual, C_in_group * kW, strideB,
                        &beta,
                        (char*)dst_d_g + ow_start * dst_elem_size, dst_type, OW, strideC,
                        N,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT
                    ));
                }
            }
        }
    }
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

bool ggml_cuda_op_conv_1d_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    ops_conv_1d_params params;
    if (!ops_extract_conv_1d_params(node, params)) {
        return false;
    }
    return ggml_cuda_op_conv_1d(backend, params.w, params.x, params.bias, node, params.stride, params.padding, params.dilation, params.groups);
}

} // namespace cuda
} // namespace ggml_ops_ext
