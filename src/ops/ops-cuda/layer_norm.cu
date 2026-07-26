#include "ops_cuda_common.cuh"

namespace ggml_ops_ext {
namespace cuda {

// Generic block-reduction helper
template <int block_size, typename T>
__global__ void layer_norm_kernel(const T* x, const T* gamma, const T* beta, T* dst, int64_t ne0, int64_t ne1,
                                  int64_t ne2, int64_t ne3, float eps, size_t nb_x0, size_t nb_x1, size_t nb_x2,
                                  size_t nb_x3, size_t nb_gamma0, size_t nb_beta0, size_t nb_dst0, size_t nb_dst1,
                                  size_t nb_dst2, size_t nb_dst3) {
    const int row = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample = blockIdx.z;
    const int tid = threadIdx.x;

    if (sample >= ne3)
        return;

    // Relocate pointers for this row
    const T* row_x = (const T*)((const char*)x + sample * nb_x3 + channel * nb_x2 + row * nb_x1);
    T* row_dst = (T*)((char*)dst + sample * nb_dst3 + channel * nb_dst2 + row * nb_dst1);

    float2 mean_var = make_float2(0.0f, 0.0f);

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ne0; col += block_size) {
        const T* px = (const T*)((const char*)row_x + col * nb_x0);
        float xi = (float)*px;
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    // sum up partial sums using GGML's optimized warp-shuffle block_reduce
    extern __shared__ float2 s_sum2[];
    mean_var = block_reduce<block_reduce_method::SUM, block_size>(mean_var, s_sum2);

    const float mean = mean_var.x / ne0;
    const float var = mean_var.y / ne0 - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ne0; col += block_size) {
        const T* px = (const T*)((const char*)row_x + col * nb_x0);
        const T* pgamma = (const T*)((const char*)gamma + col * nb_gamma0);
        const T* pbeta = (const T*)((const char*)beta + col * nb_beta0);

        T* pdst = (T*)((char*)row_dst + col * nb_dst0);

        *pdst = (T)((((float)*px - mean) * inv_std) * (float)*pgamma + (float)*pbeta);
    }
}

bool ggml_cuda_op_layer_norm(ggml_backend_t backend, struct ggml_tensor* x, struct ggml_tensor* gamma,
                             struct ggml_tensor* beta, struct ggml_tensor* dst, float eps) {
    int device = ggml_ops_ext_bridge_cuda_get_device(backend);
    cudaStream_t stream = (cudaStream_t)ggml_ops_ext_bridge_cuda_get_stream(backend);

    CUDA_CHECK(cudaSetDevice(device));

    int64_t ne0 = dst->ne[0]; // Columns (dimension along which to normalize)
    int64_t ne1 = dst->ne[1]; // Rows
    int64_t ne2 = dst->ne[2]; // Channels
    int64_t ne3 = dst->ne[3]; // Samples

    // grid.x allows 2^31-1 blocks, but grid.y/z are capped at 65535.
    if (ne1 > INT32_MAX || ne2 > 65535 || ne3 > 65535) {
        fprintf(stderr, "CUDA LayerNorm: grid dimensions out of range (%lld, %lld, %lld)\n",
                (long long)ne1, (long long)ne2, (long long)ne3);
        return false;
    }

    const dim3 blocks_num(ne1, ne2, ne3);

    if (x->type == GGML_TYPE_F32) {
        const float* x_d = (const float*)x->data;
        const float* gamma_d = (const float*)gamma->data;
        const float* beta_d = (const float*)beta->data;
        float* dst_d = (float*)dst->data;

        if (ne0 <= 1024) {
            const dim3 block_dims(WARP_SIZE, 1, 1);
            layer_norm_kernel<WARP_SIZE, float><<<blocks_num, block_dims, 0, stream>>>(
                x_d, gamma_d, beta_d, dst_d, ne0, ne1, ne2, ne3, eps, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                gamma->nb[0], beta->nb[0], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        } else {
            const dim3 block_dims(1024, 1, 1);
            layer_norm_kernel<1024, float><<<blocks_num, block_dims, 32 * sizeof(float2), stream>>>(
                x_d, gamma_d, beta_d, dst_d, ne0, ne1, ne2, ne3, eps, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                gamma->nb[0], beta->nb[0], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        }
    } else if (x->type == GGML_TYPE_F16) {
        const half* x_d = (const half*)x->data;
        const half* gamma_d = (const half*)gamma->data;
        const half* beta_d = (const half*)beta->data;
        half* dst_d = (half*)dst->data;

        if (ne0 <= 1024) {
            const dim3 block_dims(WARP_SIZE, 1, 1);
            layer_norm_kernel<WARP_SIZE, half><<<blocks_num, block_dims, 0, stream>>>(
                x_d, gamma_d, beta_d, dst_d, ne0, ne1, ne2, ne3, eps, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                gamma->nb[0], beta->nb[0], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        } else {
            const dim3 block_dims(1024, 1, 1);
            layer_norm_kernel<1024, half><<<blocks_num, block_dims, 32 * sizeof(float2), stream>>>(
                x_d, gamma_d, beta_d, dst_d, ne0, ne1, ne2, ne3, eps, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                gamma->nb[0], beta->nb[0], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        }
    } else {
        fprintf(stderr, "Unsupported data type for CUDA LayerNorm: %d\n", x->type);
        return false;
    }

    return cudaGetLastError() == cudaSuccess;
}

bool ggml_cuda_op_layer_norm_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    float eps;
    std::memcpy(&eps, node->op_params, sizeof(float));
    return ggml_cuda_op_layer_norm(backend, node->src[0], node->src[1], node->src[2], node, eps);
}

// ==================== Fused AdaLN CUDA Operator ====================

__device__ inline size_t get_tensor_offset(int64_t i0, int64_t i1, int64_t i2, int64_t i3, int64_t ne1, int64_t ne2,
                                           int64_t ne3, size_t nb0, size_t nb1, size_t nb2, size_t nb3,
                                           int64_t dst_ne2) {
    int64_t s0 = i0;
    int64_t s1 = 0;
    int64_t s2 = 0;
    int64_t s3 = 0;

    if (ne2 > 1) {
        s2 = i2 % ne2;
        s1 = i1 % ne1;
    } else if (ne1 > 1) {
        if (ne1 == dst_ne2) {
            s1 = i2;
        } else {
            s1 = i1 % ne1;
        }
    }
    if (ne3 > 1) {
        s3 = i3 % ne3;
    }

    return s3 * nb3 + s2 * nb2 + s1 * nb1 + s0 * nb0;
}

template <int block_size, typename T>
__global__ void
ada_ln_kernel(const T* x, const T* scale, const T* shift, T* dst, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
              float eps, size_t nb_x0, size_t nb_x1, size_t nb_x2, size_t nb_x3, int64_t scale_ne1, int64_t scale_ne2,
              int64_t scale_ne3, size_t nb_scale0, size_t nb_scale1, size_t nb_scale2, size_t nb_scale3,
              int64_t shift_ne1, int64_t shift_ne2, int64_t shift_ne3, size_t nb_shift0, size_t nb_shift1,
              size_t nb_shift2, size_t nb_shift3, size_t nb_dst0, size_t nb_dst1, size_t nb_dst2, size_t nb_dst3) {
    const int row = blockIdx.x;     // Sequence row (ne1)
    const int channel = blockIdx.y; // Batch dimension 1 (ne2)
    const int sample = blockIdx.z;  // Batch dimension 2 (ne3)
    const int tid = threadIdx.x;

    if (sample >= ne3)
        return;

    // Relocate pointers for this row
    const T* row_x = (const T*)((const char*)x + sample * nb_x3 + channel * nb_x2 + row * nb_x1);
    T* row_dst = (T*)((char*)dst + sample * nb_dst3 + channel * nb_dst2 + row * nb_dst1);

    float2 mean_var = make_float2(0.0f, 0.0f);

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ne0; col += block_size) {
        const T* px = (const T*)((const char*)row_x + col * nb_x0);
        float xi = (float)*px;
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    // Sum up partial sums using block_reduce
    extern __shared__ float2 s_sum2[];
    mean_var = block_reduce<block_reduce_method::SUM, block_size>(mean_var, s_sum2);

    const float mean = mean_var.x / ne0;
    const float var = mean_var.y / ne0 - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ne0; col += block_size) {
        const T* px = (const T*)((const char*)row_x + col * nb_x0);

        size_t scale_offset = get_tensor_offset(col, row, channel, sample, scale_ne1, scale_ne2, scale_ne3, nb_scale0,
                                                nb_scale1, nb_scale2, nb_scale3, ne2);
        size_t shift_offset = get_tensor_offset(col, row, channel, sample, shift_ne1, shift_ne2, shift_ne3, nb_shift0,
                                                nb_shift1, nb_shift2, nb_shift3, ne2);

        const T* pscale = (const T*)((const char*)scale + scale_offset);
        const T* pshift = (const T*)((const char*)shift + shift_offset);

        T* pdst = (T*)((char*)row_dst + col * nb_dst0);

        *pdst = (T)((((float)*px - mean) * inv_std) * (1.0f + (float)*pscale) + (float)*pshift);
    }
}

bool ggml_cuda_op_ada_ln(ggml_backend_t backend, struct ggml_tensor* x, struct ggml_tensor* scale,
                         struct ggml_tensor* shift, struct ggml_tensor* dst, float eps) {
    int device = ggml_ops_ext_bridge_cuda_get_device(backend);
    cudaStream_t stream = (cudaStream_t)ggml_ops_ext_bridge_cuda_get_stream(backend);

    CUDA_CHECK(cudaSetDevice(device));

    int64_t ne0 = dst->ne[0]; // Columns
    int64_t ne1 = dst->ne[1]; // Rows
    int64_t ne2 = dst->ne[2]; // Channels
    int64_t ne3 = dst->ne[3]; // Samples

    // grid.x allows 2^31-1 blocks, but grid.y/z are capped at 65535.
    if (ne1 > INT32_MAX || ne2 > 65535 || ne3 > 65535) {
        fprintf(stderr, "CUDA AdaLN: grid dimensions out of range (%lld, %lld, %lld)\n",
                (long long)ne1, (long long)ne2, (long long)ne3);
        return false;
    }

    const dim3 blocks_num(ne1, ne2, ne3);

    if (x->type == GGML_TYPE_F32) {
        const float* x_d = (const float*)x->data;
        const float* scale_d = (const float*)scale->data;
        const float* shift_d = (const float*)shift->data;
        float* dst_d = (float*)dst->data;

        if (ne0 <= 1024) {
            const dim3 block_dims(WARP_SIZE, 1, 1);
            ada_ln_kernel<WARP_SIZE, float><<<blocks_num, block_dims, 0, stream>>>(
                x_d, scale_d, shift_d, dst_d, ne0, ne1, ne2, ne3, eps, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                scale->ne[1], scale->ne[2], scale->ne[3], scale->nb[0], scale->nb[1], scale->nb[2], scale->nb[3],
                shift->ne[1], shift->ne[2], shift->ne[3], shift->nb[0], shift->nb[1], shift->nb[2], shift->nb[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        } else {
            const dim3 block_dims(1024, 1, 1);
            ada_ln_kernel<1024, float><<<blocks_num, block_dims, 32 * sizeof(float2), stream>>>(
                x_d, scale_d, shift_d, dst_d, ne0, ne1, ne2, ne3, eps, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                scale->ne[1], scale->ne[2], scale->ne[3], scale->nb[0], scale->nb[1], scale->nb[2], scale->nb[3],
                shift->ne[1], shift->ne[2], shift->ne[3], shift->nb[0], shift->nb[1], shift->nb[2], shift->nb[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        }
    } else if (x->type == GGML_TYPE_F16) {
        const half* x_d = (const half*)x->data;
        const half* scale_d = (const half*)scale->data;
        const half* shift_d = (const half*)shift->data;
        half* dst_d = (half*)dst->data;

        if (ne0 <= 1024) {
            const dim3 block_dims(WARP_SIZE, 1, 1);
            ada_ln_kernel<WARP_SIZE, half><<<blocks_num, block_dims, 0, stream>>>(
                x_d, scale_d, shift_d, dst_d, ne0, ne1, ne2, ne3, eps, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                scale->ne[1], scale->ne[2], scale->ne[3], scale->nb[0], scale->nb[1], scale->nb[2], scale->nb[3],
                shift->ne[1], shift->ne[2], shift->ne[3], shift->nb[0], shift->nb[1], shift->nb[2], shift->nb[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        } else {
            const dim3 block_dims(1024, 1, 1);
            ada_ln_kernel<1024, half><<<blocks_num, block_dims, 32 * sizeof(float2), stream>>>(
                x_d, scale_d, shift_d, dst_d, ne0, ne1, ne2, ne3, eps, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                scale->ne[1], scale->ne[2], scale->ne[3], scale->nb[0], scale->nb[1], scale->nb[2], scale->nb[3],
                shift->ne[1], shift->ne[2], shift->ne[3], shift->nb[0], shift->nb[1], shift->nb[2], shift->nb[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        }
    } else {
        fprintf(stderr, "Unsupported data type for CUDA AdaLN: %d\n", x->type);
        return false;
    }

    return cudaGetLastError() == cudaSuccess;
}

bool ggml_cuda_op_ada_ln_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    float eps;
    std::memcpy(&eps, node->op_params, sizeof(float));
    return ggml_cuda_op_ada_ln(backend, node->src[0], node->src[1], node->src[2], node, eps);
}

} // namespace cuda
} // namespace ggml_ops_ext
