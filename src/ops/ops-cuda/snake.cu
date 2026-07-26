#include "ops_cuda_common.cuh"

namespace ggml_ops_ext {
namespace cuda {

template <typename T>
__device__ inline T snake_device(T x_val, float alpha) {
    float val = (float)x_val;
    if (fabsf(alpha) < 1e-6f) {
        return (T)val;
    }
    float sin_val = sinf(alpha * val);
    return (T)(val + (sin_val * sin_val) / alpha);
}

template <typename T>
__global__ void snake_kernel(const T* x, T* dst, float alpha, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = snake_device<T>(x[idx], alpha);
    }
}

template <typename T>
__global__ void snake_strided_kernel(
    const T* x, T* dst, float alpha,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
    size_t nb_x0, size_t nb_x1, size_t nb_x2, size_t nb_x3,
    size_t nb_dst0, size_t nb_dst1, size_t nb_dst2, size_t nb_dst3
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = ne0 * ne1 * ne2 * ne3;
    if (idx < total) {
        int64_t i0 = idx % ne0;
        int64_t tmp = idx / ne0;
        int64_t i1 = tmp % ne1;
        tmp = tmp / ne1;
        int64_t i2 = tmp % ne2;
        int64_t i3 = tmp / ne2;

        const T* px = (const T*)((const char*)x + i3*nb_x3 + i2*nb_x2 + i1*nb_x1 + i0*nb_x0);
        T* pdst = (T*)((char*)dst + i3*nb_dst3 + i2*nb_dst2 + i1*nb_dst1 + i0*nb_dst0);

        *pdst = snake_device<T>(*px, alpha);
    }
}

bool ggml_cuda_op_snake(
    ggml_backend_t backend,
    struct ggml_tensor* x,
    struct ggml_tensor* dst
) {
    int device = ggml_ops_ext_bridge_cuda_get_device(backend);
    cudaStream_t stream = (cudaStream_t)ggml_ops_ext_bridge_cuda_get_stream(backend);

    CUDA_CHECK(cudaSetDevice(device));

    // Extract alpha parameter from op_params
    float* p = (float*)dst->op_params;
    float alpha = p[0];

    int block_size = 256;
    int64_t nelements = ggml_nelements(dst);

    if (x->type == GGML_TYPE_F32) {
        const float* x_d = (const float*)x->data;
        float* dst_d = (float*)dst->data;

        if (ggml_is_contiguous(x) && ggml_is_contiguous(dst)) {
            int grid_size = (nelements + block_size - 1) / block_size;
            snake_kernel<float><<<grid_size, block_size, 0, stream>>>(x_d, dst_d, alpha, nelements);
        } else {
            int64_t total = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
            int grid_size = (total + block_size - 1) / block_size;
            snake_strided_kernel<float><<<grid_size, block_size, 0, stream>>>(
                x_d, dst_d, alpha,
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]
            );
        }
    } else if (x->type == GGML_TYPE_F16) {
        const half* x_d = (const half*)x->data;
        half* dst_d = (half*)dst->data;

        if (ggml_is_contiguous(x) && ggml_is_contiguous(dst)) {
            int grid_size = (nelements + block_size - 1) / block_size;
            snake_kernel<half><<<grid_size, block_size, 0, stream>>>(x_d, dst_d, alpha, nelements);
        } else {
            int64_t total = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
            int grid_size = (total + block_size - 1) / block_size;
            snake_strided_kernel<half><<<grid_size, block_size, 0, stream>>>(
                x_d, dst_d, alpha,
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]
            );
        }
    } else {
        fprintf(stderr, "Unsupported data type for CUDA Snake: %d\n", x->type);
        return false;
    }

    return cudaGetLastError() == cudaSuccess;
}

bool ggml_cuda_op_snake_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    return ggml_cuda_op_snake(backend, node->src[0], node);
}

// ────────────────────────────────────────────────────────────
// SnakeBeta Custom CUDA Implementation
// ────────────────────────────────────────────────────────────

__device__ inline float load_val(const void* ptr, int64_t idx, int type) {
    if (type == 0) { // GGML_TYPE_F32
        return ((const float*)ptr)[idx];
    } else { // GGML_TYPE_F16
        return __half2float(((const half*)ptr)[idx]);
    }
}

template <typename T>
__device__ inline T snake_beta_device(T x_val, float alpha, float beta) {
    float val = (float)x_val;
    if (fabsf(beta) < 1e-6f) {
        return (T)val;
    }
    float sin_val = sinf(alpha * val);
    return (T)(val + (sin_val * sin_val) / beta);
}

template <typename T>
__global__ void snake_beta_kernel(
    const T* x, T* dst, const void* alpha, int alpha_type, const void* beta, int beta_type,
    int64_t ne0, int64_t ne1, int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t c = (idx / ne0) % ne1;
        float a_val = load_val(alpha, c, alpha_type);
        float b_val = load_val(beta, c, beta_type);
        dst[idx] = snake_beta_device<T>(x[idx], a_val, b_val);
    }
}

template <typename T>
__global__ void snake_beta_strided_kernel(
    const T* x, T* dst, const void* alpha, int alpha_type, const void* beta, int beta_type,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
    size_t nb_x0, size_t nb_x1, size_t nb_x2, size_t nb_x3,
    size_t nb_dst0, size_t nb_dst1, size_t nb_dst2, size_t nb_dst3
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = ne0 * ne1 * ne2 * ne3;
    if (idx < total) {
        int64_t i0 = idx % ne0;
        int64_t tmp = idx / ne0;
        int64_t i1 = tmp % ne1; // channel
        tmp = tmp / ne1;
        int64_t i2 = tmp % ne2;
        int64_t i3 = tmp / ne2;

        const T* px = (const T*)((const char*)x + i3*nb_x3 + i2*nb_x2 + i1*nb_x1 + i0*nb_x0);
        T* pdst = (T*)((char*)dst + i3*nb_dst3 + i2*nb_dst2 + i1*nb_dst1 + i0*nb_dst0);

        float a_val = load_val(alpha, i1, alpha_type);
        float b_val = load_val(beta, i1, beta_type);
        *pdst = snake_beta_device<T>(*px, a_val, b_val);
    }
}

bool ggml_cuda_op_snake_beta(
    ggml_backend_t backend,
    struct ggml_tensor* x,
    struct ggml_tensor* alpha,
    struct ggml_tensor* beta,
    struct ggml_tensor* dst
) {
    int device = ggml_ops_ext_bridge_cuda_get_device(backend);
    cudaStream_t stream = (cudaStream_t)ggml_ops_ext_bridge_cuda_get_stream(backend);

    CUDA_CHECK(cudaSetDevice(device));

    int block_size = 256;
    int64_t nelements = ggml_nelements(dst);

    int alpha_type = (alpha->type == GGML_TYPE_F32) ? 0 : 1;
    int beta_type = (beta->type == GGML_TYPE_F32) ? 0 : 1;

    if (x->type == GGML_TYPE_F32) {
        const float* x_d = (const float*)x->data;
        float* dst_d = (float*)dst->data;

        if (ggml_is_contiguous(x) && ggml_is_contiguous(dst)) {
            int grid_size = (nelements + block_size - 1) / block_size;
            snake_beta_kernel<float><<<grid_size, block_size, 0, stream>>>(
                x_d, dst_d, alpha->data, alpha_type, beta->data, beta_type,
                dst->ne[0], dst->ne[1], nelements
            );
        } else {
            int64_t total = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
            int grid_size = (total + block_size - 1) / block_size;
            snake_beta_strided_kernel<float><<<grid_size, block_size, 0, stream>>>(
                x_d, dst_d, alpha->data, alpha_type, beta->data, beta_type,
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]
            );
        }
    } else if (x->type == GGML_TYPE_F16) {
        const half* x_d = (const half*)x->data;
        half* dst_d = (half*)dst->data;

        if (ggml_is_contiguous(x) && ggml_is_contiguous(dst)) {
            int grid_size = (nelements + block_size - 1) / block_size;
            snake_beta_kernel<half><<<grid_size, block_size, 0, stream>>>(
                x_d, dst_d, alpha->data, alpha_type, beta->data, beta_type,
                dst->ne[0], dst->ne[1], nelements
            );
        } else {
            int64_t total = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
            int grid_size = (total + block_size - 1) / block_size;
            snake_beta_strided_kernel<half><<<grid_size, block_size, 0, stream>>>(
                x_d, dst_d, alpha->data, alpha_type, beta->data, beta_type,
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]
            );
        }
    } else {
        fprintf(stderr, "Unsupported data type for CUDA SnakeBeta: %d\n", x->type);
        return false;
    }

    return cudaGetLastError() == cudaSuccess;
}

bool ggml_cuda_op_snake_beta_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    return ggml_cuda_op_snake_beta(backend, node->src[0], node->src[1], node->src[2], node);
}

} // namespace cuda
} // namespace ggml_ops_ext
