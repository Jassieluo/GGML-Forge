#include "ops_cuda_common.cuh"

#include <cstring>

namespace ggml_ops_ext {
namespace cuda {

// y = act(layer_norm(x [+ residual]) * gamma + beta)
// Activation ids match ggml_ops_gate_activation:
// 0 = silu, 1 = gelu (tanh approximation), 2 = relu, 3 = identity.
__device__ inline float fused_norm_act_activation(float v, int32_t activation) {
    switch (activation) {
        case 0: // silu
            return v / (1.0f + expf(-v));
        case 1: // gelu (tanh approximation, matching ggml)
            return 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
        case 2: // relu
            return fmaxf(v, 0.0f);
        default: // identity
            return v;
    }
}

template <int block_size, typename T>
__global__ void fused_norm_act_kernel(const T* x, const T* gamma, const T* beta, const T* residual,
                                      T* dst, int64_t ne0, float eps, int32_t activation) {
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;

    // All tensors are contiguous (enforced by the contract): dense rows of ne0.
    const T* row_x = x + row * ne0;
    const T* row_residual = residual ? residual + row * ne0 : nullptr;
    T* row_dst = dst + row * ne0;

    float2 mean_var = make_float2(0.0f, 0.0f);

    ggml_cuda_pdl_sync();
    for (int64_t col = tid; col < ne0; col += block_size) {
        float v = (float)row_x[col];
        if (row_residual) {
            v += (float)row_residual[col];
        }
        mean_var.x += v;
        mean_var.y += v * v;
    }

    // Sum up partial sums using GGML's optimized warp-shuffle block_reduce
    extern __shared__ float2 s_sum2[];
    mean_var = block_reduce<block_reduce_method::SUM, block_size>(mean_var, s_sum2);

    const float mean = mean_var.x / ne0;
    const float var = mean_var.y / ne0 - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int64_t col = tid; col < ne0; col += block_size) {
        float v = (float)row_x[col];
        if (row_residual) {
            v += (float)row_residual[col];
        }
        const float normed = (v - mean) * inv_std * (float)gamma[col] + (float)beta[col];
        row_dst[col] = (T)fused_norm_act_activation(normed, activation);
    }
}

bool ggml_cuda_op_fused_norm_act(ggml_backend_t backend, struct ggml_tensor* x,
                                 struct ggml_tensor* gamma, struct ggml_tensor* beta,
                                 struct ggml_tensor* residual, struct ggml_tensor* dst, float eps,
                                 int32_t activation) {
    int device = ggml_ops_ext_bridge_cuda_get_device(backend);
    cudaStream_t stream = (cudaStream_t)ggml_ops_ext_bridge_cuda_get_stream(backend);

    CUDA_CHECK(cudaSetDevice(device));

    const int64_t ne0 = x->ne[0];
    const int64_t rows = ggml_nelements(x) / ne0;

    // grid.x allows 2^31-1 blocks; one block per row.
    if (rows > INT32_MAX) {
        fprintf(stderr, "CUDA FusedNormAct: row count out of range (%lld)\n", (long long)rows);
        return false;
    }

    const dim3 blocks_num((unsigned int)rows, 1, 1);

    if (x->type == GGML_TYPE_F32) {
        const float* x_d = (const float*)x->data;
        const float* gamma_d = (const float*)gamma->data;
        const float* beta_d = (const float*)beta->data;
        const float* residual_d = residual ? (const float*)residual->data : nullptr;
        float* dst_d = (float*)dst->data;

        if (ne0 <= 1024) {
            const dim3 block_dims(WARP_SIZE, 1, 1);
            fused_norm_act_kernel<WARP_SIZE, float><<<blocks_num, block_dims, 0, stream>>>(
                x_d, gamma_d, beta_d, residual_d, dst_d, ne0, eps, activation);
        } else {
            const dim3 block_dims(1024, 1, 1);
            fused_norm_act_kernel<1024, float><<<blocks_num, block_dims, 32 * sizeof(float2), stream>>>(
                x_d, gamma_d, beta_d, residual_d, dst_d, ne0, eps, activation);
        }
    } else if (x->type == GGML_TYPE_F16) {
        const half* x_d = (const half*)x->data;
        const half* gamma_d = (const half*)gamma->data;
        const half* beta_d = (const half*)beta->data;
        const half* residual_d = residual ? (const half*)residual->data : nullptr;
        half* dst_d = (half*)dst->data;

        if (ne0 <= 1024) {
            const dim3 block_dims(WARP_SIZE, 1, 1);
            fused_norm_act_kernel<WARP_SIZE, half><<<blocks_num, block_dims, 0, stream>>>(
                x_d, gamma_d, beta_d, residual_d, dst_d, ne0, eps, activation);
        } else {
            const dim3 block_dims(1024, 1, 1);
            fused_norm_act_kernel<1024, half><<<blocks_num, block_dims, 32 * sizeof(float2), stream>>>(
                x_d, gamma_d, beta_d, residual_d, dst_d, ne0, eps, activation);
        }
    } else {
        fprintf(stderr, "Unsupported data type for CUDA FusedNormAct: %d\n", x->type);
        return false;
    }

    return cudaGetLastError() == cudaSuccess;
}

bool ggml_cuda_op_fused_norm_act_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    ops_fused_norm_act_params params;
    std::memcpy(&params, node->op_params, sizeof(params));
    return ggml_cuda_op_fused_norm_act(backend, node->src[0], node->src[1], node->src[2],
                                       node->src[3], node, params.eps, params.activation);
}

} // namespace cuda
} // namespace ggml_ops_ext
