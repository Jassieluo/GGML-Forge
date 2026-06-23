#include "ops_cuda_common.cuh"

namespace ggml_ops_ext {
namespace cuda {

// Generic block-reduction helper
template <int block_size>
__global__ void layer_norm_kernel(
    const float* x, const float* gamma, const float* beta, float* dst,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
    float eps,
    size_t nb_x0, size_t nb_x1, size_t nb_x2, size_t nb_x3,
    size_t nb_gamma0, size_t nb_beta0,
    size_t nb_dst0, size_t nb_dst1, size_t nb_dst2, size_t nb_dst3
) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    if (sample >= ne3) return;

    // Relocate pointers for this row
    const float* row_x = (const float*)((const char*)x + sample*nb_x3 + channel*nb_x2 + row*nb_x1);
    float* row_dst     = (float*)((char*)dst + sample*nb_dst3 + channel*nb_dst2 + row*nb_dst1);

    float2 mean_var = make_float2(0.0f, 0.0f);

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ne0; col += block_size) {
        const float* px = (const float*)((const char*)row_x + col*nb_x0);
        float xi = *px;
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
        const float* px = (const float*)((const char*)row_x + col*nb_x0);
        const float* pgamma = (const float*)((const char*)gamma + col*nb_gamma0);
        const float* pbeta = (const float*)((const char*)beta + col*nb_beta0);

        float* pdst = (float*)((char*)row_dst + col*nb_dst0);

        *pdst = ((*px - mean) * inv_std) * (*pgamma) + (*pbeta);
    }
}

bool ggml_cuda_op_layer_norm(
    ggml_backend_t backend,
    struct ggml_tensor* x,
    struct ggml_tensor* gamma,
    struct ggml_tensor* beta,
    struct ggml_tensor* dst,
    float eps
) {
    int device = ggml_ops_ext_bridge_cuda_get_device(backend);
    cudaStream_t stream = (cudaStream_t)ggml_ops_ext_bridge_cuda_get_stream(backend);

    CUDA_CHECK(cudaSetDevice(device));

    const float* x_d = (const float*)x->data;
    const float* gamma_d = (const float*)gamma->data;
    const float* beta_d = (const float*)beta->data;
    float* dst_d = (float*)dst->data;

    int64_t ne0 = dst->ne[0]; // Columns (dimension along which to normalize)
    int64_t ne1 = dst->ne[1]; // Rows
    int64_t ne2 = dst->ne[2]; // Channels
    int64_t ne3 = dst->ne[3]; // Samples

    int64_t num_rows = ne1 * ne2 * ne3;
    const dim3 blocks_num(ne1, ne2, ne3);

    if (ne0 < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        layer_norm_kernel<WARP_SIZE><<<blocks_num, block_dims, 0, stream>>>(
            x_d, gamma_d, beta_d, dst_d,
            ne0, ne1, ne2, ne3,
            eps,
            x->nb[0], x->nb[1], x->nb[2], x->nb[3],
            gamma->nb[0], beta->nb[0],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]
        );
    } else {
        const dim3 block_dims(1024, 1, 1);
        layer_norm_kernel<1024><<<blocks_num, block_dims, 32 * sizeof(float2), stream>>>(
            x_d, gamma_d, beta_d, dst_d,
            ne0, ne1, ne2, ne3,
            eps,
            x->nb[0], x->nb[1], x->nb[2], x->nb[3],
            gamma->nb[0], beta->nb[0],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]
        );
    }

    return true;
}


bool ggml_cuda_op_layer_norm_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    float eps;
    std::memcpy(&eps, node->op_params, sizeof(float));
    return ggml_cuda_op_layer_norm(backend, node->src[0], node->src[1], node->src[2], node, eps);
}

} // namespace cuda
} // namespace ggml_ops_ext
