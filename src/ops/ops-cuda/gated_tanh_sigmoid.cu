#include "ops_cuda_common.cuh"

namespace ggml_ops_ext {
namespace cuda {

__global__ void gated_tanh_sigmoid_kernel(
    const float* x, float* dst,
    int64_t hidden_channels, int64_t seq_len, int64_t batch,
    size_t nb_x0, size_t nb_x1, size_t nb_x2, size_t nb_x3,
    size_t nb_dst0, size_t nb_dst1, size_t nb_dst2, size_t nb_dst3
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = hidden_channels * seq_len * batch;
    if (idx < total) {
        int64_t i0 = idx % hidden_channels;
        int64_t tmp = idx / hidden_channels;
        int64_t i1 = tmp % seq_len;
        int64_t i2 = tmp / seq_len;

        // Left: channel i0
        const float* px_l = (const float*)((const char*)x + i2*nb_x2 + i1*nb_x1 + i0*nb_x0);
        // Right: channel i0 + hidden_channels
        const float* px_r = (const float*)((const char*)x + i2*nb_x2 + i1*nb_x1 + (i0 + hidden_channels)*nb_x0);

        float* pdst = (float*)((char*)dst + i2*nb_dst2 + i1*nb_dst1 + i0*nb_dst0);

        float val_l = *px_l;
        float val_r = *px_r;

        // Tanh: clamp val_l to [-10.f, 10.f] as it is multiplied by -2.f inside expf
        float clamped_l = fmaxf(-10.0f, fminf(val_l, 10.0f));
        float sigm_l = 1.0f / (1.0f + expf(-2.0f * clamped_l));
        float tanh_val = 2.0f * sigm_l - 1.0f;

        // Sigmoid: clamp val_r to [-20.f, 20.f]
        float clamped_r = fmaxf(-20.0f, fminf(val_r, 20.0f));
        float sigm_r = 1.0f / (1.0f + expf(-clamped_r));

        *pdst = tanh_val * sigm_r;
    }
}

bool ggml_cuda_op_gated_tanh_sigmoid(
    ggml_backend_t backend,
    struct ggml_tensor* x,
    struct ggml_tensor* dst,
    int hidden_channels
) {
    int device = ggml_ops_ext_bridge_cuda_get_device(backend);
    cudaStream_t stream = (cudaStream_t)ggml_ops_ext_bridge_cuda_get_stream(backend);

    CUDA_CHECK(cudaSetDevice(device));

    const float* x_d = (const float*)x->data;
    float* dst_d = (float*)dst->data;

    int64_t seq_len = dst->ne[1];
    int64_t batch = dst->ne[2];

    int64_t total = hidden_channels * seq_len * batch;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    gated_tanh_sigmoid_kernel<<<grid_size, block_size, 0, stream>>>(
        x_d, dst_d,
        hidden_channels, seq_len, batch,
        x->nb[0], x->nb[1], x->nb[2], x->nb[3],
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]
    );

    return true;
}

bool ggml_cuda_op_gated_tanh_sigmoid_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    int32_t* params = (int32_t*)node->op_params;
    int hidden_channels = params[0];
    return ggml_cuda_op_gated_tanh_sigmoid(backend, node->src[0], node, hidden_channels);
}

} // namespace cuda
} // namespace ggml_ops_ext
