#include "ops_cuda_common.cuh"

namespace ggml_ops_ext {
namespace cuda {

__global__ void glu_kernel_f32(const float* x, float* dst, int C, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * T) return;
    int t = idx / C;
    int c = idx % C;
    float x1 = x[t * 2 * C + c];
    float x2 = x[t * 2 * C + C + c];
    dst[idx] = x1 * (1.0f / (1.0f + expf(-x2)));
}

bool ggml_cuda_op_glu(
    ggml_backend_t backend,
    struct ggml_tensor* x,
    struct ggml_tensor* dst
) {
    int device = ggml_ops_ext_bridge_cuda_get_device(backend);
    cudaStream_t stream = (cudaStream_t)ggml_ops_ext_bridge_cuda_get_stream(backend);

    CUDA_CHECK(cudaSetDevice(device));

    int64_t C = dst->ne[0];
    int64_t T = dst->ne[1];
    int64_t nelements = C * T;

    if (x->type == GGML_TYPE_F32) {
        const float* x_d = (const float*)x->data;
        float* dst_d = (float*)dst->data;

        int block_size = 256;
        int grid_size = (nelements + block_size - 1) / block_size;
        glu_kernel_f32<<<grid_size, block_size, 0, stream>>>(x_d, dst_d, C, T);
    } else {
        fprintf(stderr, "Unsupported data type for CUDA GLU: %d\n", x->type);
        return false;
    }

    return true;
}

bool ggml_cuda_op_glu_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    return ggml_cuda_op_glu(backend, node->src[0], node);
}

} // namespace cuda
} // namespace ggml_ops_ext
