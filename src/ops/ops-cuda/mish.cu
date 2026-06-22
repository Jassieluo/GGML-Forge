#include "ops_cuda_common.cuh"

namespace ggml_ops_ext {
namespace cuda {

__global__ void mish_kernel(const float* x, float* dst, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        float clamped_val = fmaxf(-20.0f, fminf(val, 20.0f));
        float ex = expf(clamped_val);
        float ex1 = ex + 1.0f;
        float ex1_sq = ex1 * ex1;
        dst[idx] = val * (ex1_sq - 1.0f) / (ex1_sq + 1.0f);
    }
}

__global__ void mish_strided_kernel(
    const float* x, float* dst,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
    size_t nb_x0, size_t nb_x1, size_t nb_x2, size_t nb_x3,
    size_t nb_dst0, size_t nb_dst1, size_t nb_dst2, size_t nb_dst3
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = ne0 * ne1 * ne2 * ne3;
    if (idx < total) {
        int64_t i0 = idx % ne0;
        int64_t tmp = idx / ne0;
        int64_t i1 = tmp % ne1;
        tmp = tmp / ne1;
        int64_t i2 = tmp % ne2;
        int64_t i3 = tmp / ne2;

        const float* px = (const float*)((const char*)x + i3*nb_x3 + i2*nb_x2 + i1*nb_x1 + i0*nb_x0);
        float* pdst = (float*)((char*)dst + i3*nb_dst3 + i2*nb_dst2 + i1*nb_dst1 + i0*nb_dst0);

        float val = *px;
        float clamped_val = fmaxf(-20.0f, fminf(val, 20.0f));
        float ex = expf(clamped_val);
        float ex1 = ex + 1.0f;
        float ex1_sq = ex1 * ex1;
        *pdst = val * (ex1_sq - 1.0f) / (ex1_sq + 1.0f);
    }
}

bool ggml_cuda_op_mish(
    ggml_backend_t backend,
    struct ggml_tensor* x,
    struct ggml_tensor* dst
) {
    ggml_backend_cuda_context* ctx = (ggml_backend_cuda_context*)backend->context;
    int device = ctx->device;
    
    cudaStream_t stream = ctx->streams[device][ctx->curr_stream_no];
    if (stream == nullptr) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&ctx->streams[device][ctx->curr_stream_no], cudaStreamNonBlocking));
        stream = ctx->streams[device][ctx->curr_stream_no];
    }

    CUDA_CHECK(cudaSetDevice(device));

    const float* x_d = (const float*)x->data;
    float* dst_d = (float*)dst->data;

    int64_t nelements = ggml_nelements(dst);

    if (ggml_is_contiguous(x) && ggml_is_contiguous(dst)) {
        int block_size = 256;
        int grid_size = (nelements + block_size - 1) / block_size;
        mish_kernel<<<grid_size, block_size, 0, stream>>>(x_d, dst_d, nelements);
    } else {
        int64_t total = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
        int block_size = 256;
        int grid_size = (total + block_size - 1) / block_size;
        mish_strided_kernel<<<grid_size, block_size, 0, stream>>>(
            x_d, dst_d,
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            x->nb[0], x->nb[1], x->nb[2], x->nb[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]
        );
    }

    return true;
}

bool ggml_cuda_op_mish_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    return ggml_cuda_op_mish(backend, node->src[0], node);
}

} // namespace cuda
} // namespace ggml_ops_ext
