#include "ops_cuda_common.cuh"

namespace ggml_ops_ext {
namespace cuda {

#ifndef GGML_USE_CUDNN
__global__ void im2col_1d_kernel(
    const float* x, float* data_col,
    int64_t C, int64_t W, int64_t OW, int64_t kW,
    int stride, int padding, int dilation,
    int64_t N,
    size_t nb_x0, size_t nb_x1, size_t nb_x2
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * C * kW * OW;
    if (idx < total) {
        int64_t ow = idx % OW;
        int64_t tmp = idx / OW;
        int64_t ik = tmp % kW;
        tmp = tmp / kW;
        int64_t ic = tmp % C;
        int64_t n = tmp / C;

        int64_t iw = ow * stride - padding + ik * dilation;
        float val = 0.0f;
        if (iw >= 0 && iw < W) {
            const float* px = (const float*)((const char*)x + n * nb_x2 + ic * nb_x1 + iw * nb_x0);
            val = *px;
        }
        data_col[n * (C * kW * OW) + (ic * kW + ik) * OW + ow] = val;
    }
}
#endif

bool ggml_cuda_op_conv_1d(
    ggml_backend_t backend,
    struct ggml_tensor* w,
    struct ggml_tensor* x,
    struct ggml_tensor* node,
    int stride,
    int padding,
    int dilation
) {
    struct ggml_tensor* dst = node;
    // 1. Get the GGML CUDA context
    ggml_backend_cuda_context* ctx = (ggml_backend_cuda_context*)backend->context;
    int device = ctx->device;
    
    // Get CUDA stream. Create if not initialized.
    cudaStream_t stream = ctx->streams[device][ctx->curr_stream_no];
    if (stream == nullptr) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&ctx->streams[device][ctx->curr_stream_no], cudaStreamNonBlocking));
        stream = ctx->streams[device][ctx->curr_stream_no];
    }

    // Set CUDA device
    CUDA_CHECK(cudaSetDevice(device));

    // Cast weights to FP32 if needed
    const float* w_d = (const float*)w->data;
    ops_cuda_alloc<float> w_f32_alloc(stream);
    if (w->type != GGML_TYPE_F32) {
        const int64_t ne_w = ggml_nelements(w);
        w_f32_alloc.alloc(ne_w);
        int block_size = 256;
        int grid_size = (ne_w + block_size - 1) / block_size;
        if (w->type == GGML_TYPE_F16) {
            convert_f16_to_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                (const half*)w->data, w_f32_alloc.get(), ne_w);
            w_d = w_f32_alloc.get();
        } else if (w->type == GGML_TYPE_BF16) {
            convert_bf16_to_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                (const unsigned short*)w->data, w_f32_alloc.get(), ne_w);
            w_d = w_f32_alloc.get();
        } else {
            fprintf(stderr, "Unsupported weight type for CUDA convolution: %d\n", w->type);
            exit(1);
        }
    }

    const float* x_d = (const float*)x->data;
    float* dst_d = (float*)dst->data;

    int64_t N = x->ne[2];
    int64_t C = x->ne[1];
    int64_t W = x->ne[0];
    int64_t K = w->ne[2];
    int64_t kW = w->ne[0];
    int64_t OW = dst->ne[0];

#ifdef GGML_USE_CUDNN
    // Get or create cuDNN handle and set stream
    cudnnHandle_t cudnn = get_cudnn_handle(device);
    CUDNN_CHECK(cudnnSetStream(cudnn, stream));

    // Create cuDNN descriptors
    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           N, C, 1, W));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           K, C, 1, kW));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           N, K, 1, OW));

    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
                                                 0 /*pad_h*/, padding /*pad_w*/,
                                                 1 /*stride_h*/, stride /*stride_w*/,
                                                 1 /*dilation_h*/, dilation /*dilation_w*/,
                                                 CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // Algorithm selection
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    size_t workspace_size = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, x_desc, w_desc, conv_desc, y_desc, algo, &workspace_size));

    // Workspace allocation using async allocator
    ops_cuda_alloc<uint8_t> workspace_alloc(stream);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        workspace_alloc.alloc(workspace_size);
        workspace = workspace_alloc.get();
    }

    // Execute convolution
    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(
        cudnn,
        &alpha, x_desc, x_d,
        w_desc, w_d,
        conv_desc, algo,
        workspace, workspace_size,
        &beta, y_desc, dst_d));

    // Cleanup
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(w_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
#else
    // Get cuBLAS handle from backend context and set stream
    cublasHandle_t cublas = ctx->cublas_handle();
    CUBLAS_CHECK(cublasSetStream(cublas, stream));

    bool is_1x1 = (kW == 1 && stride == 1 && padding == 0 && dilation == 1);
    float alpha = 1.0f;
    float beta = 0.0f;

    if (is_1x1) {
        // 1x1 Convolution Shortcut: Bypasses im2col completely, direct GEMM write
        for (int64_t n = 0; n < N; ++n) {
            CUBLAS_CHECK(cublasSgemm(
                cublas,
                CUBLAS_OP_N, CUBLAS_OP_N,
                OW, K, C,
                &alpha,
                x_d + n * (C * OW), OW,
                w_d, C,
                &beta,
                dst_d + n * (K * OW), OW
            ));
        }
    } else {
        // Standard convolution with im2col
        float* data_col = nullptr;
        ops_cuda_alloc<float> col_alloc_temp(stream);
        
        if (node->src[2]) {
            data_col = (float*)node->src[2]->data;
        } else {
            size_t col_size = N * C * kW * OW;
            col_alloc_temp.alloc(col_size);
            data_col = col_alloc_temp.get();
        }

        // Launch im2col kernel
        int col_size = N * C * kW * OW;
        int block_size = 256;
        int grid_size = (col_size + block_size - 1) / block_size;
        im2col_1d_kernel<<<grid_size, block_size, 0, stream>>>(
            x_d, data_col,
            C, W, OW, kW,
            stride, padding, dilation,
            N,
            x->nb[0], x->nb[1], x->nb[2]
        );

        // Call cublasSgemm for each batch
        for (int64_t n = 0; n < N; ++n) {
            CUBLAS_CHECK(cublasSgemm(
                cublas,
                CUBLAS_OP_N, CUBLAS_OP_N,
                OW, K, C * kW,
                &alpha,
                data_col + n * (C * kW * OW), OW,
                w_d, C * kW,
                &beta,
                dst_d + n * (K * OW), OW
            ));
        }
    }
#endif

    return true;
}

bool ggml_cuda_op_conv_1d_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    ops_conv_1d_params params;
    if (!ops_extract_conv_1d_params(node, params)) {
        return false;
    }
    return ggml_cuda_op_conv_1d(backend, params.w, params.x, node, params.stride, params.padding, params.dilation);
}

} // namespace cuda
} // namespace ggml_ops_ext
