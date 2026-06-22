#include "ops_cuda_common.cuh"

namespace ggml_ops_ext {
namespace cuda {

#ifndef GGML_USE_CUDNN
__global__ void col2im_1d_kernel(
    const float* data_col, float* dst,
    int64_t C, int64_t W, int64_t OW, int64_t kW,
    int stride, int padding, int dilation,
    int64_t N,
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
            int64_t temp = ow + padding - ik * dilation;
            if (temp >= 0 && temp % stride == 0) {
                int64_t iw = temp / stride;
                if (iw >= 0 && iw < W) {
                    sum += data_col[n * (C * kW * W) + iw * (C * kW) + c * kW + ik];
                }
            }
        }
        float* pdst = (float*)((char*)dst + n * nb_dst2 + c * nb_dst1 + ow * nb_dst0);
        *pdst = sum;
    }
}
#endif

bool ggml_cuda_op_conv_transpose_1d(
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

    const int kW = (int)w->ne[0]; // kernel_size
    const int C  = (int)w->ne[1]; // out_channels (C in weight notation)
    const int K  = (int)w->ne[2]; // in_channels (K in weight notation)
    const int W  = (int)x->ne[0]; // input seq_len
    const int N  = (int)x->ne[2]; // batch
    const int OW = (int)dst->ne[0]; // output seq_len

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
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dx_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           N, C, 1, OW));
    // dy = input
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dy_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           N, K, 1, W));
    // Filter descriptor
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           K, C, 1, kW));

    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
                                                 0 /*pad_h*/, padding /*pad_w*/,
                                                 1 /*stride_h*/, stride /*stride_w*/,
                                                 1 /*dilation_h*/, dilation /*dilation_w*/,
                                                 CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    // Get cuBLAS handle from backend context and set stream
    cublasHandle_t cublas = ctx->cublas_handle();
    CUBLAS_CHECK(cublasSetStream(cublas, stream));

    bool is_1x1 = (kW == 1 && stride == 1 && padding == 0 && dilation == 1);
    float alpha = 1.0f;
    float beta = 0.0f;

    if (is_1x1) {
        // 1x1 Transposed Convolution Shortcut: direct GEMM into destination, 0-Workspace
        for (int64_t n = 0; n < N; ++n) {
            CUBLAS_CHECK(cublasSgemm(
                cublas,
                CUBLAS_OP_N, CUBLAS_OP_T,
                C, W, K,
                &alpha,
                w_d, C,
                x_d + n * (K * W), W,
                &beta,
                dst_d + n * (C * W), C
            ));
        }
    } else {
        // Standard transposed convolution using im2col / col2im
        float* data_col = nullptr;
        ops_cuda_alloc<float> col_alloc_temp(stream);
        
        if (node->src[2]) {
            data_col = (float*)node->src[2]->data;
        } else {
            size_t col_size = N * C * kW * W;
            col_alloc_temp.alloc(col_size);
            data_col = col_alloc_temp.get();
        }

        // Call cublasSgemm for each batch
        for (int64_t n = 0; n < N; ++n) {
            CUBLAS_CHECK(cublasSgemm(
                cublas,
                CUBLAS_OP_N, CUBLAS_OP_T,
                C * kW, W, K,
                &alpha,
                w_d, C * kW,
                x_d + n * (K * W), W,
                &beta,
                data_col + n * (C * kW * W), C * kW
            ));
        }

        // Launch col2im kernel to fold data_col into dst
        int64_t total_dst_elements = N * C * OW;
        int block_size = 256;
        int grid_size = (total_dst_elements + block_size - 1) / block_size;
        col2im_1d_kernel<<<grid_size, block_size, 0, stream>>>(
            data_col, dst_d,
            C, W, OW, kW,
            stride, padding, dilation,
            N,
            dst->nb[0], dst->nb[1], dst->nb[2]
        );
    }
#endif

    return true;
}

bool ggml_cuda_op_conv_transpose_1d_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    ops_conv_transpose_1d_params params;
    if (!ops_extract_conv_transpose_1d_params(node, params)) {
        return false;
    }
    return ggml_cuda_op_conv_transpose_1d(backend, params.w, params.x, node, params.stride, params.padding, params.dilation);
}

} // namespace cuda
} // namespace ggml_ops_ext
