#include "conv1d-cudnn.cuh"
#include "convert.cuh"

#include <vector>

// cuDNN 1D convolution using 2D API with H=1.
// Leverages Tensor Cores automatically (FP16 accumulate → FP32 on Volta+).
//
// Tensor layouts (all NCHW, H=1):
//   Input  a (weight): ggml [kW, in_ch, out_ch]  ≡ cuDNN filter [out_ch, in_ch, 1, kW]
//   Input  b (data):   ggml [W, in_ch, N]         ≡ cuDNN tensor [N, in_ch, 1, W]
//   Output dst:        ggml [OW, out_ch, N]       ≡ cuDNN tensor [N, out_ch, 1, OW]

void ggml_cuda_op_conv_1d_cudnn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // weight [kW, in_ch, out_ch]
    const ggml_tensor * src1 = dst->src[1]; // data   [W, in_ch, N]

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    cudaStream_t stream = ctx.stream();
    cudnnHandle_t handle = ctx.cudnn_handle();

    const int32_t * opts = (const int32_t *)dst->op_params;
    const int s0 = opts[0]; // stride
    const int p0 = opts[1]; // padding
    const int d0 = opts[2]; // dilation

    const int kW = (int)src0->ne[0];  // kernel_size
    const int C  = (int)src0->ne[1];  // in_channels
    const int K  = (int)src0->ne[2];  // out_channels
    const int W  = (int)src1->ne[0];  // seq_len
    const int N  = (int)src1->ne[2]; // batch (ne[2] = 1 for 2D inputs)

    // Output dims
    const int OW = (int)dst->ne[0]; // out_len (precomputed in ggml_conv_1d_cudnn)

    // ── Tensor descriptors ──────────────────────────────────────
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
                                                0 /*pad_h*/, p0 /*pad_w*/,
                                                1 /*stride_h*/, s0 /*stride_w*/,
                                                1 /*dilation_h*/, d0 /*dilation_w*/,
                                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // ── Algorithm selection: use IMPLICIT_PRECOMP_GEMM (fast, Tensor Core on Volta+) ──
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    size_t workspace_size = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        handle, x_desc, w_desc, conv_desc, y_desc, algo, &workspace_size));

    // ── Workspace allocation ──
    ggml_cuda_pool_alloc<uint8_t> workspace_alloc(ctx.pool());
    void * workspace = nullptr;
    if (workspace_size > 0) {
        workspace_alloc.alloc(workspace_size);
        workspace = workspace_alloc.get();
    }

    // ── Cast weight to FP32 if needed ──
    const float * src0_d = (const float *)src0->data;
    ggml_cuda_pool_alloc<float> src0_f32_alloc(ctx.pool());
    if (src0->type != GGML_TYPE_F32) {
        const int64_t ne_src0 = ggml_nelements(src0);
        src0_f32_alloc.alloc(ne_src0);
        const auto to_fp32 = ggml_get_to_fp32_cuda(src0->type);
        GGML_ASSERT(to_fp32 != nullptr);
        to_fp32(src0->data, src0_f32_alloc.get(), ne_src0, stream);
        src0_d = src0_f32_alloc.get();
    }

    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;

    // ── Execute convolution ──
    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(
        handle,
        &alpha, x_desc, src1_d,
        w_desc, src0_d,
        conv_desc, algo,
        workspace, workspace_size,
        &beta, y_desc, dst_d));

    // ── Cleanup ──
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(w_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
}
