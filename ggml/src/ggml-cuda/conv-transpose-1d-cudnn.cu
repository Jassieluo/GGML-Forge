#include "conv-transpose-1d-cudnn.cuh"
#include "convert.cuh"

#include <vector>

// cuDNN 1D transposed convolution via cudnnConvolutionBackwardData.
// Uses Tensor Cores automatically on Volta+ GPUs.
//
// Transposed conv is the backward pass of a regular conv w.r.t the data:
//   dx = cudnnConvolutionBackwardData(w, dy)
// where:
//   dy_desc = input  tensor [N, out_ch,    1, IW]  (ggml: [IW, out_ch, N])
//   w_desc  = weight tensor [out_ch, in_ch, 1, kW] (ggml: [kW, in_ch, out_ch])
//   dx_desc = output tensor [N, in_ch,     1, OW]  (ggml: [OW, in_ch,  N])
//   conv_desc uses the forward conv params (stride, padding, dilation)

void ggml_cuda_op_conv_transpose_1d_cudnn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // weight [kW, in_ch, out_ch]
    const ggml_tensor * src1 = dst->src[1]; // data   [IW, out_ch, N]

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

    const int kW  = (int)src0->ne[0]; // kernel_size
    const int C   = (int)src0->ne[1]; // in_channels (becomes output channels)
    const int K   = (int)src0->ne[2]; // out_channels (becomes input channels)
    const int IW  = (int)src1->ne[0]; // input seq_len
    const int N   = (int)src1->ne[2]; // batch

    const int OW  = (int)dst->ne[0];  // output seq_len

    // ── Tensor descriptors ──────────────────────────────────────
    // dx_desc = our OUTPUT = [N, in_ch,     1, OW]
    // dy_desc = our INPUT  = [N, out_ch,    1, IW]
    // w_desc  = weight     = [out_ch, in_ch, 1, kW]

    cudnnTensorDescriptor_t dx_desc, dy_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dx_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dy_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    // dx = output
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dx_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           N, C, 1, OW));
    // dy = input
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dy_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           N, K, 1, IW));
    // filter
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           K, C, 1, kW));

    // Forward conv params (what dx→dy would use)
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
                                                0 /*pad_h*/, p0 /*pad_w*/,
                                                1 /*stride_h*/, s0 /*stride_w*/,
                                                1 /*dilation_h*/, d0 /*dilation_w*/,
                                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // ── Algorithm selection: use default algo (fast path, no benchmarking) ──
    cudnnConvolutionBwdDataAlgo_t algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    size_t workspace_size = 0;
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle, w_desc, dy_desc, conv_desc, dx_desc, algo, &workspace_size));

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

    // ── Execute: dx = ConvBackwardData(w, dy) ──
    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionBackwardData(
        handle,
        &alpha, w_desc, src0_d,
        dy_desc, src1_d,
        conv_desc, algo,
        workspace, workspace_size,
        &beta, dx_desc, dst_d));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dx_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dy_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(w_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
}
