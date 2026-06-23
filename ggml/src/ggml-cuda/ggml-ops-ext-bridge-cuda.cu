#include "ggml-ops-ext-bridge.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda/common.cuh"

// These functions are compiled inside ggml-cuda target and have
// legal access to ggml_backend_cuda_context and ggml's internal
// symbols (ggml_cuda_set_device, ggml_cuda_error, CUDA_CHECK, etc.).

int ggml_ops_ext_bridge_cuda_get_device(ggml_backend_t backend) {
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *)backend->context;
    if (!ctx) return 0;
    return ctx->device;
}

void * ggml_ops_ext_bridge_cuda_get_stream(ggml_backend_t backend) {
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *)backend->context;
    if (!ctx) return nullptr;

    int device = ctx->device;

    // Ensure stream is created (lazy init, same as ggml's own code)
    if (ctx->streams[device][ctx->curr_stream_no] == nullptr) {
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaStreamCreateWithFlags(
            &ctx->streams[device][ctx->curr_stream_no],
            cudaStreamNonBlocking));
    }

    return (void *)ctx->streams[device][ctx->curr_stream_no];
}

void * ggml_ops_ext_bridge_cuda_get_cublas(ggml_backend_t backend) {
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *)backend->context;
    if (!ctx) return nullptr;

    int device = ctx->device;

    // Ensure cublas handle is created (lazy init)
    if (ctx->cublas_handles[device] == nullptr) {
        ggml_cuda_set_device(device);
        CUBLAS_CHECK(cublasCreate(&ctx->cublas_handles[device]));
        CUBLAS_CHECK(cublasSetMathMode(
            ctx->cublas_handles[device],
            CUBLAS_TF32_TENSOR_OP_MATH));
    }

    return (void *)ctx->cublas_handles[device];
}
