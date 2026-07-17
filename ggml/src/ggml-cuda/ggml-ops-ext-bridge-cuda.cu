#include "ggml-ops-ext-bridge.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/convert.cuh"

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

bool ggml_ops_ext_bridge_cuda_dequantize(
    ggml_backend_t backend, const struct ggml_tensor * src, void * dst, enum ggml_type dst_type
) {
    if (!backend || !src || !dst) return false;
    cudaStream_t stream = (cudaStream_t)ggml_ops_ext_bridge_cuda_get_stream(backend);
    if (dst_type == GGML_TYPE_F32) {
        to_fp32_cuda_t convert = ggml_get_to_fp32_cuda(src->type);
        if (!convert) return false;
        convert(src->data, (float *)dst, ggml_nelements(src), stream);
        return true;
    }
    if (dst_type == GGML_TYPE_F16) {
        to_fp16_cuda_t convert = ggml_get_to_fp16_cuda(src->type);
        if (!convert) return false;
        convert(src->data, (half *)dst, ggml_nelements(src), stream);
        return true;
    }
    return false;
}
