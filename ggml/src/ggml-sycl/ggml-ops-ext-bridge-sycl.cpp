#include "ggml-ops-ext-bridge.h"
#include "ggml-backend-impl.h"
#include "common.hpp"
#include "convert.hpp"

// This file is compiled inside ggml-sycl target and has
// legal access to ggml_backend_sycl_context.

#ifdef _WIN32
extern "C" __declspec(dllexport) void * ggml_ops_ext_bridge_sycl_get_queue(ggml_backend_t backend) {
#else
extern "C" void * ggml_ops_ext_bridge_sycl_get_queue(ggml_backend_t backend) {
#endif
    ggml_backend_sycl_context * sycl_ctx =
        (ggml_backend_sycl_context *)backend->context;
    if (!sycl_ctx) return nullptr;

    return (void *)sycl_ctx->stream();
}

#ifdef _WIN32
extern "C" __declspec(dllexport) bool ggml_ops_ext_bridge_sycl_dequantize(
#else
extern "C" bool ggml_ops_ext_bridge_sycl_dequantize(
#endif
    ggml_backend_t backend, const struct ggml_tensor * src, void * dst, enum ggml_type dst_type
) {
    if (!backend || !src || !dst) return false;
    auto * queue = (dpct::queue_ptr)ggml_ops_ext_bridge_sycl_get_queue(backend);
    ggml_tensor conversion_dst = {};
    conversion_dst.src[0] = const_cast<ggml_tensor *>(src);
    if (dst_type == GGML_TYPE_F32) {
        to_fp32_sycl_t convert = ggml_get_to_fp32_sycl(src->type, &conversion_dst);
        if (!convert) return false;
        convert(src->data, (float *)dst, ggml_nelements(src), queue);
        return true;
    }
    if (dst_type == GGML_TYPE_F16) {
        to_fp16_sycl_t convert = ggml_get_to_fp16_sycl(src->type, &conversion_dst);
        if (!convert) return false;
        convert(src->data, (sycl::half *)dst, ggml_nelements(src), queue);
        return true;
    }
    return false;
}

#ifdef _WIN32
extern "C" __declspec(dllexport) void * ggml_ops_ext_bridge_sycl_pool_alloc(
#else
extern "C" void * ggml_ops_ext_bridge_sycl_pool_alloc(
#endif
    ggml_backend_t backend, size_t size, size_t * actual_size
) {
    if (!backend || !actual_size) return nullptr;
    auto * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    return sycl_ctx ? sycl_ctx->pool().alloc(size, actual_size) : nullptr;
}

#ifdef _WIN32
extern "C" __declspec(dllexport) void ggml_ops_ext_bridge_sycl_pool_free(
#else
extern "C" void ggml_ops_ext_bridge_sycl_pool_free(
#endif
    ggml_backend_t backend, void * ptr, size_t actual_size
) {
    if (!backend || !ptr) return;
    auto * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    if (sycl_ctx) sycl_ctx->pool().free(ptr, actual_size);
}
