#include "ops/ops.h"
#include "ops_sycl.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "common.hpp" // From ggml-sycl
#include <iostream>

namespace ggml_ops_ext {
namespace sycl {

class SnakeSYCLKernelF32;
class SnakeStridedSYCLKernelF32;
class SnakeSYCLKernelF16;
class SnakeStridedSYCLKernelF16;

template <typename T>
inline T snake_device(T x_val, float alpha) {
    float val = (float)x_val;
    if (::sycl::fabs(alpha) < 1e-6f) {
        return (T)val;
    }
    float sin_val = ::sycl::sin(alpha * val);
    return (T)(val + (sin_val * sin_val) / alpha);
}

template <typename T, typename KernelCont, typename KernelStrided>
void launch_snake_sycl(::sycl::queue* q, const T* x_d, T* dst_d, float alpha, struct ggml_tensor* x, struct ggml_tensor* dst) {
    int64_t nelements = ggml_nelements(dst);

    if (ggml_is_contiguous(x) && ggml_is_contiguous(dst)) {
        q->submit([&](::sycl::handler &cgh) {
            cgh.parallel_for<KernelCont>(
                ::sycl::range<1>(nelements),
                [=](::sycl::id<1> id) {
                    int64_t idx = id[0];
                    dst_d[idx] = snake_device<T>(x_d[idx], alpha);
                }
            );
        });
    } else {
        int64_t ne0 = dst->ne[0];
        int64_t ne1 = dst->ne[1];
        int64_t ne2 = dst->ne[2];
        int64_t ne3 = dst->ne[3];

        size_t nb_x0 = x->nb[0];
        size_t nb_x1 = x->nb[1];
        size_t nb_x2 = x->nb[2];
        size_t nb_x3 = x->nb[3];

        size_t nb_dst0 = dst->nb[0];
        size_t nb_dst1 = dst->nb[1];
        size_t nb_dst2 = dst->nb[2];
        size_t nb_dst3 = dst->nb[3];

        int64_t total = ne0 * ne1 * ne2 * ne3;

        q->submit([&](::sycl::handler &cgh) {
            cgh.parallel_for<KernelStrided>(
                ::sycl::range<1>(total),
                [=](::sycl::id<1> id) {
                    int64_t idx = id[0];
                    int64_t i0 = idx % ne0;
                    int64_t tmp = idx / ne0;
                    int64_t i1 = tmp % ne1;
                    tmp = tmp / ne1;
                    int64_t i2 = tmp % ne2;
                    int64_t i3 = tmp / ne2;

                    const T* px = (const T*)((const char*)x_d + i3*nb_x3 + i2*nb_x2 + i1*nb_x1 + i0*nb_x0);
                    T* pdst = (T*)((char*)dst_d + i3*nb_dst3 + i2*nb_dst2 + i1*nb_dst1 + i0*nb_dst0);

                    *pdst = snake_device<T>(*px, alpha);
                }
            );
        });
    }
}

bool ggml_sycl_op_snake(
    ggml_backend_t backend,
    struct ggml_tensor* x,
    struct ggml_tensor* dst
) {
    ::sycl::queue* q = (::sycl::queue*)ggml_ops_ext_bridge_sycl_get_queue(backend);
    if (!q) return false;

    // Extract alpha parameter from op_params
    float* p = (float*)dst->op_params;
    float alpha = p[0];

    GGML_ASSERT(x->type == dst->type);

    if (x->type == GGML_TYPE_F32) {
        launch_snake_sycl<float, SnakeSYCLKernelF32, SnakeStridedSYCLKernelF32>(
            q, (const float*)x->data, (float*)dst->data, alpha, x, dst);
    } else if (x->type == GGML_TYPE_F16) {
        launch_snake_sycl<::sycl::half, SnakeSYCLKernelF16, SnakeStridedSYCLKernelF16>(
            q, (const ::sycl::half*)x->data, (::sycl::half*)dst->data, alpha, x, dst);
    } else {
        std::cerr << "[ops-sycl] Snake error: unsupported data type: " << x->type << std::endl;
        return false;
    }

    q->wait();
    return true;
}

bool ggml_sycl_op_snake_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    return ggml_sycl_op_snake(backend, node->src[0], node);
}

} // namespace sycl
} // namespace ggml_ops_ext
