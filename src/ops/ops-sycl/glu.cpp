#include "common.hpp"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "ops/ops.h"
#include "ops_sycl.h"

namespace ggml_ops_ext {
namespace sycl {

template <typename T> class GluSYCLKernel;

template <typename T> static void launch_glu(::sycl::queue* q, const T* x, T* dst, int64_t C, int64_t T_len) {
    const int64_t nelements = C * T_len;
    q->submit([&](::sycl::handler& cgh) {
        cgh.parallel_for<GluSYCLKernel<T>>(::sycl::range<1>(nelements), [=](::sycl::id<1> id) {
            int64_t idx = id[0];
            int64_t t = idx / C;
            int64_t c = idx % C;
            float x1 = (float)x[t * 2 * C + c];
            float x2 = (float)x[t * 2 * C + C + c];
            dst[idx] = (T)(x1 * (1.0f / (1.0f + ::sycl::exp(-x2))));
        });
    });
}

bool ggml_sycl_op_glu(ggml_backend_t backend, struct ggml_tensor* x, struct ggml_tensor* dst) {
    if (!x || !dst)
        return false;

    ::sycl::queue* q = (::sycl::queue*)ggml_ops_ext_bridge_sycl_get_queue(backend);
    if (!q)
        return false;

    int64_t C = dst->ne[0];
    int64_t T = ggml_nelements(dst) / C;

    if (x->type == GGML_TYPE_F32) {
        const float* x_d = (const float*)x->data;
        float* dst_d = (float*)dst->data;

        launch_glu(q, x_d, dst_d, C, T);
    } else if (x->type == GGML_TYPE_F16) {
        launch_glu(q, (const ::sycl::half*)x->data, (::sycl::half*)dst->data, C, T);
    } else {
        return false;
    }
    return true;
}

bool ggml_sycl_op_glu_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    return ggml_sycl_op_glu(backend, node ? node->src[0] : nullptr, node);
}

} // namespace sycl
} // namespace ggml_ops_ext
