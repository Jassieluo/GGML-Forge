#include "ops/ops.h"
#include "ops_sycl.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "common.hpp"

namespace ggml_ops_ext {
namespace sycl {

class GluSYCLKernelF32;

bool ggml_sycl_op_glu(
    ggml_backend_t backend,
    struct ggml_tensor* x,
    struct ggml_tensor* dst
) {
    if (!x || !dst) return false;

    ::sycl::queue* q = (::sycl::queue*)ggml_ops_ext_bridge_sycl_get_queue(backend);
    if (!q) return false;

    int64_t C = dst->ne[0];
    int64_t T = dst->ne[1];
    int64_t nelements = C * T;

    if (x->type == GGML_TYPE_F32) {
        const float* x_d = (const float*)x->data;
        float* dst_d = (float*)dst->data;

        q->submit([&](::sycl::handler &cgh) {
            cgh.parallel_for<GluSYCLKernelF32>(
                ::sycl::range<1>(nelements),
                [=](::sycl::id<1> id) {
                    int64_t idx = id[0];
                    int64_t t = idx / C;
                    int64_t c = idx % C;
                    float x1 = x_d[t * 2 * C + c];
                    float x2 = x_d[t * 2 * C + C + c];
                    dst_d[idx] = x1 * (1.0f / (1.0f + ::sycl::exp(-x2)));
                }
            );
        });
    } else {
        return false;
    }

    q->wait();
    return true;
}

bool ggml_sycl_op_glu_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    return ggml_sycl_op_glu(backend, node ? node->src[0] : nullptr, node);
}

} // namespace sycl
} // namespace ggml_ops_ext
