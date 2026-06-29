#include "ops/ops.h"
#include <cmath>
#include <omp.h>

namespace ggml_ops_ext {
namespace cpu {

bool ops_cpu_op_glu(ggml_backend_t backend, struct ggml_tensor* node) {
    (void)backend;
    if (node->op != GGML_OP_OPS_VIRT_GLU) return false;

    struct ggml_tensor* x = node->src[0];
    struct ggml_tensor* dst = node;

    const float* x_d = (const float*)x->data;
    float* dst_d = (float*)dst->data;

    int64_t C = dst->ne[0];
    int64_t T = dst->ne[1];

    #pragma omp parallel for collapse(2)
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t c = 0; c < C; ++c) {
            float x1 = x_d[t * 2 * C + c];
            float x2 = x_d[t * 2 * C + C + c];
            float sig = 1.0f / (1.0f + std::exp(-x2));
            dst_d[t * C + c] = x1 * sig;
        }
    }

    return true;
}

} // namespace cpu
} // namespace ggml_ops_ext
