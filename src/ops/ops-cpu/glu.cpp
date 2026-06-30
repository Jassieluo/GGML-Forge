#include "ops/ops.h"
#include "ops_cpu_common.h"
#include <cmath>
#include <omp.h>

namespace ggml_ops_ext {
namespace cpu {

template <typename T_in, typename T_out>
static void compute_glu(const T_in* x_d, T_out* dst_d, int64_t C, int64_t T) {
    #pragma omp parallel for collapse(2)
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t c = 0; c < C; ++c) {
            float x1 = read_val(&x_d[t * 2 * C + c]);
            float x2 = read_val(&x_d[t * 2 * C + C + c]);
            float sig = 1.0f / (1.0f + std::exp(-x2));
            write_val(&dst_d[t * C + c], x1 * sig);
        }
    }
}

bool ops_cpu_op_glu(ggml_backend_t backend, struct ggml_tensor* node) {
    (void)backend;
    if ((int)node->op != GGML_OP_OPS_VIRT_GLU) return false;

    struct ggml_tensor* x = node->src[0];
    struct ggml_tensor* dst = node;

    int64_t C = dst->ne[0];
    int64_t T = dst->ne[1];

    if (x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        compute_glu((const float*)x->data, (float*)dst->data, C, T);
    } else if (x->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
        compute_glu((const ggml_fp16_t*)x->data, (float*)dst->data, C, T);
    } else if (x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        compute_glu((const float*)x->data, (ggml_fp16_t*)dst->data, C, T);
    } else if (x->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        compute_glu((const ggml_fp16_t*)x->data, (ggml_fp16_t*)dst->data, C, T);
    } else {
        return false;
    }

    return true;
}

} // namespace cpu
} // namespace ggml_ops_ext
