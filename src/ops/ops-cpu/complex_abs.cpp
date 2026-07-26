#include "ops/cpu.h"
#include "ops/ops.h"
#include "ops_cpu_common.h"

#include <cmath>
#include <cstring>

namespace ggml_ops_ext::cpu {

bool ops_cpu_op_complex_abs(ggml_backend_t backend, ggml_tensor* node) {
    const ggml_tensor* x = node->src[0];
    ops_complex_abs_params params;
    std::memcpy(&params, node->op_params, sizeof(params));

    const int64_t n = x->ne[0];
    const int64_t rows = x->ne[1];
    const int64_t batch = x->ne[3];
    const float* in = static_cast<const float*>(x->data);
    float* out = static_cast<float*>(node->data);
    const int threads = backend_thread_count(backend);
    const bool squared = params.squared != 0;

#pragma omp parallel for collapse(2) num_threads(threads) schedule(static)
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t r = 0; r < rows; ++r) {
            const float* re = in + ((b * 2 + 0) * rows + r) * n;
            const float* im = in + ((b * 2 + 1) * rows + r) * n;
            float* out_row = out + (b * rows + r) * n;
            for (int64_t i = 0; i < n; ++i) {
                const float power = re[i] * re[i] + im[i] * im[i];
                out_row[i] = squared ? power : std::sqrt(power);
            }
        }
    }
    return true;
}

} // namespace ggml_ops_ext::cpu
