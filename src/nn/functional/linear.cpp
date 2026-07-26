#include "nn/functional/linear.h"

#include <stdexcept>
#include <string>

namespace nn::functional {

static bool can_multiply(const ggml_tensor* weight, const ggml_tensor* input) {
    return weight->ne[0] == input->ne[0] &&
           weight->ne[2] == input->ne[2] && weight->ne[3] == input->ne[3];
}

static bool use_cpu_policy(ggml_backend_t backend) {
    if (!backend) return true;
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    return !device || ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU;
}

ggml_tensor* linear(
    ggml_context* ctx, ggml_tensor* input, ggml_tensor* weight,
    ggml_tensor* bias, ggml_backend_t backend
) {
    if (weight->ne[2] > 1 && weight->ne[0] == 1) {
        weight = ggml_cont(ctx, ggml_reshape_2d(ctx, weight, weight->ne[1], weight->ne[2]));
    }
    ggml_tensor* matmul_weight = weight;
    ggml_tensor* matmul_input = input;
    if (use_cpu_policy(backend)) {
        if (matmul_weight->type == GGML_TYPE_F16) {
            matmul_weight = ggml_cont(ctx, ggml_cast(ctx, matmul_weight, GGML_TYPE_F32));
        }
        if (matmul_input->type == GGML_TYPE_F16) {
            matmul_input = ggml_cont(ctx, ggml_cast(ctx, matmul_input, GGML_TYPE_F32));
        }
    }
    if (!can_multiply(matmul_weight, matmul_input)) {
        throw std::invalid_argument(
            "nn::functional::linear: incompatible weight [" +
            std::to_string(matmul_weight->ne[0]) + ", " + std::to_string(matmul_weight->ne[1]) +
            "] and input [" +
            std::to_string(matmul_input->ne[0]) + ", " + std::to_string(matmul_input->ne[1]) +
            "] shapes");
    }
    ggml_tensor* output = ggml_mul_mat(ctx, matmul_weight, matmul_input);
    ggml_mul_mat_set_prec(output, GGML_PREC_F32);
    if (bias) output = ggml_add(ctx, output, ggml_reshape_2d(ctx, bias, bias->ne[0], 1));
    return output;
}

} // namespace nn::functional
