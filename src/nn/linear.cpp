#include "nn/nn.h"
#include <iostream>

namespace nn {


namespace functional {

static bool our_ggml_can_mul_mat(const struct ggml_tensor* a, const struct ggml_tensor* b) {
    return a->ne[0] == b->ne[0] && a->ne[2] == b->ne[2] && a->ne[3] == b->ne[3];
}

static bool use_cpu_linear_policy(ggml_backend_t backend) {
    if (!backend) return true;
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    return !device || ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU;
}

struct ggml_tensor* linear(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* w,
    struct ggml_tensor* b,
    ggml_backend_t backend
) {
    if (w->ne[2] > 1 && w->ne[0] == 1) {
        w = ggml_cont(ctx, ggml_reshape_2d(ctx, w, w->ne[1], w->ne[2]));  // [in, out]
    }
    struct ggml_tensor* matmul_weight = w;
    struct ggml_tensor* matmul_input = x;
    if (use_cpu_linear_policy(backend)) {
        if (matmul_weight->type == GGML_TYPE_F16) {
            matmul_weight = ggml_cont(ctx, ggml_cast(ctx, matmul_weight, GGML_TYPE_F32));
        }
        if (matmul_input->type == GGML_TYPE_F16) {
            matmul_input = ggml_cont(ctx, ggml_cast(ctx, matmul_input, GGML_TYPE_F32));
        }
    }
    if (!our_ggml_can_mul_mat(matmul_weight, matmul_input)) {
        std::cerr << "[ggml_linear ERROR] w name: " << (matmul_weight->name[0] ? matmul_weight->name : "NULL")
                  << " shape: [" << matmul_weight->ne[0] << ", " << matmul_weight->ne[1] << ", " << matmul_weight->ne[2] << ", " << matmul_weight->ne[3] << "]"
                  << " | x name: " << (matmul_input->name[0] ? matmul_input->name : "NULL")
                  << " shape: [" << matmul_input->ne[0] << ", " << matmul_input->ne[1] << ", " << matmul_input->ne[2] << ", " << matmul_input->ne[3] << "]"
                  << std::endl;
    }
    struct ggml_tensor* out = ggml_mul_mat(ctx, matmul_weight, matmul_input);
    ggml_mul_mat_set_prec(out, GGML_PREC_F32);
    if (b) {
        struct ggml_tensor* b2d = ggml_reshape_2d(ctx, b, b->ne[0], 1);
        out = ggml_add(ctx, out, b2d);
    }
    return out;
}

} // namespace functional

struct ggml_tensor* Linear::forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    return F::linear(ctx, x, weight.tensor(), bias.local_tensor(), backend);
}

} // namespace nn
