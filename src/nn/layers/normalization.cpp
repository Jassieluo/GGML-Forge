#include "nn/layers/normalization.h"

#include "nn/functional/normalization.h"
#include "ops/ops.h"

namespace nn {

ggml_tensor* LayerNorm::forward(
    ggml_context* ctx, ggml_tensor* input, ggml_backend_t selected_backend
) {
    ggml_backend_t target = selected_backend ? selected_backend : backend;
    return functional::layer_norm(
        ctx, input, gamma.local_tensor(), beta.local_tensor(), eps, target);
}

ggml_tensor* InstanceNorm::forward(
    ggml_context* ctx, ggml_tensor* input, ggml_backend_t selected_backend
) {
    ggml_backend_t target = selected_backend ? selected_backend : backend;
    return ggml_ops_instance_norm(
        ctx, input, gamma.local_tensor(), beta.local_tensor(), eps, target);
}

ggml_tensor* AdaLN::forward(
    ggml_context* ctx, ggml_tensor* input, ggml_tensor* scale,
    ggml_tensor* shift, ggml_backend_t selected_backend
) {
    ggml_backend_t target = selected_backend ? selected_backend : backend;
    return ggml_ops_ada_ln(ctx, input, scale, shift, eps, target);
}

AdaLayerNormZero::Output AdaLayerNormZero::forward(
    ggml_context* ctx, ggml_tensor* input, ggml_tensor* embedding,
    ggml_backend_t selected_backend
) {
    ggml_backend_t target = selected_backend ? selected_backend : backend;
    ggml_tensor* projection = linear.forward(ctx, ggml_silu(ctx, embedding));
    const int64_t dimension = projection->ne[0] / 6;
    const int64_t batch = projection->ne[1];
    const size_t element_size = ggml_element_size(projection);
    const size_t row_stride = projection->nb[1];
    auto view = [&](int index) {
        return ggml_view_2d(
            ctx, projection, dimension, batch, row_stride,
            static_cast<size_t>(index) * dimension * element_size);
    };
    ggml_tensor* shift_msa = view(0);
    ggml_tensor* scale_msa = view(1);
    ggml_tensor* gate_msa = view(2);
    ggml_tensor* shift_mlp = view(3);
    ggml_tensor* scale_mlp = view(4);
    ggml_tensor* gate_mlp = view(5);
    ggml_tensor* modulated = ggml_ops_ada_ln(
        ctx, input, scale_msa, shift_msa, eps, target);
    return {modulated, gate_msa, shift_mlp, scale_mlp, gate_mlp};
}

} // namespace nn
