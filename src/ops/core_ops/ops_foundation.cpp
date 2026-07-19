#include "ops/ops.h"

namespace {

ggml_tensor* repeat_vector(
    ggml_context* ctx, ggml_tensor* vector, ggml_tensor* target,
    int64_t n0, int64_t n1, int64_t n2, int64_t n3
) {
    if (!vector || ggml_nelements(vector) != n0 * n1 * n2 * n3) return nullptr;
    return ggml_repeat(ctx, ggml_reshape_4d(ctx, vector, n0, n1, n2, n3), target);
}

ggml_tensor* move_axis_first(ggml_context* ctx, ggml_tensor* input, int axis, int order[4], int inverse[4]) {
    if (!input || axis < 0 || axis >= 4) return nullptr;
    order[0] = axis;
    int next = 1;
    for (int i = 0; i < 4; ++i) if (i != axis) order[next++] = i;
    for (int i = 0; i < 4; ++i) inverse[order[i]] = i;
    return axis == 0 ? input : ggml_cont(ctx, ggml_permute(ctx, input, order[0], order[1], order[2], order[3]));
}

ggml_tensor* restore_axis(ggml_context* ctx, ggml_tensor* input, int axis, const int inverse[4]) {
    return axis == 0 ? input : ggml_cont(ctx, ggml_permute(ctx, input, inverse[0], inverse[1], inverse[2], inverse[3]));
}

} // namespace

ggml_tensor* ggml_ops_rms_norm(
    ggml_context* ctx, ggml_tensor* input, ggml_tensor* weight, float eps
) {
    if (!ctx || !input || eps < 0.0f) return nullptr;
    ggml_tensor* output = ggml_rms_norm(ctx, input, eps);
    if (!weight) return output;
    ggml_tensor* affine = repeat_vector(ctx, weight, output, input->ne[0], 1, 1, 1);
    return affine ? ggml_mul(ctx, output, affine) : nullptr;
}

ggml_tensor* ggml_ops_group_norm(
    ggml_context* ctx, ggml_tensor* input, int groups,
    ggml_tensor* weight, ggml_tensor* bias, float eps
) {
    if (!ctx || !input || groups <= 0 || input->ne[2] <= 0 || input->ne[2] % groups || eps < 0.0f) return nullptr;
    ggml_tensor* output = ggml_group_norm(ctx, input, groups, eps);
    if (weight) {
        ggml_tensor* affine = repeat_vector(ctx, weight, output, 1, 1, input->ne[2], 1);
        if (!affine) return nullptr;
        output = ggml_mul(ctx, output, affine);
    }
    if (bias) {
        ggml_tensor* offset = repeat_vector(ctx, bias, output, 1, 1, input->ne[2], 1);
        if (!offset) return nullptr;
        output = ggml_add(ctx, output, offset);
    }
    return output;
}

ggml_tensor* ggml_ops_l2_normalize(
    ggml_context* ctx, ggml_tensor* input, int axis, float eps
) {
    if (!ctx || !input || eps < 0.0f) return nullptr;
    int order[4], inverse[4];
    ggml_tensor* permuted = move_axis_first(ctx, input, axis, order, inverse);
    if (!permuted) return nullptr;
    return restore_axis(ctx, ggml_l2_norm(ctx, permuted, eps), axis, inverse);
}

ggml_tensor* ggml_ops_gated_activation(
    ggml_context* ctx, ggml_tensor* input, ggml_ops_gate_activation activation, int axis
) {
    if (!ctx || !input || axis < 0 || axis >= 4 || input->ne[axis] <= 0 || input->ne[axis] % 2) return nullptr;
    int order[4], inverse[4];
    ggml_tensor* value = move_axis_first(ctx, input, axis, order, inverse);
    if (!value) return nullptr;
    const int64_t half = value->ne[0] / 2;
    ggml_tensor* gate = ggml_view_4d(ctx, value, half, value->ne[1], value->ne[2], value->ne[3],
                                    value->nb[1], value->nb[2], value->nb[3], 0);
    ggml_tensor* linear = ggml_view_4d(ctx, value, half, value->ne[1], value->ne[2], value->ne[3],
                                      value->nb[1], value->nb[2], value->nb[3], half * value->nb[0]);
    gate = ggml_cont(ctx, gate);
    linear = ggml_cont(ctx, linear);
    switch (activation) {
        case ggml_ops_gate_activation::silu: gate = ggml_silu(ctx, gate); break;
        case ggml_ops_gate_activation::gelu: gate = ggml_gelu(ctx, gate); break;
        case ggml_ops_gate_activation::relu: gate = ggml_relu(ctx, gate); break;
        case ggml_ops_gate_activation::identity: break;
    }
    return restore_axis(ctx, ggml_mul(ctx, gate, linear), axis, inverse);
}
