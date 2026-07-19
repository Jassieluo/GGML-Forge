#include "ops/ops.h"
#include "ggml-impl.h"

namespace {

ggml_tensor* make_conv_nd_node(
    ggml_context* ctx,
    ggml_tensor* weight,
    ggml_tensor* input,
    ggml_tensor* bias,
    const ggml_ops_ext::ops_conv_nd_config& config,
    ggml_backend_t backend,
    ggml_ops_ext::ops_virt_op_type op,
    int spatial_dims,
    bool transposed
) {
    using namespace ggml_ops_ext;
    if (!ctx || !weight || !input || config.spatial_dims != spatial_dims) return nullptr;

    ops_conv_nd_encoded_params encoded;
    if (!ops_encode_conv_nd_params(config, transposed, encoded)) return nullptr;
    ggml_tensor* srcs[] = { weight, input, bias };
    ops_request request = {
        backend ? ggml_backend_get_device(backend) : nullptr,
        op,
        srcs,
        bias ? 3 : 2,
        &encoded,
        sizeof(encoded),
        nullptr,
    };
    ops_conv_nd_desc desc;
    if (!ops_validate_conv_nd_contract(request, spatial_dims, transposed, &desc)) return nullptr;
    if (backend && !ggml_ops_backend_supports_op(
            backend, op, srcs, bias ? 3 : 2, &encoded, sizeof(encoded))) return nullptr;

    int64_t ne[4] = { 1, 1, 1, 1 };
    if (spatial_dims == 2) {
        ne[0] = desc.output_size[0];
        ne[1] = desc.output_size[1];
        ne[2] = desc.output_channels;
        ne[3] = desc.batch;
    } else {
        ne[0] = desc.output_size[0] * desc.output_size[1] * desc.output_size[2];
        ne[1] = desc.output_channels;
        ne[2] = desc.batch;
    }
    ggml_tensor* result = ops_new_virtual_node(
        ctx, op, input->type, 4, ne, bias ? 3 : 2, srcs);
    if (!result) return nullptr;
    ggml_set_op_params(result, &encoded, sizeof(encoded));
    return result;
}

} // namespace

ggml_tensor* ggml_ops_conv_2d(
    ggml_context* ctx, ggml_tensor* weight, ggml_tensor* input,
    const ggml_ops_ext::ops_conv_nd_config& config, ggml_backend_t backend, ggml_tensor* bias
) {
    return make_conv_nd_node(ctx, weight, input, bias, config, backend,
        ggml_ops_ext::GGML_OP_OPS_VIRT_CONV_2D, 2, false);
}

ggml_tensor* ggml_ops_conv_transpose_2d(
    ggml_context* ctx, ggml_tensor* weight, ggml_tensor* input,
    const ggml_ops_ext::ops_conv_nd_config& config, ggml_backend_t backend, ggml_tensor* bias
) {
    return make_conv_nd_node(ctx, weight, input, bias, config, backend,
        ggml_ops_ext::GGML_OP_OPS_VIRT_CONV_TRANSPOSE_2D, 2, true);
}

ggml_tensor* ggml_ops_conv_3d(
    ggml_context* ctx, ggml_tensor* weight, ggml_tensor* input,
    const ggml_ops_ext::ops_conv_nd_config& config, ggml_backend_t backend, ggml_tensor* bias
) {
    return make_conv_nd_node(ctx, weight, input, bias, config, backend,
        ggml_ops_ext::GGML_OP_OPS_VIRT_CONV_3D, 3, false);
}

ggml_tensor* ggml_ops_conv_transpose_3d(
    ggml_context* ctx, ggml_tensor* weight, ggml_tensor* input,
    const ggml_ops_ext::ops_conv_nd_config& config, ggml_backend_t backend, ggml_tensor* bias
) {
    return make_conv_nd_node(ctx, weight, input, bias, config, backend,
        ggml_ops_ext::GGML_OP_OPS_VIRT_CONV_TRANSPOSE_3D, 3, true);
}
