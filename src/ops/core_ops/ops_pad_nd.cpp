#include "ggml-impl.h"
#include "ops/ops.h"

namespace {

ggml_tensor* make_pad_nd_node(ggml_context* context, ggml_tensor* input,
                              const ggml_ops_ext::ops_pad_nd_config& config, ggml_backend_t backend,
                              ggml_ops_ext::ops_virt_op_type op, int spatial_dims) {
    using namespace ggml_ops_ext;
    if (!context || !input || config.spatial_dims != spatial_dims) {
        return nullptr;
    }
    ops_pad_nd_encoded_params encoded;
    if (!ops_encode_pad_nd_params(config, encoded)) {
        return nullptr;
    }
    ggml_tensor* sources[] = {input};
    ops_request request = {backend ? ggml_backend_get_device(backend) : nullptr,
                           op,
                           sources,
                           1,
                           &encoded,
                           sizeof(encoded),
                           nullptr};
    ops_pad_nd_desc desc;
    if (!ops_validate_pad_nd_contract(request, spatial_dims, &desc)) {
        return nullptr;
    }
    if (backend &&
        !ggml_ops_backend_supports_op(backend, op, sources, 1, &encoded, sizeof(encoded))) {
        return nullptr;
    }
    int64_t shape[4] = {1, 1, 1, 1};
    if (spatial_dims == 1) {
        shape[0] = desc.output_size[0];
        shape[1] = desc.channels;
        shape[2] = desc.batch;
    } else if (spatial_dims == 2) {
        shape[0] = desc.output_size[0];
        shape[1] = desc.output_size[1];
        shape[2] = desc.channels;
        shape[3] = desc.batch;
    } else {
        shape[0] = desc.output_size[0] * desc.output_size[1] * desc.output_size[2];
        shape[1] = desc.channels;
        shape[2] = desc.batch;
    }
    ggml_tensor* result = ops_new_virtual_node(context, op, input->type, 4, shape, 1, sources);
    if (result) {
        ggml_set_op_params(result, &encoded, sizeof(encoded));
    }
    return result;
}

} // namespace

ggml_tensor* ggml_ops_pad_1d(ggml_context* context, ggml_tensor* input,
                             const ggml_ops_ext::ops_pad_nd_config& config,
                             ggml_backend_t backend) {
    return make_pad_nd_node(context, input, config, backend, ggml_ops_ext::GGML_OP_OPS_VIRT_PAD_1D,
                            1);
}

ggml_tensor* ggml_ops_pad_2d(ggml_context* context, ggml_tensor* input,
                             const ggml_ops_ext::ops_pad_nd_config& config,
                             ggml_backend_t backend) {
    return make_pad_nd_node(context, input, config, backend, ggml_ops_ext::GGML_OP_OPS_VIRT_PAD_2D,
                            2);
}

ggml_tensor* ggml_ops_pad_3d(ggml_context* context, ggml_tensor* input,
                             const ggml_ops_ext::ops_pad_nd_config& config,
                             ggml_backend_t backend) {
    return make_pad_nd_node(context, input, config, backend, ggml_ops_ext::GGML_OP_OPS_VIRT_PAD_3D,
                            3);
}
