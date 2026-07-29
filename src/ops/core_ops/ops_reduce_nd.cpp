#include "ggml-impl.h"
#include "ops/ops.h"

ggml_tensor* ggml_ops_reduce_nd(ggml_context* context, ggml_tensor* input,
                                const ggml_ops_ext::ops_reduce_nd_config& config,
                                ggml_backend_t backend) {
    using namespace ggml_ops_ext;
    if (!context || !input) return nullptr;
    ops_reduce_nd_encoded_params encoded{};
    if (!ops_encode_reduce_nd_params(config, encoded)) return nullptr;
    ggml_tensor* sources[] = {input};
    ops_request request = {backend ? ggml_backend_get_device(backend) : nullptr,
                           GGML_OP_OPS_VIRT_REDUCE_ND, sources, 1,
                           &encoded, sizeof(encoded), nullptr};
    ops_reduce_nd_desc desc;
    if (!ops_validate_reduce_nd_contract(request, &desc)) return nullptr;
    if (backend && !ggml_ops_backend_supports_op(backend, GGML_OP_OPS_VIRT_REDUCE_ND,
                                                 sources, 1, &encoded, sizeof(encoded))) {
        return nullptr;
    }
    ggml_tensor* result = ops_new_virtual_node(context, GGML_OP_OPS_VIRT_REDUCE_ND,
                                               input->type, 4, desc.output_shape, 1, sources);
    if (result) ggml_set_op_params(result, &encoded, sizeof(encoded));
    return result;
}

ggml_tensor* ggml_ops_arg_reduce_nd(ggml_context* context, ggml_tensor* input,
                                    const ggml_ops_ext::ops_arg_reduce_nd_config& config,
                                    ggml_backend_t backend) {
    using namespace ggml_ops_ext;
    if (!context || !input) return nullptr;
    ops_arg_reduce_nd_encoded_params encoded{};
    if (!ops_encode_arg_reduce_nd_params(config, encoded)) return nullptr;
    ggml_tensor* sources[] = {input};
    ops_request request = {backend ? ggml_backend_get_device(backend) : nullptr,
                           GGML_OP_OPS_VIRT_ARG_REDUCE_ND, sources, 1,
                           &encoded, sizeof(encoded), nullptr};
    ops_arg_reduce_nd_desc desc;
    if (!ops_validate_arg_reduce_nd_contract(request, &desc)) return nullptr;
    if (backend && !ggml_ops_backend_supports_op(backend, GGML_OP_OPS_VIRT_ARG_REDUCE_ND,
                                                 sources, 1, &encoded, sizeof(encoded))) return nullptr;
    ggml_tensor* result = ops_new_virtual_node(context, GGML_OP_OPS_VIRT_ARG_REDUCE_ND,
                                               GGML_TYPE_I32, 4, desc.output_shape, 1, sources);
    if (result) ggml_set_op_params(result, &encoded, sizeof(encoded));
    return result;
}
