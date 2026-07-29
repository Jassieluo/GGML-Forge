#include "ggml-impl.h"
#include "ops/ops.h"

namespace {

ggml_tensor* make_selection_node(ggml_context* context, ggml_ops_ext::ops_virt_op_type op,
                                 ggml_type output_type, ggml_tensor** sources, int source_count,
                                 const ggml_ops_ext::ops_selection_params* params,
                                 ggml_backend_t backend) {
    using namespace ggml_ops_ext;
    if (!context || !sources || !sources[0]) return nullptr;
    ops_request request = {backend ? ggml_backend_get_device(backend) : nullptr, op, sources,
                           source_count, params, params ? sizeof(*params) : 0, nullptr};
    const bool valid = op == GGML_OP_OPS_VIRT_COMPARE ? static_cast<bool>(ops_validate_compare(request))
                     : op == GGML_OP_OPS_VIRT_LOGICAL ? static_cast<bool>(ops_validate_logical(request))
                                                      : static_cast<bool>(ops_validate_where(request));
    if (!valid || (backend && !ggml_ops_backend_supports_op(
                       backend, op, sources, source_count, params, params ? sizeof(*params) : 0))) return nullptr;
    ggml_tensor* result = ops_new_virtual_node(context, op, output_type, 4, sources[0]->ne,
                                               source_count, sources);
    if (result && params) ggml_set_op_params(result, params, sizeof(*params));
    return result;
}

} // namespace

ggml_tensor* ggml_ops_compare(ggml_context* context, ggml_tensor* lhs, ggml_tensor* rhs,
                              ggml_ops_ext::ops_compare_mode mode, ggml_backend_t backend) {
    ggml_ops_ext::ops_selection_params params{static_cast<uint8_t>(mode), {}};
    ggml_tensor* sources[] = {lhs, rhs};
    return make_selection_node(context, ggml_ops_ext::GGML_OP_OPS_VIRT_COMPARE,
                               GGML_TYPE_I32, sources, 2, &params, backend);
}

ggml_tensor* ggml_ops_logical(ggml_context* context, ggml_tensor* lhs, ggml_tensor* rhs,
                              ggml_ops_ext::ops_logical_mode mode, ggml_backend_t backend) {
    ggml_ops_ext::ops_selection_params params{static_cast<uint8_t>(mode), {}};
    ggml_tensor* sources[] = {lhs, rhs};
    const int count = mode == ggml_ops_ext::ops_logical_mode::logical_not ? 1 : 2;
    return make_selection_node(context, ggml_ops_ext::GGML_OP_OPS_VIRT_LOGICAL,
                               GGML_TYPE_I32, sources, count, &params, backend);
}

ggml_tensor* ggml_ops_where(ggml_context* context, ggml_tensor* condition, ggml_tensor* when_true,
                            ggml_tensor* when_false, ggml_backend_t backend) {
    ggml_tensor* sources[] = {condition, when_true, when_false};
    return make_selection_node(context, ggml_ops_ext::GGML_OP_OPS_VIRT_WHERE,
                               when_true ? when_true->type : GGML_TYPE_F32,
                               sources, 3, nullptr, backend);
}
