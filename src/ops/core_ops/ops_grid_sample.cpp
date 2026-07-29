#include "ggml-impl.h"
#include "ops/ops.h"

ggml_tensor* ggml_ops_grid_sample_2d(ggml_context* context, ggml_tensor* input, ggml_tensor* grid,
                                     const ggml_ops_ext::ops_grid_sample_2d_config& config,
                                     ggml_backend_t backend) {
    using namespace ggml_ops_ext;
    if (!context || !input || !grid) return nullptr;
    ops_grid_sample_2d_params params{static_cast<uint8_t>(config.mode), static_cast<uint8_t>(config.padding),
                                     static_cast<uint8_t>(config.align_corners), 0};
    ggml_tensor* sources[] = {input, grid};
    ops_request request = {backend ? ggml_backend_get_device(backend) : nullptr,
                           GGML_OP_OPS_VIRT_GRID_SAMPLE_2D, sources, 2, &params, sizeof(params), nullptr};
    ops_grid_sample_2d_desc desc; if (!ops_validate_grid_sample_2d(request, &desc)) return nullptr;
    if (backend && !ggml_ops_backend_supports_op(backend, request.op_id, sources, 2, &params, sizeof(params))) return nullptr;
    const int64_t shape[4] = {desc.output_width, desc.output_height, desc.channels, desc.batch};
    ggml_tensor* result = ops_new_virtual_node(context, GGML_OP_OPS_VIRT_GRID_SAMPLE_2D,
                                               input->type, 4, shape, 2, sources);
    if (result) ggml_set_op_params(result, &params, sizeof(params)); return result;
}
