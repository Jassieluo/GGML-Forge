#include "nn/functional/padding.h"
#include "ops/ops.h"

namespace nn::functional {

ggml_tensor* pad1d(ggml_context* context, ggml_tensor* input,
                   const ggml_ops_ext::ops_pad_nd_config& source_config, ggml_backend_t backend) {
    auto config = source_config;
    config.spatial_dims = 1;
    return ggml_ops_pad_1d(context, input, config, backend);
}

ggml_tensor* pad2d(ggml_context* context, ggml_tensor* input,
                   const ggml_ops_ext::ops_pad_nd_config& source_config, ggml_backend_t backend) {
    auto config = source_config;
    config.spatial_dims = 2;
    return ggml_ops_pad_2d(context, input, config, backend);
}

ggml_tensor* pad3d(ggml_context* context, ggml_tensor* input,
                   const ggml_ops_ext::ops_pad_nd_config& source_config, ggml_backend_t backend) {
    auto config = source_config;
    config.spatial_dims = 3;
    return ggml_ops_pad_3d(context, input, config, backend);
}

} // namespace nn::functional
