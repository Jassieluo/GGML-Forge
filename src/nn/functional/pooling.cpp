#include "nn/functional/pooling.h"
#include "ops/ops.h"

namespace nn::functional {
namespace {

ggml_tensor* pool_nd(
    ggml_context* context,
    ggml_tensor* input,
    const ggml_ops_ext::ops_pool_nd_config& source_config,
    ggml_ops_ext::ops_pool_mode mode,
    int spatial_dims,
    ggml_backend_t backend
) {
    auto config = source_config;
    config.spatial_dims = spatial_dims;
    config.mode = mode;
    if (spatial_dims == 1) {
        return ggml_ops_pool_1d(context, input, config, backend);
    }
    if (spatial_dims == 2) {
        return ggml_ops_pool_2d(context, input, config, backend);
    }
    return ggml_ops_pool_3d(context, input, config, backend);
}

ggml_tensor* adaptive_pool_nd(
    ggml_context* context,
    ggml_tensor* input,
    const ggml_ops_ext::ops_adaptive_pool_nd_config& source_config,
    ggml_ops_ext::ops_pool_mode mode,
    int spatial_dims,
    ggml_backend_t backend
) {
    auto config = source_config;
    config.spatial_dims = spatial_dims;
    config.mode = mode;
    if (spatial_dims == 1) {
        return ggml_ops_adaptive_pool_1d(context, input, config, backend);
    }
    if (spatial_dims == 2) {
        return ggml_ops_adaptive_pool_2d(context, input, config, backend);
    }
    return ggml_ops_adaptive_pool_3d(context, input, config, backend);
}

} // namespace

ggml_tensor* max_pool1d(
    ggml_context* context,
    ggml_tensor* input,
    const ggml_ops_ext::ops_pool_nd_config& config,
    ggml_backend_t backend
) {
    return pool_nd(context, input, config, ggml_ops_ext::ops_pool_mode::maximum, 1, backend);
}

ggml_tensor* avg_pool1d(
    ggml_context* context,
    ggml_tensor* input,
    const ggml_ops_ext::ops_pool_nd_config& config,
    ggml_backend_t backend
) {
    return pool_nd(context, input, config, ggml_ops_ext::ops_pool_mode::average, 1, backend);
}

ggml_tensor* max_pool2d(
    ggml_context* context,
    ggml_tensor* input,
    const ggml_ops_ext::ops_pool_nd_config& config,
    ggml_backend_t backend
) {
    return pool_nd(context, input, config, ggml_ops_ext::ops_pool_mode::maximum, 2, backend);
}

ggml_tensor* avg_pool2d(
    ggml_context* context,
    ggml_tensor* input,
    const ggml_ops_ext::ops_pool_nd_config& config,
    ggml_backend_t backend
) {
    return pool_nd(context, input, config, ggml_ops_ext::ops_pool_mode::average, 2, backend);
}

ggml_tensor* max_pool3d(
    ggml_context* context,
    ggml_tensor* input,
    const ggml_ops_ext::ops_pool_nd_config& config,
    ggml_backend_t backend
) {
    return pool_nd(context, input, config, ggml_ops_ext::ops_pool_mode::maximum, 3, backend);
}

ggml_tensor* avg_pool3d(
    ggml_context* context,
    ggml_tensor* input,
    const ggml_ops_ext::ops_pool_nd_config& config,
    ggml_backend_t backend
) {
    return pool_nd(context, input, config, ggml_ops_ext::ops_pool_mode::average, 3, backend);
}

ggml_tensor* adaptive_max_pool1d(
    ggml_context* context,
    ggml_tensor* input,
    const ggml_ops_ext::ops_adaptive_pool_nd_config& config,
    ggml_backend_t backend
) {
    return adaptive_pool_nd(
        context, input, config, ggml_ops_ext::ops_pool_mode::maximum, 1, backend);
}

ggml_tensor* adaptive_avg_pool1d(
    ggml_context* context,
    ggml_tensor* input,
    const ggml_ops_ext::ops_adaptive_pool_nd_config& config,
    ggml_backend_t backend
) {
    return adaptive_pool_nd(
        context, input, config, ggml_ops_ext::ops_pool_mode::average, 1, backend);
}

ggml_tensor* adaptive_max_pool2d(
    ggml_context* context,
    ggml_tensor* input,
    const ggml_ops_ext::ops_adaptive_pool_nd_config& config,
    ggml_backend_t backend
) {
    return adaptive_pool_nd(
        context, input, config, ggml_ops_ext::ops_pool_mode::maximum, 2, backend);
}

ggml_tensor* adaptive_avg_pool2d(
    ggml_context* context,
    ggml_tensor* input,
    const ggml_ops_ext::ops_adaptive_pool_nd_config& config,
    ggml_backend_t backend
) {
    return adaptive_pool_nd(
        context, input, config, ggml_ops_ext::ops_pool_mode::average, 2, backend);
}

ggml_tensor* adaptive_max_pool3d(
    ggml_context* context,
    ggml_tensor* input,
    const ggml_ops_ext::ops_adaptive_pool_nd_config& config,
    ggml_backend_t backend
) {
    return adaptive_pool_nd(
        context, input, config, ggml_ops_ext::ops_pool_mode::maximum, 3, backend);
}

ggml_tensor* adaptive_avg_pool3d(
    ggml_context* context,
    ggml_tensor* input,
    const ggml_ops_ext::ops_adaptive_pool_nd_config& config,
    ggml_backend_t backend
) {
    return adaptive_pool_nd(
        context, input, config, ggml_ops_ext::ops_pool_mode::average, 3, backend);
}

} // namespace nn::functional
