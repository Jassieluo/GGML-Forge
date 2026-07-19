#pragma once

#include "ggml-backend.h"
#include "ggml.h"
#include "ops/contracts/pool_nd.h"
#include "ops/contracts/adaptive_pool_nd.h"

namespace nn::functional {

ggml_tensor* max_pool1d(ggml_context* ctx, ggml_tensor* input,
                        const ggml_ops_ext::ops_pool_nd_config& config,
                        ggml_backend_t backend = nullptr);
ggml_tensor* avg_pool1d(ggml_context* ctx, ggml_tensor* input,
                        const ggml_ops_ext::ops_pool_nd_config& config,
                        ggml_backend_t backend = nullptr);
ggml_tensor* max_pool2d(ggml_context* ctx, ggml_tensor* input,
                        const ggml_ops_ext::ops_pool_nd_config& config,
                        ggml_backend_t backend = nullptr);
ggml_tensor* avg_pool2d(ggml_context* ctx, ggml_tensor* input,
                        const ggml_ops_ext::ops_pool_nd_config& config,
                        ggml_backend_t backend = nullptr);
ggml_tensor* max_pool3d(ggml_context* ctx, ggml_tensor* input,
                        const ggml_ops_ext::ops_pool_nd_config& config,
                        ggml_backend_t backend = nullptr);
ggml_tensor* avg_pool3d(ggml_context* ctx, ggml_tensor* input,
                        const ggml_ops_ext::ops_pool_nd_config& config,
                        ggml_backend_t backend = nullptr);

ggml_tensor* adaptive_max_pool1d(ggml_context* ctx, ggml_tensor* input,
                                 const ggml_ops_ext::ops_adaptive_pool_nd_config& config,
                                 ggml_backend_t backend = nullptr);
ggml_tensor* adaptive_avg_pool1d(ggml_context* ctx, ggml_tensor* input,
                                 const ggml_ops_ext::ops_adaptive_pool_nd_config& config,
                                 ggml_backend_t backend = nullptr);
ggml_tensor* adaptive_max_pool2d(ggml_context* ctx, ggml_tensor* input,
                                 const ggml_ops_ext::ops_adaptive_pool_nd_config& config,
                                 ggml_backend_t backend = nullptr);
ggml_tensor* adaptive_avg_pool2d(ggml_context* ctx, ggml_tensor* input,
                                 const ggml_ops_ext::ops_adaptive_pool_nd_config& config,
                                 ggml_backend_t backend = nullptr);
ggml_tensor* adaptive_max_pool3d(ggml_context* ctx, ggml_tensor* input,
                                 const ggml_ops_ext::ops_adaptive_pool_nd_config& config,
                                 ggml_backend_t backend = nullptr);
ggml_tensor* adaptive_avg_pool3d(ggml_context* ctx, ggml_tensor* input,
                                 const ggml_ops_ext::ops_adaptive_pool_nd_config& config,
                                 ggml_backend_t backend = nullptr);

} // namespace nn::functional
