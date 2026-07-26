#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ops/types.h"
#include "ops/contracts/conv1d.h"
#include "ops/contracts/conv_nd.h"
#include "ops/contracts/pool_nd.h"
#include "ops/contracts/pad_nd.h"
#include "ops/contracts/adaptive_pool_nd.h"
#include "ops/contracts/resize_nd.h"
#include "ops/contracts/activation.h"
#include "ops/contracts/normalization.h"
#include "ops/contracts/attention.h"
#include "ops/contracts/sequence.h"
#include "ops/contracts/quantization.h"
#include <cmath>
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <memory>

namespace ggml_ops_ext {

// Virtual operator types starting from 2000 to avoid any conflict with GGML core
enum ops_virt_op_type {
    GGML_OP_OPS_VIRT_BASE = 2000,

    GGML_OP_OPS_VIRT_CONV_1D,
    GGML_OP_OPS_VIRT_CONV_TRANSPOSE_1D,
    GGML_OP_OPS_VIRT_MISH,
    GGML_OP_OPS_VIRT_GATED_TANH_SIGMOID,
    GGML_OP_OPS_VIRT_LAYER_NORM,
    GGML_OP_OPS_VIRT_DOUBLE_SWISH,

    // Reserved slot with no implementation anywhere (no builder, contract, or
    // kernel); probes reject it. Do NOT remove — that would renumber every
    // op id after it. Implement or repurpose in place when needed.
    GGML_OP_OPS_VIRT_FUSED_ATTN,
    GGML_OP_OPS_VIRT_FUSED_NORM_ACT,
    GGML_OP_OPS_VIRT_POS_ENCODING,

    GGML_OP_OPS_VIRT_GLU,
    GGML_OP_OPS_VIRT_RELATIVE_PE_KEYS,
    GGML_OP_OPS_VIRT_RELATIVE_PE_VALUES,
    GGML_OP_OPS_VIRT_INSTANCE_NORM,
    GGML_OP_OPS_VIRT_SNAKE,
    GGML_OP_OPS_VIRT_SNAKE_BETA,
    GGML_OP_OPS_VIRT_ADA_LN,
    GGML_OP_OPS_VIRT_ALIAS_FREE_ACTIVATION,
    GGML_OP_OPS_VIRT_KV_CACHE_UPDATE,
    GGML_OP_OPS_VIRT_CONV_2D,
    GGML_OP_OPS_VIRT_CONV_TRANSPOSE_2D,
    GGML_OP_OPS_VIRT_CONV_3D,
    GGML_OP_OPS_VIRT_CONV_TRANSPOSE_3D,
    GGML_OP_OPS_VIRT_POOL_1D,
    GGML_OP_OPS_VIRT_POOL_2D,
    GGML_OP_OPS_VIRT_POOL_3D,
    GGML_OP_OPS_VIRT_PAD_1D,
    GGML_OP_OPS_VIRT_PAD_2D,
    GGML_OP_OPS_VIRT_PAD_3D,
    GGML_OP_OPS_VIRT_ADAPTIVE_POOL_1D,
    GGML_OP_OPS_VIRT_ADAPTIVE_POOL_2D,
    GGML_OP_OPS_VIRT_ADAPTIVE_POOL_3D,
    GGML_OP_OPS_VIRT_RESIZE_1D,
    GGML_OP_OPS_VIRT_RESIZE_2D,
    GGML_OP_OPS_VIRT_RESIZE_3D,
    GGML_OP_OPS_VIRT_GATED_ACTIVATION,
    // Fused temperature-softmax + top-k/top-p sampling over a logits row.
    GGML_OP_OPS_VIRT_SAMPLE_DIST,

    GGML_OP_OPS_VIRT_COUNT
};

typedef ops_status (*ops_kernel_execute_t)(const ops_execution_context& context,
                                           struct ggml_tensor* node);

typedef ops_probe_result (*ops_kernel_probe_t)(const ops_request& request);

struct ops_kernel_entry {
    int op_id; // Standard ggml_op or ggml_ops_ext::ops_virt_op_type
    const char* name;
    int priority;
    ops_kernel_execute_t execute;
    ops_kernel_probe_t probe = nullptr;
};

struct ops_backend_registration {
    const char* backend_name_prefix;
    const ops_kernel_entry* kernels;
    int n_kernels;
};

// Convenience adapter for kernels whose execution state is fully represented by
// ggml_backend_t. New kernels may register an ops_kernel_execute_t directly.
template <bool (*Kernel)(ggml_backend_t, struct ggml_tensor*)>
inline ops_status ops_backend_kernel_adapter(const ops_execution_context& context,
                                             struct ggml_tensor* node) {
    return Kernel(context.backend, node)
               ? ops_status::ok()
               : ops_status::error(ops_status_code::execution_failed, "kernel execution failed");
}

template <bool (*Kernel)(ggml_backend_t, struct ggml_tensor*)>
constexpr ops_kernel_entry make_ops_kernel(int op_id, const char* name, ops_kernel_probe_t probe,
                                           int priority = 0) {
    return {op_id, name, priority, ops_backend_kernel_adapter<Kernel>, probe};
}

enum class ops_support_profile {
    cpu,
    gpu,
};

inline bool ops_validate_request_contract(ops_support_profile profile, const ops_request& request) {
    switch (request.op_id) {
    case GGML_OP_OPS_VIRT_MISH:
    case GGML_OP_OPS_VIRT_DOUBLE_SWISH:
        return ops_validate_unary_activation(request);
    case GGML_OP_OPS_VIRT_SNAKE:
        return ops_validate_snake(request);
    case GGML_OP_OPS_VIRT_GATED_TANH_SIGMOID:
        return ops_validate_gated_tanh_sigmoid(request);
    case GGML_OP_OPS_VIRT_GLU:
        return ops_validate_glu(request);
    case GGML_OP_OPS_VIRT_GATED_ACTIVATION:
        return ops_validate_gated_activation(request);
    case GGML_OP_OPS_VIRT_LAYER_NORM:
    case GGML_OP_OPS_VIRT_ADA_LN:
        return ops_validate_affine_norm(request);
    case GGML_OP_OPS_VIRT_FUSED_ATTN:
        return ops_validate_fused_attention(request, profile == ops_support_profile::gpu);
    case GGML_OP_OPS_VIRT_RELATIVE_PE_KEYS:
        return ops_validate_relative_pe_keys(request);
    case GGML_OP_OPS_VIRT_RELATIVE_PE_VALUES:
        return ops_validate_relative_pe_values(request);
    case GGML_OP_OPS_VIRT_INSTANCE_NORM:
        return ops_validate_instance_norm(request);
    case GGML_OP_OPS_VIRT_SNAKE_BETA:
        return ops_validate_snake_beta(request);
    case GGML_OP_OPS_VIRT_ALIAS_FREE_ACTIVATION:
        return ops_validate_alias_free_activation(request);
    case GGML_OP_OPS_VIRT_KV_CACHE_UPDATE:
        return ops_validate_kv_cache_update(request);
    case GGML_OP_OPS_VIRT_FUSED_NORM_ACT:
        return ops_validate_fused_norm_act(request);
    case GGML_OP_OPS_VIRT_POS_ENCODING:
        return ops_validate_pos_encoding(request);
    case GGML_OP_OPS_VIRT_SAMPLE_DIST:
        return ops_validate_sample_dist(request);
    default:
        return false;
    }
}

inline bool ops_describe_conv_weight(int op_id, const ggml_tensor* w, const ggml_tensor* x,
                                     int64_t groups, ops_conv_weight_desc& desc) {
    return ops_describe_conv_weight(op_id, GGML_OP_OPS_VIRT_CONV_1D,
                                    GGML_OP_OPS_VIRT_CONV_TRANSPOSE_1D, w, x, groups, desc);
}

inline bool ops_validate_conv_request(int op_id, struct ggml_tensor* const* srcs, int n_srcs,
                                      const void* raw_params, size_t params_size) {
    return (bool)ops_validate_conv_contract(op_id, GGML_OP_OPS_VIRT_CONV_1D,
                                            GGML_OP_OPS_VIRT_CONV_TRANSPOSE_1D, srcs, n_srcs,
                                            raw_params, params_size);
}

inline bool ops_validate_conv_request(const ops_request& request) {
    if (request.op_id == GGML_OP_OPS_VIRT_CONV_2D ||
        request.op_id == GGML_OP_OPS_VIRT_CONV_TRANSPOSE_2D ||
        request.op_id == GGML_OP_OPS_VIRT_CONV_3D ||
        request.op_id == GGML_OP_OPS_VIRT_CONV_TRANSPOSE_3D) {
        const int dims = request.op_id == GGML_OP_OPS_VIRT_CONV_2D ||
                                 request.op_id == GGML_OP_OPS_VIRT_CONV_TRANSPOSE_2D
                             ? 2
                             : 3;
        const bool transposed = request.op_id == GGML_OP_OPS_VIRT_CONV_TRANSPOSE_2D ||
                                request.op_id == GGML_OP_OPS_VIRT_CONV_TRANSPOSE_3D;
        return (bool)ops_validate_conv_nd_contract(request, dims, transposed);
    }
    int64_t output_length = 0;
    ops_conv_weight_desc weight_desc;
    if (!ops_validate_conv_contract(request.op_id, GGML_OP_OPS_VIRT_CONV_1D,
                                    GGML_OP_OPS_VIRT_CONV_TRANSPOSE_1D, request.srcs,
                                    request.n_srcs, request.params, request.params_size,
                                    &output_length, &weight_desc)) {
        return false;
    }
    if (!request.output) {
        return true;
    }
    const ggml_tensor* x = request.srcs[1];
    return request.output->type == x->type && request.output->ne[0] == output_length &&
           request.output->ne[1] == weight_desc.output_channels &&
           request.output->ne[2] == x->ne[2] && request.output->ne[3] == x->ne[3];
}

// Process-wide bridge lifetime. Multiple runtimes share one immutable registry.
void acquire_ops_hook();
void release_ops_hook();

class ops_backend_lane_guard {
  public:
    explicit ops_backend_lane_guard(ggml_backend_t backend);
    ~ops_backend_lane_guard();
    ops_backend_lane_guard(const ops_backend_lane_guard&) = delete;
    ops_backend_lane_guard& operator=(const ops_backend_lane_guard&) = delete;

  private:
    void* lane_ = nullptr;
    std::shared_ptr<void> owner_;
};

// Serialize whole-graph execution only when sessions share one backend instance.
// Separate backend instances use independent lanes and remain concurrent.
enum ggml_status ops_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph* graph);

// Scheduler variant: takes every backend lane the scheduler may dispatch to
// (in a stable global order) before computing, so multi-backend graphs get the
// same serialization guarantee as ops_backend_graph_compute.
enum ggml_status ops_backend_sched_graph_compute(ggml_backend_sched_t sched,
                                                 struct ggml_cgraph* graph);

// Kernel registration and dispatch. Registration is cold-path; dispatch selects
// the highest-priority compatible kernel for the concrete backend device.
bool register_ops_backend(const ops_backend_registration& registration);
ops_probe_result probe_ops_kernel(ggml_backend_dev_t device, int op_id,
                                  struct ggml_tensor* const* srcs, int n_srcs, const void* params,
                                  size_t params_size);
ops_probe_result probe_ops_kernel(const ops_request& request);
ops_status execute_ops_kernel(ggml_backend_t backend, struct ggml_tensor* node);

// Custom node factory helper
struct ggml_tensor* ops_new_virtual_node(struct ggml_context* ctx, ops_virt_op_type op,
                                         ggml_type type, int n_dims, const int64_t* ne, int n_srcs,
                                         struct ggml_tensor** srcs);

inline struct ggml_tensor* force_w_f32(struct ggml_context* ctx, struct ggml_tensor* w) {
    if (!w) {
        return nullptr;
    }
    if (w->type == GGML_TYPE_F32) {
        return w;
    }
    struct ggml_tensor* casted = ggml_cast(ctx, w, GGML_TYPE_F32);
    return ggml_cont(ctx, casted);
}

// Helper to check if a tensor is a custom operator
inline bool ops_is_virt_op(struct ggml_tensor* tensor) {
    if (!tensor) {
        return false;
    }
    int op_val = (int)tensor->op;
    return op_val >= GGML_OP_OPS_VIRT_BASE && op_val < GGML_OP_OPS_VIRT_COUNT;
}

struct ops_conv_1d_params {
    struct ggml_tensor* w;
    struct ggml_tensor* x;
    struct ggml_tensor* bias;
    int stride;
    int padding;
    int dilation;
    int groups;
};

inline bool ops_extract_conv_1d_params(struct ggml_tensor* node, ops_conv_1d_params& params) {
    if ((int)node->op == GGML_OP_OPS_VIRT_CONV_1D) {
        params.w = node->src[0];
        params.x = node->src[1];
        params.bias = node->src[2]; // Can be nullptr
        int32_t* p = (int32_t*)node->op_params;
        params.stride = p[0];
        params.padding = p[1];
        params.dilation = p[2];
        params.groups = p[3];
        return true;
    }

    if (node->op == GGML_OP_MUL_MAT) {
        struct ggml_tensor* src0 = node->src[0];
        struct ggml_tensor* src1 = node->src[1];
        if (!src1) {
            return false;
        }

        struct ggml_tensor* im2col_node = nullptr;
        if (src1->op == GGML_OP_IM2COL) {
            im2col_node = src1;
        } else if (src1->op == GGML_OP_RESHAPE && src1->src[0] &&
                   src1->src[0]->op == GGML_OP_IM2COL) {
            im2col_node = src1->src[0];
        }

        if (im2col_node) {
            struct ggml_tensor* w = src0;
            if (w->op == GGML_OP_RESHAPE) {
                w = w->src[0];
            }
            params.w = w;
            params.x = im2col_node->src[1];
            params.bias = nullptr;
            int32_t* p = (int32_t*)im2col_node->op_params;
            params.stride = p[0];
            params.padding = p[2];
            params.dilation = p[4];
            params.groups = 1;
            return true;
        }
    }
    return false;
}

struct ops_conv_transpose_1d_params {
    struct ggml_tensor* w;
    struct ggml_tensor* x;
    struct ggml_tensor* bias;
    int stride;
    int padding;
    int dilation;
    int groups;
};

inline bool ops_extract_conv_transpose_1d_params(struct ggml_tensor* node,
                                                 ops_conv_transpose_1d_params& params) {
    if ((int)node->op == GGML_OP_OPS_VIRT_CONV_TRANSPOSE_1D) {
        params.w = node->src[0];
        params.x = node->src[1];
        params.bias = node->src[2]; // Can be nullptr
        int32_t* p = (int32_t*)node->op_params;
        params.stride = p[0];
        params.padding = p[1];
        params.dilation = p[2];
        params.groups = p[3];
        return true;
    }
    return false;
}

struct ops_attention_params {
    struct ggml_tensor* q;
    struct ggml_tensor* k;
    struct ggml_tensor* v;
    struct ggml_tensor* bias;
    struct ggml_tensor* attn_w;
    struct ggml_tensor* valid_length;
    struct ggml_tensor* dependency;
    float scale;
    int32_t window_size;
};

inline bool ops_extract_attention_params(struct ggml_tensor* node, ops_attention_params& params) {
    if ((int)node->op == GGML_OP_OPS_VIRT_FUSED_ATTN) {
        params.q = node->src[0];
        params.k = node->src[1];
        params.v = node->src[2];
        params.bias = node->src[3];
        params.attn_w = node->src[4];
        params.valid_length = node->src[5];
        params.dependency = node->src[6];

        int32_t* p = (int32_t*)node->op_params;
        std::memcpy(&params.scale, &p[0], sizeof(float));
        params.window_size = p[1];
        return true;
    }
    return false;
}

struct ops_kv_cache_update_params {
    struct ggml_tensor* cache_k;
    struct ggml_tensor* cache_v;
    struct ggml_tensor* new_k;
    struct ggml_tensor* new_v;
    struct ggml_tensor* position;
};

struct ops_conv_nd_node_params {
    ggml_tensor* weight = nullptr;
    ggml_tensor* input = nullptr;
    ggml_tensor* bias = nullptr;
    ops_conv_nd_encoded_params encoded = {};
};

inline bool ops_extract_conv_nd_params(ggml_tensor* node, ops_conv_nd_node_params& params) {
    if (!node) {
        return false;
    }
    const int op = static_cast<int>(node->op);
    if (op != GGML_OP_OPS_VIRT_CONV_2D && op != GGML_OP_OPS_VIRT_CONV_TRANSPOSE_2D &&
        op != GGML_OP_OPS_VIRT_CONV_3D && op != GGML_OP_OPS_VIRT_CONV_TRANSPOSE_3D) {
        return false;
    }
    params.weight = node->src[0];
    params.input = node->src[1];
    params.bias = node->src[2];
    std::memcpy(&params.encoded, node->op_params, sizeof(params.encoded));
    return params.weight && params.input;
}

inline bool ops_extract_kv_cache_update_params(struct ggml_tensor* node,
                                               ops_kv_cache_update_params& params) {
    if ((int)node->op != GGML_OP_OPS_VIRT_KV_CACHE_UPDATE) {
        return false;
    }
    params.cache_k = node->src[0];
    params.cache_v = node->src[1];
    params.new_k = node->src[2];
    params.new_v = node->src[3];
    params.position = node->src[4];
    return true;
}

struct ops_relative_pe_keys_params {
    struct ggml_tensor* q;
    struct ggml_tensor* emb_rel_k;
    float scale;
    int32_t window_size;
};

inline bool ops_extract_relative_pe_keys_params(struct ggml_tensor* node,
                                                ops_relative_pe_keys_params& params) {
    if ((int)node->op == GGML_OP_OPS_VIRT_RELATIVE_PE_KEYS) {
        params.q = node->src[0];
        params.emb_rel_k = node->src[1];
        int32_t* p = (int32_t*)node->op_params;
        std::memcpy(&params.scale, &p[0], sizeof(float));
        params.window_size = p[1];
        return true;
    }
    return false;
}

struct ops_relative_pe_values_params {
    struct ggml_tensor* attn_w;
    struct ggml_tensor* emb_rel_v;
    int32_t window_size;
};

inline bool ops_extract_relative_pe_values_params(struct ggml_tensor* node,
                                                  ops_relative_pe_values_params& params) {
    if ((int)node->op == GGML_OP_OPS_VIRT_RELATIVE_PE_VALUES) {
        params.attn_w = node->src[0];
        params.emb_rel_v = node->src[1];
        int32_t* p = (int32_t*)node->op_params;
        params.window_size = p[0];
        return true;
    }
    return false;
}

struct ops_instance_norm_params {
    struct ggml_tensor* x;
    struct ggml_tensor* gamma; // Optional (nullptr if none)
    struct ggml_tensor* beta;  // Optional (nullptr if none)
    float eps;
};

inline bool ops_extract_instance_norm_params(struct ggml_tensor* node,
                                             ops_instance_norm_params& params) {
    if ((int)node->op == GGML_OP_OPS_VIRT_INSTANCE_NORM) {
        params.x = node->src[0];
        params.gamma = node->src[1];
        params.beta = node->src[2];
        float* p = (float*)node->op_params;
        params.eps = p[0];
        return true;
    }
    return false;
}

struct ops_snake_params {
    struct ggml_tensor* x;
    float alpha;
};

inline bool ops_extract_snake_params(struct ggml_tensor* node, ops_snake_params& params) {
    if ((int)node->op == GGML_OP_OPS_VIRT_SNAKE) {
        params.x = node->src[0];
        float* p = (float*)node->op_params;
        params.alpha = p[0];
        return true;
    }
    return false;
}

struct ops_snake_beta_params {
    struct ggml_tensor* x;
    struct ggml_tensor* alpha;
    struct ggml_tensor* beta;
};

inline bool ops_extract_snake_beta_params(struct ggml_tensor* node, ops_snake_beta_params& params) {
    if ((int)node->op == GGML_OP_OPS_VIRT_SNAKE_BETA) {
        params.x = node->src[0];
        params.alpha = node->src[1];
        params.beta = node->src[2];
        return true;
    }
    return false;
}

} // namespace ggml_ops_ext

// Global namespace custom operator wrapper functions
// Embedding/Gather uses GGML's native GET_ROWS kernels. Weight is a shared 2D
// table, rows may be floating-point or quantized, and indices may have up to
// three I32 dimensions. Output shape is [embedding_width, ...index_shape].
struct ggml_tensor* ggml_ops_embedding(struct ggml_context* ctx,
                                       struct ggml_tensor* weight,
                                       struct ggml_tensor* indices);

struct ggml_tensor* ggml_ops_attention(struct ggml_context* ctx, struct ggml_tensor* q,
                                       struct ggml_tensor* k, struct ggml_tensor* v,
                                       struct ggml_tensor* bias, struct ggml_tensor* attn_w,
                                       float scale, int32_t window_size, ggml_backend_t backend,
                                       struct ggml_tensor* valid_length = nullptr,
                                       struct ggml_tensor* dependency = nullptr);

struct ggml_tensor* ggml_ops_kv_cache_update(struct ggml_context* ctx, struct ggml_tensor* cache_k,
                                             struct ggml_tensor* cache_v, struct ggml_tensor* new_k,
                                             struct ggml_tensor* new_v,
                                             struct ggml_tensor* position, ggml_backend_t backend);

struct ggml_tensor* ggml_ops_conv_1d(struct ggml_context* ctx, struct ggml_tensor* w,
                                     struct ggml_tensor* x, int stride, int padding, int dilation,
                                     int groups, ggml_backend_t backend,
                                     struct ggml_tensor* bias = nullptr);

struct ggml_tensor* ggml_ops_conv_transpose_1d(struct ggml_context* ctx, struct ggml_tensor* w,
                                               struct ggml_tensor* x, int stride, int padding,
                                               int dilation, int groups, ggml_backend_t backend,
                                               struct ggml_tensor* bias = nullptr);

struct ggml_tensor* ggml_ops_conv_2d(struct ggml_context* ctx, struct ggml_tensor* weight,
                                     struct ggml_tensor* input,
                                     const ggml_ops_ext::ops_conv_nd_config& config,
                                     ggml_backend_t backend, struct ggml_tensor* bias = nullptr);

struct ggml_tensor* ggml_ops_conv_transpose_2d(struct ggml_context* ctx, struct ggml_tensor* weight,
                                               struct ggml_tensor* input,
                                               const ggml_ops_ext::ops_conv_nd_config& config,
                                               ggml_backend_t backend,
                                               struct ggml_tensor* bias = nullptr);

// Conv3D uses packed activations [W*H*D, C, N] because GGML tensors have four
// physical dimensions. config.input_size carries the logical W/H/D shape.
struct ggml_tensor* ggml_ops_conv_3d(struct ggml_context* ctx, struct ggml_tensor* weight,
                                     struct ggml_tensor* input,
                                     const ggml_ops_ext::ops_conv_nd_config& config,
                                     ggml_backend_t backend, struct ggml_tensor* bias = nullptr);

struct ggml_tensor* ggml_ops_conv_transpose_3d(struct ggml_context* ctx, struct ggml_tensor* weight,
                                               struct ggml_tensor* input,
                                               const ggml_ops_ext::ops_conv_nd_config& config,
                                               ggml_backend_t backend,
                                               struct ggml_tensor* bias = nullptr);

struct ggml_tensor* ggml_ops_pool_1d(struct ggml_context* ctx, struct ggml_tensor* input,
                                     const ggml_ops_ext::ops_pool_nd_config& config,
                                     ggml_backend_t backend = nullptr);

struct ggml_tensor* ggml_ops_pool_2d(struct ggml_context* ctx, struct ggml_tensor* input,
                                     const ggml_ops_ext::ops_pool_nd_config& config,
                                     ggml_backend_t backend = nullptr);

struct ggml_tensor* ggml_ops_pool_3d(struct ggml_context* ctx, struct ggml_tensor* input,
                                     const ggml_ops_ext::ops_pool_nd_config& config,
                                     ggml_backend_t backend = nullptr);

struct ggml_tensor* ggml_ops_pad_1d(struct ggml_context* ctx, struct ggml_tensor* input,
                                    const ggml_ops_ext::ops_pad_nd_config& config,
                                    ggml_backend_t backend = nullptr);

struct ggml_tensor* ggml_ops_pad_2d(struct ggml_context* ctx, struct ggml_tensor* input,
                                    const ggml_ops_ext::ops_pad_nd_config& config,
                                    ggml_backend_t backend = nullptr);

struct ggml_tensor* ggml_ops_pad_3d(struct ggml_context* ctx, struct ggml_tensor* input,
                                    const ggml_ops_ext::ops_pad_nd_config& config,
                                    ggml_backend_t backend = nullptr);

struct ggml_tensor*
ggml_ops_adaptive_pool_1d(struct ggml_context* ctx, struct ggml_tensor* input,
                          const ggml_ops_ext::ops_adaptive_pool_nd_config& config,
                          ggml_backend_t backend = nullptr);

struct ggml_tensor*
ggml_ops_adaptive_pool_2d(struct ggml_context* ctx, struct ggml_tensor* input,
                          const ggml_ops_ext::ops_adaptive_pool_nd_config& config,
                          ggml_backend_t backend = nullptr);

struct ggml_tensor*
ggml_ops_adaptive_pool_3d(struct ggml_context* ctx, struct ggml_tensor* input,
                          const ggml_ops_ext::ops_adaptive_pool_nd_config& config,
                          ggml_backend_t backend = nullptr);

struct ggml_tensor* ggml_ops_resize_1d(struct ggml_context* ctx, struct ggml_tensor* input,
                                       const ggml_ops_ext::ops_resize_nd_config& config,
                                       ggml_backend_t backend = nullptr);
struct ggml_tensor* ggml_ops_resize_2d(struct ggml_context* ctx, struct ggml_tensor* input,
                                       const ggml_ops_ext::ops_resize_nd_config& config,
                                       ggml_backend_t backend = nullptr);
struct ggml_tensor* ggml_ops_resize_3d(struct ggml_context* ctx, struct ggml_tensor* input,
                                       const ggml_ops_ext::ops_resize_nd_config& config,
                                       ggml_backend_t backend = nullptr);

struct ggml_tensor* ggml_ops_mish(struct ggml_context* ctx, struct ggml_tensor* x,
                                  ggml_backend_t backend);

enum class ggml_ops_gate_activation : uint8_t {
    silu,
    gelu,
    relu,
    identity,
};

// Foundational operators composed from native GGML nodes. They remain in the
// Ops API so shape/axis semantics are shared by all model categories.
struct ggml_tensor* ggml_ops_rms_norm(struct ggml_context* ctx, struct ggml_tensor* input,
                                      struct ggml_tensor* weight = nullptr, float eps = 1e-5f);

struct ggml_tensor* ggml_ops_group_norm(struct ggml_context* ctx, struct ggml_tensor* input,
                                        int groups, struct ggml_tensor* weight = nullptr,
                                        struct ggml_tensor* bias = nullptr, float eps = 1e-5f);

struct ggml_tensor* ggml_ops_l2_normalize(struct ggml_context* ctx, struct ggml_tensor* input,
                                          int axis = 0, float eps = 1e-12f);

struct ggml_tensor* ggml_ops_gated_activation(struct ggml_context* ctx, struct ggml_tensor* input,
                                              ggml_ops_gate_activation activation, int axis = 0,
                                              ggml_backend_t backend = nullptr);

struct ggml_tensor* ggml_ops_gated_tanh_sigmoid(struct ggml_context* ctx, struct ggml_tensor* x,
                                                int hidden_channels, ggml_backend_t backend);

struct ggml_tensor* ggml_ops_layer_norm(struct ggml_context* ctx, struct ggml_tensor* x,
                                        struct ggml_tensor* gamma, struct ggml_tensor* beta,
                                        float eps, ggml_backend_t backend);

struct ggml_tensor* ggml_ops_double_swish(struct ggml_context* ctx, struct ggml_tensor* x,
                                          ggml_backend_t backend);

struct ggml_tensor* ggml_ops_glu(struct ggml_context* ctx, struct ggml_tensor* x,
                                 ggml_backend_t backend);

struct ggml_tensor* ggml_ops_relative_pe_keys(struct ggml_context* ctx, struct ggml_tensor* q,
                                              struct ggml_tensor* emb_rel_k, float scale,
                                              int32_t window_size, ggml_backend_t backend);

struct ggml_tensor* ggml_ops_relative_pe_values(struct ggml_context* ctx,
                                                struct ggml_tensor* attn_w,
                                                struct ggml_tensor* emb_rel_v,
                                                struct ggml_tensor* attention_output,
                                                int32_t window_size, ggml_backend_t backend);

struct ggml_tensor* ggml_ops_instance_norm(struct ggml_context* ctx, struct ggml_tensor* x,
                                           struct ggml_tensor* gamma, struct ggml_tensor* beta,
                                           float eps, ggml_backend_t backend);

struct ggml_tensor* ggml_ops_snake(struct ggml_context* ctx, struct ggml_tensor* x, float alpha,
                                   ggml_backend_t backend);

struct ggml_tensor* ggml_ops_snake_beta(struct ggml_context* ctx, struct ggml_tensor* x,
                                        struct ggml_tensor* alpha, struct ggml_tensor* beta,
                                        ggml_backend_t backend);

struct ggml_tensor* ggml_ops_alias_free_activation(struct ggml_context* ctx, struct ggml_tensor* x,
                                                   struct ggml_tensor* up_filter,
                                                   struct ggml_tensor* down_filter,
                                                   struct ggml_tensor* alpha,
                                                   struct ggml_tensor* beta,
                                                   ggml_backend_t backend);

struct ggml_tensor* ggml_ops_ada_ln(struct ggml_context* ctx, struct ggml_tensor* x,
                                    struct ggml_tensor* scale, struct ggml_tensor* shift, float eps,
                                    ggml_backend_t backend);

// y = act(layer_norm(x [+ residual]) * gamma + beta). residual may be nullptr.
struct ggml_tensor* ggml_ops_fused_norm_act(struct ggml_context* ctx, struct ggml_tensor* x,
                                            struct ggml_tensor* gamma, struct ggml_tensor* beta,
                                            struct ggml_tensor* residual, float eps,
                                            ggml_ops_gate_activation activation,
                                            ggml_backend_t backend);

// y = x + sinusoidal PE over dims (feature, time); position (I32, 1 element)
// optionally shifts the starting position at execution time (may be nullptr).
// Kernel-required: returns nullptr when the backend has no kernel.
struct ggml_tensor* ggml_ops_pos_encoding(struct ggml_context* ctx, struct ggml_tensor* x,
                                          struct ggml_tensor* position, float base,
                                          int32_t offset, ggml_backend_t backend);

// Fused temperature softmax + top-k/top-p sampling over one logits row.
// uniform is a host-supplied F32 random value in [0, 1); output is I32 [1].
// Kernel-required: returns nullptr when the backend has no kernel.
struct ggml_tensor* ggml_ops_sample_dist(struct ggml_context* ctx, struct ggml_tensor* logits,
                                         struct ggml_tensor* uniform, int32_t top_k, float top_p,
                                         float temperature, ggml_backend_t backend);

bool ggml_ops_backend_supports_op(ggml_backend_t backend, int op_id,
                                  struct ggml_tensor* const* srcs = nullptr, int n_srcs = 0,
                                  const void* params = nullptr, size_t params_size = 0);
