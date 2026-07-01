#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <cstring>

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
    
    // Future custom/fused operators to be designed
    GGML_OP_OPS_VIRT_FUSED_ATTN,
    GGML_OP_OPS_VIRT_FUSED_NORM_ACT,
    GGML_OP_OPS_VIRT_POS_ENCODING,
    
    GGML_OP_OPS_VIRT_GLU,
    GGML_OP_OPS_VIRT_RELATIVE_PE_KEYS,
    GGML_OP_OPS_VIRT_RELATIVE_PE_VALUES,
    GGML_OP_OPS_VIRT_INSTANCE_NORM,
    GGML_OP_OPS_VIRT_SNAKE,
    GGML_OP_OPS_VIRT_ADA_LN,

    GGML_OP_OPS_VIRT_COUNT
};


typedef bool (*ops_op_handler_t)(ggml_backend_t backend, struct ggml_tensor* node);

struct ops_handler_entry {
    int op_id; // Standard ggml_op or ggml_ops_ext::ops_virt_op_type
    ops_op_handler_t handler;
};

typedef struct ggml_tensor* (*ops_op_builder_t)(
    struct ggml_context* ctx,
    int op_id,
    struct ggml_tensor** srcs,
    int n_srcs,
    const int32_t* params,
    int n_params,
    ggml_backend_t backend
);

struct ops_builder_entry {
    int op_id; // Standard ggml_op or ggml_ops_ext::ops_virt_op_type
    ops_op_builder_t builder;
};

struct ops_backend_interface {
    // Backend name prefix, e.g. "CUDA" or "SYCL"
    const char* backend_name_prefix;

    // Unified handler registry
    const ops_handler_entry* handlers;
    int n_handlers;

    // Unified builder registry
    const ops_builder_entry* builders;
    int n_builders;
};

// Main lifecycle registry
void install_ops_hook(ggml_backend_t backend);
void uninstall_ops_hook(ggml_backend_t backend);

// Backend registration (called by individual backend modules)
void register_ops_backend(const ops_backend_interface& iface);
const ops_backend_interface* find_ops_backend(ggml_backend_t backend);
ops_op_builder_t find_ops_builder(ggml_backend_t backend, int op_id);

// Custom node factory helper
struct ggml_tensor* ops_new_virtual_node(
    struct ggml_context* ctx,
    ops_virt_op_type op,
    ggml_type type,
    int n_dims,
    const int64_t* ne,
    int n_srcs,
    struct ggml_tensor** srcs
);

inline struct ggml_tensor* force_w_f32(struct ggml_context* ctx, struct ggml_tensor* w) {
    if (!w) return nullptr;
    if (w->type == GGML_TYPE_F32) return w;
    struct ggml_tensor* casted = ggml_cast(ctx, w, GGML_TYPE_F32);
    return ggml_cont(ctx, casted);
}

// Helper to check if a tensor is a custom operator
inline bool ops_is_virt_op(struct ggml_tensor* tensor) {
    if (!tensor) return false;
    int op_val = (int)tensor->op;
    return op_val >= GGML_OP_OPS_VIRT_BASE && op_val < GGML_OP_OPS_VIRT_COUNT;
}

struct ops_conv_1d_params {
    struct ggml_tensor* w;
    struct ggml_tensor* x;
    int stride;
    int padding;
    int dilation;
    int groups;
};

inline bool ops_extract_conv_1d_params(struct ggml_tensor* node, ops_conv_1d_params& params) {
    if ((int)node->op == GGML_OP_OPS_VIRT_CONV_1D) {
        params.w = node->src[0];
        params.x = node->src[1];
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
        if (!src1) return false;

        struct ggml_tensor* im2col_node = nullptr;
        if (src1->op == GGML_OP_IM2COL) {
            im2col_node = src1;
        } else if (src1->op == GGML_OP_RESHAPE && src1->src[0] && src1->src[0]->op == GGML_OP_IM2COL) {
            im2col_node = src1->src[0];
        }

        if (im2col_node) {
            struct ggml_tensor* w = src0;
            if (w->op == GGML_OP_RESHAPE) {
                w = w->src[0];
            }
            params.w = w;
            params.x = im2col_node->src[1];
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
    int stride;
    int padding;
    int dilation;
    int groups;
};

inline bool ops_extract_conv_transpose_1d_params(struct ggml_tensor* node, ops_conv_transpose_1d_params& params) {
    if ((int)node->op == GGML_OP_OPS_VIRT_CONV_TRANSPOSE_1D) {
        params.w = node->src[0];
        params.x = node->src[1];
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
        
        int32_t* p = (int32_t*)node->op_params;
        std::memcpy(&params.scale, &p[0], sizeof(float));
        params.window_size = p[1];
        return true;
    }
    return false;
}

struct ops_relative_pe_keys_params {
    struct ggml_tensor* q;
    struct ggml_tensor* emb_rel_k;
    float scale;
    int32_t window_size;
};

inline bool ops_extract_relative_pe_keys_params(struct ggml_tensor* node, ops_relative_pe_keys_params& params) {
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

inline bool ops_extract_relative_pe_values_params(struct ggml_tensor* node, ops_relative_pe_values_params& params) {
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

inline bool ops_extract_instance_norm_params(struct ggml_tensor* node, ops_instance_norm_params& params) {
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

} // namespace ggml_ops_ext

// Global namespace custom operator wrapper functions
struct ggml_tensor* ggml_ops_attention(
    struct ggml_context* ctx,
    struct ggml_tensor* q,
    struct ggml_tensor* k,
    struct ggml_tensor* v,
    struct ggml_tensor* bias,
    struct ggml_tensor* attn_w,
    float scale,
    int32_t window_size,
    ggml_backend_t backend
);

struct ggml_tensor* ggml_ops_conv_1d(
    struct ggml_context* ctx,
    struct ggml_tensor* w,
    struct ggml_tensor* x,
    int stride,
    int padding,
    int dilation,
    int groups,
    ggml_backend_t backend
);

struct ggml_tensor* ggml_ops_conv_transpose_1d(
    struct ggml_context* ctx,
    struct ggml_tensor* w,
    struct ggml_tensor* x,
    int stride,
    int padding,
    int dilation,
    int groups,
    ggml_backend_t backend
);

struct ggml_tensor* ggml_ops_mish(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    ggml_backend_t backend
);

struct ggml_tensor* ggml_ops_gated_tanh_sigmoid(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    int hidden_channels,
    ggml_backend_t backend
);

struct ggml_tensor* ggml_ops_layer_norm(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* gamma,
    struct ggml_tensor* beta,
    float eps,
    ggml_backend_t backend
);

struct ggml_tensor* ggml_ops_double_swish(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    ggml_backend_t backend
);

struct ggml_tensor* ggml_ops_glu(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    ggml_backend_t backend
);

struct ggml_tensor* ggml_ops_relative_pe_keys(
    struct ggml_context* ctx,
    struct ggml_tensor* q,
    struct ggml_tensor* emb_rel_k,
    float scale,
    int32_t window_size,
    ggml_backend_t backend
);

struct ggml_tensor* ggml_ops_relative_pe_values(
    struct ggml_context* ctx,
    struct ggml_tensor* attn_w,
    struct ggml_tensor* emb_rel_v,
    int32_t window_size,
    ggml_backend_t backend
);

struct ggml_tensor* ggml_ops_instance_norm(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* gamma,
    struct ggml_tensor* beta,
    float eps,
    ggml_backend_t backend
);

struct ggml_tensor* ggml_ops_snake(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    float alpha,
    ggml_backend_t backend
);

struct ggml_tensor* ggml_ops_ada_ln(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* scale,
    struct ggml_tensor* shift,
    float eps,
    ggml_backend_t backend
);

bool ggml_ops_backend_supports_op(ggml_backend_t backend, int op_id);
