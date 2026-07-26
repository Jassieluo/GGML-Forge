#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>

namespace ggml_ops_ext {

enum class ops_status_code : uint8_t {
    success,
    not_handled,
    invalid_request,
    unsupported,
    execution_failed,
};

struct ops_status {
    ops_status_code code = ops_status_code::success;
    const char* message = nullptr;

    constexpr explicit operator bool() const { return code == ops_status_code::success; }
    static constexpr ops_status ok() { return {}; }
    static constexpr ops_status error(ops_status_code code, const char* message) {
        return { code, message };
    }
};

enum class ops_quant_scheme : uint8_t {
    none,
    q4_0,
    q4_1,
    q5_0,
    q5_1,
    q8_0,
    q2_k,
    q3_k,
    q4_k,
    q5_k,
    q6_k,
    iq4_nl,
    iq4_xs,
    mxfp4,
};

enum class ops_weight_layout : uint8_t {
    native,
    channel_rows,
    flattened_rows,
    flexible_rows,
    backend_prepared,
};

struct ops_quantization_desc {
    ops_quant_scheme scheme = ops_quant_scheme::none;
    ggml_type storage_type = GGML_TYPE_F32;
    ggml_type compute_type = GGML_TYPE_F32;
    ggml_type accumulation_type = GGML_TYPE_F32;
    int32_t axis = -1;
    int32_t block_size = 0;
    ops_weight_layout layout = ops_weight_layout::native;
};

inline ops_quantization_desc ops_describe_quantization(
    const ggml_tensor* tensor,
    ops_weight_layout layout = ops_weight_layout::native
) {
    ops_quantization_desc desc;
    if (!tensor) return desc;
    desc.storage_type = tensor->type;
    desc.layout = layout;
    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
            desc.scheme = ops_quant_scheme::q4_0;
            desc.block_size = 32;
            break;
        case GGML_TYPE_Q4_1:
            desc.scheme = ops_quant_scheme::q4_1;
            desc.block_size = 32;
            break;
        case GGML_TYPE_Q5_0:
            desc.scheme = ops_quant_scheme::q5_0;
            desc.block_size = 32;
            break;
        case GGML_TYPE_Q5_1:
            desc.scheme = ops_quant_scheme::q5_1;
            desc.block_size = 32;
            break;
        case GGML_TYPE_Q8_0:
            desc.scheme = ops_quant_scheme::q8_0;
            desc.block_size = 32;
            break;
        case GGML_TYPE_Q4_K:
            desc.scheme = ops_quant_scheme::q4_k;
            desc.block_size = 256;
            break;
        case GGML_TYPE_Q2_K:
            desc.scheme = ops_quant_scheme::q2_k;
            desc.block_size = 256;
            break;
        case GGML_TYPE_Q3_K:
            desc.scheme = ops_quant_scheme::q3_k;
            desc.block_size = 256;
            break;
        case GGML_TYPE_Q5_K:
            desc.scheme = ops_quant_scheme::q5_k;
            desc.block_size = 256;
            break;
        case GGML_TYPE_Q6_K:
            desc.scheme = ops_quant_scheme::q6_k;
            desc.block_size = 256;
            break;
        case GGML_TYPE_IQ4_NL:
            desc.scheme = ops_quant_scheme::iq4_nl;
            desc.block_size = 32;
            break;
        case GGML_TYPE_IQ4_XS:
            desc.scheme = ops_quant_scheme::iq4_xs;
            desc.block_size = 256;
            break;
        case GGML_TYPE_MXFP4:
            desc.scheme = ops_quant_scheme::mxfp4;
            desc.block_size = 32;
            break;
        default:
            break;
    }
    return desc;
}

// Kernels that fold ne[2]*ne[3] into one batch axis index it with nb[2] alone;
// that is only valid when dims 2 and 3 are packed together in memory.
inline bool ops_batch_fold_packed(const ggml_tensor* tensor) {
    return tensor->ne[3] <= 1 ||
           tensor->nb[3] == static_cast<size_t>(tensor->ne[2]) * tensor->nb[2];
}

struct ops_request {
    ggml_backend_dev_t device = nullptr;
    int op_id = 0;
    ggml_tensor* const* srcs = nullptr;
    int n_srcs = 0;
    const void* params = nullptr;
    size_t params_size = 0;
    ggml_tensor* output = nullptr;
};

struct ops_execution_context {
    ggml_backend_t backend = nullptr;
    ggml_backend_dev_t device = nullptr;
    void* stream = nullptr;
    void* scratch = nullptr;
    size_t scratch_size = 0;
};

struct ops_probe_result {
    bool supported = false;
    size_t workspace_bytes = 0;
    const char* reason = nullptr;

    constexpr ops_probe_result() = default;
    constexpr ops_probe_result(bool supported) : supported(supported) {}
    constexpr ops_probe_result(bool supported, size_t workspace_bytes, const char* reason)
        : supported(supported), workspace_bytes(workspace_bytes), reason(reason) {}
};

} // namespace ggml_ops_ext
