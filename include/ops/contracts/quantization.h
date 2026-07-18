#pragma once

#include "ops/types.h"

#include <array>
#include <cstddef>

namespace ggml_ops_ext {

// Describes how a persistent tensor is consumed. This is an execution
// contract, not an export preference: listed types are read directly by the
// operator on every supported project backend.
enum class ops_parameter_usage : uint8_t {
    generic,
    opaque_storage,
    linear_weight,
    embedding_weight,
    conv1d_weight,
    conv2d_weight,
    conv_transpose1d_weight,
    bias,
    norm_affine,
    scalar,
    relative_position,
};

struct ops_storage_capability {
    std::array<ggml_type, 5> direct_types = {
        GGML_TYPE_COUNT, GGML_TYPE_COUNT, GGML_TYPE_COUNT,
        GGML_TYPE_COUNT, GGML_TYPE_COUNT,
    };
    size_t direct_type_count = 0;
    ops_weight_layout quantized_layout = ops_weight_layout::native;

    constexpr bool supports(ggml_type type) const noexcept {
        for (size_t i = 0; i < direct_type_count; ++i) {
            if (direct_types[i] == type) return true;
        }
        return false;
    }
};

constexpr ops_storage_capability ops_direct_storage_capability(
    ops_parameter_usage usage
) noexcept {
    using U = ops_parameter_usage;
    switch (usage) {
        case U::linear_weight:
            return {{{GGML_TYPE_Q4_K, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0,
                      GGML_TYPE_F16, GGML_TYPE_F32}}, 5, ops_weight_layout::native};
        case U::embedding_weight:
            // CUDA GET_ROWS does not directly implement K-quants.
            return {{{GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_F16,
                      GGML_TYPE_F32, GGML_TYPE_COUNT}}, 4, ops_weight_layout::native};
        case U::conv1d_weight:
        case U::conv_transpose1d_weight:
            return {{{GGML_TYPE_Q4_K, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0,
                      GGML_TYPE_F16, GGML_TYPE_F32}}, 5, ops_weight_layout::channel_rows};
        case U::conv2d_weight:
            return {{{GGML_TYPE_Q4_K, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0,
                      GGML_TYPE_F16, GGML_TYPE_F32}}, 5, ops_weight_layout::flattened_rows};
        case U::opaque_storage:
            return {{{GGML_TYPE_Q4_K, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0,
                      GGML_TYPE_F16, GGML_TYPE_F32}}, 5, ops_weight_layout::native};
        case U::generic:
        case U::relative_position:
            return {{{GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_COUNT,
                      GGML_TYPE_COUNT, GGML_TYPE_COUNT}}, 2, ops_weight_layout::native};
        case U::bias:
        case U::norm_affine:
        case U::scalar:
            return {{{GGML_TYPE_F32, GGML_TYPE_COUNT, GGML_TYPE_COUNT,
                      GGML_TYPE_COUNT, GGML_TYPE_COUNT}}, 1, ops_weight_layout::native};
    }
    return {};
}

} // namespace ggml_ops_ext
