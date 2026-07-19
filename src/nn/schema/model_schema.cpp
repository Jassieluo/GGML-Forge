#include "nn/schema/model_schema.h"

#include "nn/core/module.h"

#include <sstream>

namespace nn {

namespace {

const char* schema_storage_type_name(ggml_type type) noexcept {
    switch (type) {
        case GGML_TYPE_Q2_K: return "Q2_K";
        case GGML_TYPE_Q3_K: return "Q3_K";
        case GGML_TYPE_Q4_K: return "Q4_K";
        case GGML_TYPE_Q5_K: return "Q5_K";
        case GGML_TYPE_Q6_K: return "Q6_K";
        case GGML_TYPE_Q4_0: return "Q4_0";
        case GGML_TYPE_Q4_1: return "Q4_1";
        case GGML_TYPE_Q5_0: return "Q5_0";
        case GGML_TYPE_Q5_1: return "Q5_1";
        case GGML_TYPE_Q8_0: return "Q8_0";
        case GGML_TYPE_IQ4_NL: return "IQ4_NL";
        case GGML_TYPE_IQ4_XS: return "IQ4_XS";
        case GGML_TYPE_MXFP4: return "MXFP4";
        case GGML_TYPE_BF16: return "BF16";
        case GGML_TYPE_F16: return "F16";
        case GGML_TYPE_F32: return "F32";
        default: return "UNKNOWN";
    }
}

} // namespace

bool ParameterSchema::supports_direct_storage(ggml_type type) const noexcept {
    for (ggml_type candidate : direct_storage_types) {
        if (candidate == type) return true;
    }
    return false;
}

ModelSchema ModelSchema::from(const ModuleBase& module) {
    ModelSchema result;
    module.for_each_parameter([&](std::string_view path, const Parameter& parameter) {
        const auto capability = parameter.storage_capability();
        ParameterSchema item;
        item.path = path;
        item.logical_shape = parameter.spec().logical_shape;
        item.presence = parameter.spec().presence;
        item.usage = parameter.spec().usage;
        item.quantized_layout = capability.quantized_layout;
        item.direct_storage_types.reserve(capability.direct_type_count);
        for (size_t i = 0; i < capability.direct_type_count; ++i) {
            item.direct_storage_types.push_back(capability.direct_types[i]);
        }
        result.parameters_.push_back(std::move(item));
    });
    return result;
}

const ParameterSchema* ModelSchema::find(std::string_view path) const noexcept {
    for (const ParameterSchema& parameter : parameters_) {
        if (parameter.path == path) return &parameter;
    }
    return nullptr;
}

std::string ModelSchema::to_json() const {
    std::ostringstream output;
    output << "{\"parameters\":[";
    for (size_t index = 0; index < parameters_.size(); ++index) {
        const ParameterSchema& parameter = parameters_[index];
        if (index != 0) output << ',';
        output << "{\"path\":\"" << parameter.path
               << "\",\"required\":" << (parameter.required() ? "true" : "false")
               << ",\"usage\":\"" << parameter_usage_name(parameter.usage)
               << "\",\"direct_storage_types\":[";
        for (size_t type_index = 0; type_index < parameter.direct_storage_types.size(); ++type_index) {
            if (type_index != 0) output << ',';
            output << '\"' << schema_storage_type_name(parameter.direct_storage_types[type_index]) << '\"';
        }
        output << "],\"quantized_layout\":\"" << weight_layout_name(parameter.quantized_layout) << '\"';
        if (parameter.logical_shape) {
            output << ",\"logical_shape\":[";
            for (size_t dim = 0; dim < parameter.logical_shape->size(); ++dim) {
                if (dim != 0) output << ',';
                output << (*parameter.logical_shape)[dim];
            }
            output << ']';
        }
        output << '}';
    }
    output << "]}";
    return output.str();
}

const char* parameter_usage_name(Parameter::Usage usage) noexcept {
    using U = Parameter::Usage;
    switch (usage) {
        case U::generic: return "generic";
        case U::opaque_storage: return "opaque_storage";
        case U::linear_weight: return "linear_weight";
        case U::embedding_weight: return "embedding_weight";
        case U::conv1d_weight: return "conv1d_weight";
        case U::conv2d_weight: return "conv2d_weight";
        case U::conv_transpose1d_weight: return "conv_transpose1d_weight";
        case U::conv_transpose2d_weight: return "conv_transpose2d_weight";
        case U::conv3d_weight: return "conv3d_weight";
        case U::conv_transpose3d_weight: return "conv_transpose3d_weight";
        case U::bias: return "bias";
        case U::norm_affine: return "norm_affine";
        case U::scalar: return "scalar";
        case U::relative_position: return "relative_position";
    }
    return "unknown";
}

const char* weight_layout_name(ggml_ops_ext::ops_weight_layout layout) noexcept {
    using L = ggml_ops_ext::ops_weight_layout;
    switch (layout) {
        case L::native: return "native";
        case L::channel_rows: return "channel_rows";
        case L::flattened_rows: return "flattened_rows";
        case L::flexible_rows: return "flexible_rows";
        case L::backend_prepared: return "backend_prepared";
    }
    return "unknown";
}

} // namespace nn
