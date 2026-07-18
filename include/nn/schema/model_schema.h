#pragma once

#include "nn/core/parameter.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace nn {

class ModuleBase;

struct ParameterSchema {
    std::string path;
    std::optional<Shape> logical_shape;
    Parameter::Presence presence = Parameter::Presence::required;
    Parameter::Usage usage = Parameter::Usage::generic;
    std::vector<ggml_type> direct_storage_types;
    ggml_ops_ext::ops_weight_layout quantized_layout =
        ggml_ops_ext::ops_weight_layout::native;

    bool required() const noexcept { return presence == Parameter::Presence::required; }
    bool supports_direct_storage(ggml_type type) const noexcept;
};

class ModelSchema {
public:
    static ModelSchema from(const ModuleBase& module);

    const std::vector<ParameterSchema>& parameters() const noexcept { return parameters_; }
    const ParameterSchema* find(std::string_view path) const noexcept;
    std::string to_json() const;

private:
    std::vector<ParameterSchema> parameters_;
};

const char* parameter_usage_name(Parameter::Usage usage) noexcept;
const char* weight_layout_name(ggml_ops_ext::ops_weight_layout layout) noexcept;

} // namespace nn
