#pragma once

#include "nn/module.h"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace nn {

class ParameterDict final : public Module {
public:
    Parameter& define(std::string name, Parameter::Spec spec = {}) {
        if (name.empty()) throw std::invalid_argument("parameter name cannot be empty");
        if (parameters_.find(name) != parameters_.end()) {
            throw std::invalid_argument("parameter is already defined: " + name);
        }
        auto parameter = std::make_unique<Parameter>(std::move(spec));
        Parameter* result = parameter.get();
        const std::string registered_name = "item_" + std::to_string(parameters_.size());
        registered_names_.emplace(registered_name, name);
        parameters_.emplace(std::move(name), std::move(parameter));
        register_parameter(registered_name, *result);
        return *result;
    }

    Parameter* find(std::string_view name) noexcept {
        auto found = parameters_.find(std::string(name));
        return found == parameters_.end() ? nullptr : found->second.get();
    }

    const Parameter* find(std::string_view name) const noexcept {
        auto found = parameters_.find(std::string(name));
        return found == parameters_.end() ? nullptr : found->second.get();
    }

    Parameter& at(std::string_view name) {
        Parameter* parameter = find(name);
        if (!parameter) throw std::out_of_range("parameter is not defined: " + std::string(name));
        return *parameter;
    }

    const Parameter& at(std::string_view name) const {
        const Parameter* parameter = find(name);
        if (!parameter) throw std::out_of_range("parameter is not defined: " + std::string(name));
        return *parameter;
    }

    bool contains(std::string_view name) const noexcept { return find(name) != nullptr; }
    size_t size() const noexcept { return parameters_.size(); }

    std::optional<std::string_view> key_for_registered_name(std::string_view name) const noexcept {
        auto found = registered_names_.find(std::string(name));
        return found == registered_names_.end() ? std::nullopt
                                                : std::optional<std::string_view>(found->second);
    }

private:
    std::unordered_map<std::string, std::unique_ptr<Parameter>> parameters_;
    std::unordered_map<std::string, std::string> registered_names_;
};

} // namespace nn
