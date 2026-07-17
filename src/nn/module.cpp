#include "nn/module.h"

#include <stdexcept>

namespace nn {

void Module::register_parameter(std::string name, Parameter& parameter) {
    if (name.empty() || name.find('.') != std::string::npos) throw std::invalid_argument("invalid parameter name");
    for (const auto& item : parameters_) if (item.first == name) throw std::invalid_argument("duplicate parameter name");
    for (const auto& item : children_) if (item.first == name) throw std::invalid_argument("module and parameter names must be unique");
    parameters_.emplace_back(std::move(name), &parameter);
}

void Module::register_module(std::string name, Module& module) {
    if (name.empty()) throw std::invalid_argument("invalid module name");
    for (const auto& item : children_) if (item.first == name) throw std::invalid_argument("duplicate module name");
    for (const auto& item : parameters_) if (item.first == name) throw std::invalid_argument("module and parameter names must be unique");
    children_.emplace_back(std::move(name), &module);
}

void Module::register_module(std::string name, Module* module) {
    if (!module) throw std::invalid_argument("child module cannot be null");
    register_module(std::move(name), *module);
}

void Module::visit(const std::string& prefix, const ParameterVisitor& visitor) {
    for (auto& item : parameters_) visitor(prefix.empty() ? item.first : prefix + "." + item.first, *item.second);
    for (auto& item : children_) item.second->visit(prefix.empty() ? item.first : prefix + "." + item.first, visitor);
}

void Module::visit(const std::string& prefix, const ConstParameterVisitor& visitor) const {
    for (const auto& item : parameters_) visitor(prefix.empty() ? item.first : prefix + "." + item.first, *item.second);
    for (const auto& item : children_) static_cast<const Module*>(item.second)->visit(
        prefix.empty() ? item.first : prefix + "." + item.first, visitor);
}

void Module::for_each_parameter(const ParameterVisitor& visitor) { visit("", visitor); }
void Module::for_each_parameter(const ConstParameterVisitor& visitor) const { visit("", visitor); }

Parameter* Module::find_parameter(std::string_view path) {
    Parameter* result = nullptr;
    for_each_parameter([&](std::string_view name, Parameter& parameter) {
        if (name == path) result = &parameter;
    });
    return result;
}

size_t Module::parameter_count() const {
    size_t count = 0;
    for_each_parameter([&](std::string_view, const Parameter&) { ++count; });
    return count;
}

void Module::to(ggml_backend_t target) {
    backend = target;
    for (auto& child : children_) child.second->to(target);
}

void Module::attach_state_dict(std::shared_ptr<const StateDict> state) {
    if (!state) throw std::invalid_argument("state dict cannot be null");
    state_dict_ = std::move(state);
}

void Module::clear_registered_modules() noexcept {
    children_.clear();
}

} // namespace nn
