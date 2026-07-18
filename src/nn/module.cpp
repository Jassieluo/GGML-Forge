#include "nn/module.h"
#include "nn/model_schema.h"

#include <stdexcept>

namespace nn {

Parameter& ModuleBase::parameter(std::string name, Parameter value) {
    auto owned = std::make_unique<Parameter>(std::move(value));
    Parameter& result = *owned;
    register_parameter(std::move(name), result);
    owned_parameters_.push_back(std::move(owned));
    return result;
}

void ModuleBase::register_parameter(std::string name, Parameter& parameter) {
    if (name.empty() || name.find('.') != std::string::npos) throw std::invalid_argument("invalid parameter name");
    for (const auto& item : parameters_) if (item.first == name) throw std::invalid_argument("duplicate parameter name");
    for (const auto& item : children_) if (item.first == name) throw std::invalid_argument("module and parameter names must be unique");
    parameters_.emplace_back(std::move(name), &parameter);
}

void ModuleBase::register_module(std::string name, ModuleBase& module) {
    if (name.empty()) throw std::invalid_argument("invalid module name");
    for (const auto& item : children_) if (item.first == name) throw std::invalid_argument("duplicate module name");
    for (const auto& item : parameters_) if (item.first == name) throw std::invalid_argument("module and parameter names must be unique");
    children_.emplace_back(std::move(name), &module);
}

void ModuleBase::visit(const std::string& prefix, const ParameterVisitor& visitor) {
    for (auto& item : parameters_) visitor(prefix.empty() ? item.first : prefix + "." + item.first, *item.second);
    for (auto& item : children_) item.second->visit(prefix.empty() ? item.first : prefix + "." + item.first, visitor);
}

void ModuleBase::visit(const std::string& prefix, const ConstParameterVisitor& visitor) const {
    for (const auto& item : parameters_) visitor(prefix.empty() ? item.first : prefix + "." + item.first, *item.second);
    for (const auto& item : children_) static_cast<const ModuleBase*>(item.second)->visit(
        prefix.empty() ? item.first : prefix + "." + item.first, visitor);
}

void ModuleBase::for_each_parameter(const ParameterVisitor& visitor) { visit("", visitor); }
void ModuleBase::for_each_parameter(const ConstParameterVisitor& visitor) const { visit("", visitor); }

Parameter* ModuleBase::find_parameter(std::string_view path) {
    Parameter* result = nullptr;
    for_each_parameter([&](std::string_view name, Parameter& parameter) {
        if (name == path) result = &parameter;
    });
    return result;
}

size_t ModuleBase::parameter_count() const {
    size_t count = 0;
    for_each_parameter([&](std::string_view, const Parameter&) { ++count; });
    return count;
}

ModelSchema ModuleBase::schema() const {
    return ModelSchema::from(*this);
}

void ModuleBase::to(ggml_backend_t target) {
    backend = target;
    for (auto& child : children_) child.second->to(target);
}

void ModuleBase::attach_state_dict(std::shared_ptr<const StateDict> state) {
    if (!state) throw std::invalid_argument("state dict cannot be null");
    state_dict_ = std::move(state);
}

void ModuleBase::clear_submodules() noexcept {
    children_.clear();
    owned_children_.clear();
}

} // namespace nn
