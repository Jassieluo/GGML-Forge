#pragma once

#include "nn/core/parameter.h"
#include "ggml-backend.h"

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace nn {

class StateDict;
class ModelSchema;

class ModuleBase {
public:
    using ParameterVisitor = std::function<void(std::string_view, Parameter&)>;
    using ConstParameterVisitor = std::function<void(std::string_view, const Parameter&)>;

    ModuleBase() = default;
    virtual ~ModuleBase() = default;
    ModuleBase(const ModuleBase&) = delete;
    ModuleBase& operator=(const ModuleBase&) = delete;
    ModuleBase(ModuleBase&&) = delete;
    ModuleBase& operator=(ModuleBase&&) = delete;

    void for_each_parameter(const ParameterVisitor& visitor);
    void for_each_parameter(const ConstParameterVisitor& visitor) const;
    Parameter* find_parameter(std::string_view path);
    size_t parameter_count() const;
    ModelSchema schema() const;

    void to(ggml_backend_t backend);
    ggml_backend_t backend = nullptr;

    std::shared_ptr<const StateDict> state_dict() const noexcept { return state_dict_; }
    void attach_state_dict(std::shared_ptr<const StateDict> state);

protected:
    template <typename T, typename... Args>
    T& submodule(std::string name, Args&&... args) {
        static_assert(std::is_base_of_v<ModuleBase, T>, "submodule type must derive from nn::Module");
        if (name.empty() || name.find('.') != std::string::npos) {
            throw std::invalid_argument("submodule name must be one path segment");
        }
        auto child = std::make_unique<T>(std::forward<Args>(args)...);
        T& result = *child;
        register_module(std::move(name), result);
        owned_children_.push_back(std::move(child));
        return result;
    }

    Parameter& parameter(std::string name, Parameter value = {});
    void clear_submodules() noexcept;

private:
    void register_parameter(std::string name, Parameter& parameter);
    void register_module(std::string name, ModuleBase& module);

    std::vector<std::pair<std::string, Parameter*>> parameters_;
    std::vector<std::pair<std::string, ModuleBase*>> children_;
    std::vector<std::unique_ptr<Parameter>> owned_parameters_;
    std::vector<std::unique_ptr<ModuleBase>> owned_children_;
    std::shared_ptr<const StateDict> state_dict_;

    void visit(const std::string& prefix, const ParameterVisitor& visitor);
    void visit(const std::string& prefix, const ConstParameterVisitor& visitor) const;
};

template <typename Derived>
class Module : public ModuleBase {
public:
    template <typename... Args>
    decltype(auto) operator()(Args&&... args) {
        return static_cast<Derived&>(*this).forward(std::forward<Args>(args)...);
    }
};

} // namespace nn
