#pragma once

#include "nn/parameter.h"
#include "ggml-backend.h"

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace nn {

class StateDict;

class Module {
public:
    using ParameterVisitor = std::function<void(std::string_view, Parameter&)>;
    using ConstParameterVisitor = std::function<void(std::string_view, const Parameter&)>;

    Module() = default;
    virtual ~Module() = default;
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
    Module(Module&&) = delete;
    Module& operator=(Module&&) = delete;

    void register_parameter(std::string name, Parameter& parameter);
    void register_module(std::string name, Module& module);
    void register_module(std::string name, Module* module);

    void for_each_parameter(const ParameterVisitor& visitor);
    void for_each_parameter(const ConstParameterVisitor& visitor) const;
    Parameter* find_parameter(std::string_view path);
    size_t parameter_count() const;

    void to(ggml_backend_t backend);
    ggml_backend_t backend = nullptr;

    std::shared_ptr<const StateDict> state_dict() const noexcept { return state_dict_; }
    void attach_state_dict(std::shared_ptr<const StateDict> state);

protected:
    void clear_registered_modules() noexcept;

private:
    std::vector<std::pair<std::string, Parameter*>> parameters_;
    std::vector<std::pair<std::string, Module*>> children_;
    std::shared_ptr<const StateDict> state_dict_;

    void visit(const std::string& prefix, const ParameterVisitor& visitor);
    void visit(const std::string& prefix, const ConstParameterVisitor& visitor) const;
};

} // namespace nn
