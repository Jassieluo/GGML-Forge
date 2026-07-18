#pragma once

#include "nn/core/module.h"

#include <string>
#include <vector>

namespace nn {

class ParameterList final : public Module<ParameterList> {
public:
    Parameter& append(Parameter value = {}) {
        Parameter& item = parameter(std::to_string(items_.size()), std::move(value));
        items_.push_back(&item);
        return item;
    }

    Parameter& operator[](size_t index) { return *items_.at(index); }
    const Parameter& operator[](size_t index) const { return *items_.at(index); }
    size_t size() const noexcept { return items_.size(); }

private:
    std::vector<Parameter*> items_;
};

} // namespace nn
