#pragma once

#include "nn/core/module.h"

#include <string>
#include <unordered_map>

namespace nn {

template <typename T>
class ModuleDict final : public Module<ModuleDict<T>> {
public:
    template <typename... Args>
    T& emplace(std::string name, Args&&... args) {
        if (items_.find(name) != items_.end()) throw std::invalid_argument("duplicate module dictionary key");
        T& item = this->template submodule<T>(name, std::forward<Args>(args)...);
        items_.emplace(std::move(name), &item);
        return item;
    }

    T& at(const std::string& name) { return *items_.at(name); }
    const T& at(const std::string& name) const { return *items_.at(name); }
    bool contains(const std::string& name) const { return items_.find(name) != items_.end(); }
    size_t size() const noexcept { return items_.size(); }

private:
    std::unordered_map<std::string, T*> items_;
};

} // namespace nn
