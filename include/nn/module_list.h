#pragma once

#include "nn/module.h"

#include <cstddef>
#include <string>
#include <vector>

namespace nn {

template <typename T>
class ModuleList final : public Module<ModuleList<T>> {
public:
    template <typename... Args>
    T& emplace_back(Args&&... args) {
        T& item = this->template submodule<T>(
            std::to_string(items_.size()), std::forward<Args>(args)...);
        items_.push_back(&item);
        return item;
    }

    T& operator[](size_t index) { return *items_.at(index); }
    const T& operator[](size_t index) const { return *items_.at(index); }
    size_t size() const noexcept { return items_.size(); }
    bool empty() const noexcept { return items_.empty(); }

    void clear() noexcept {
        items_.clear();
        this->clear_submodules();
    }

    class iterator {
    public:
        explicit iterator(typename std::vector<T*>::iterator value) : value_(value) {}
        T& operator*() const { return **value_; }
        iterator& operator++() { ++value_; return *this; }
        bool operator!=(const iterator& other) const { return value_ != other.value_; }
    private:
        typename std::vector<T*>::iterator value_;
    };

    iterator begin() { return iterator(items_.begin()); }
    iterator end() { return iterator(items_.end()); }

private:
    std::vector<T*> items_;
};

} // namespace nn
