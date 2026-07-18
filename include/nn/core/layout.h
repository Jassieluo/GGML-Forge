#pragma once

#include "nn/core/types.h"

#include <initializer_list>
#include <utility>
#include <vector>

namespace nn {

class Layout {
public:
    static Layout identity();
    static Layout permuted(std::initializer_list<int> axes);
    static Layout permuted(const std::vector<int>& axes);

    bool is_identity() const noexcept { return axes_.empty(); }
    const std::vector<int>& axes() const noexcept { return axes_; }
    Shape logical_shape(const Shape& storage_shape) const;

private:
    explicit Layout(std::vector<int> axes) : axes_(std::move(axes)) {}
    std::vector<int> axes_;
};

} // namespace nn
