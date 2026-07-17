#include "nn/parameter.h"

#include <stdexcept>
#include <utility>

namespace nn {

Layout Layout::identity() {
    return Layout({});
}

Layout Layout::permuted(std::initializer_list<int> axes) {
    return permuted(std::vector<int>(axes));
}

Layout Layout::permuted(const std::vector<int>& axes) {
    if (axes.empty()) throw std::invalid_argument("layout permutation cannot be empty");
    std::vector<bool> seen(axes.size(), false);
    for (int axis : axes) {
        if (axis < 0 || static_cast<size_t>(axis) >= axes.size() || seen[static_cast<size_t>(axis)]) {
            throw std::invalid_argument("layout axes must be a permutation");
        }
        seen[static_cast<size_t>(axis)] = true;
    }
    bool identity = true;
    for (size_t i = 0; i < axes.size(); ++i) identity = identity && axes[i] == static_cast<int>(i);
    return Layout(identity ? std::vector<int>{} : axes);
}

Shape Layout::logical_shape(const Shape& storage_shape) const {
    if (is_identity()) return storage_shape;
    if (axes_.size() != storage_shape.size()) throw std::invalid_argument("layout rank does not match storage shape");
    Shape result(storage_shape.size());
    for (size_t i = 0; i < axes_.size(); ++i) result[i] = storage_shape[static_cast<size_t>(axes_[i])];
    return result;
}

Parameter Parameter::required(std::optional<Shape> shape, StoragePolicy storage_policy) {
    return Parameter({std::move(shape), Presence::required, storage_policy});
}

Parameter Parameter::optional(std::optional<Shape> shape, StoragePolicy storage_policy) {
    return Parameter({std::move(shape), Presence::optional, storage_policy});
}

ggml_tensor* Parameter::tensor() const {
    if (!tensor_) throw std::logic_error("parameter is not bound");
    return tensor_;
}

void Parameter::tie(Parameter& parameter) {
    if (&parameter == this) throw std::invalid_argument("parameter cannot be tied to itself");
    if (is_bound()) throw std::logic_error("bound parameter cannot be tied");
    if (parameter.is_tied()) throw std::invalid_argument("parameter tie chains are not supported");
    if (spec_.logical_shape && parameter.spec_.logical_shape &&
        *spec_.logical_shape != *parameter.spec_.logical_shape) {
        throw std::invalid_argument("tied parameters have incompatible logical shapes");
    }
    tied_to_ = &parameter;
}

void Parameter::resolve_tie() {
    if (!tied_to_) throw std::logic_error("parameter is not tied");
    if (!tied_to_->is_bound()) throw std::logic_error("tied parameter target is not bound");
    tensor_ = tied_to_->tensor_;
    logical_shape_ = tied_to_->logical_shape_;
    layout_ = tied_to_->layout_;
}

void Parameter::bind(ggml_tensor* tensor) {
    if (!tensor) throw std::invalid_argument("parameter tensor cannot be null");
    Shape shape;
    const int rank = ggml_n_dims(tensor);
    shape.reserve(static_cast<size_t>(rank));
    for (int dim = 0; dim < rank; ++dim) shape.push_back(tensor->ne[dim]);
    bind(tensor, std::move(shape));
}

void Parameter::bind(ggml_tensor* tensor, Shape logical_shape, Layout layout) {
    if (!tensor) throw std::invalid_argument("parameter tensor cannot be null");
    if (is_bound()) throw std::logic_error("parameter is already bound");
    if (tied_to_) throw std::logic_error("tied parameter must be resolved from its target");
    if (spec_.logical_shape && *spec_.logical_shape != logical_shape) {
        throw std::invalid_argument("parameter logical shape does not match its specification");
    }
    tensor_ = tensor;
    logical_shape_ = std::move(logical_shape);
    layout_ = std::move(layout);
}

void Parameter::clear() noexcept {
    tensor_ = nullptr;
    logical_shape_.clear();
    layout_ = Layout::identity();
}

} // namespace nn
