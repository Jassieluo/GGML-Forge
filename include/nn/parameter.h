#pragma once

#include "ggml.h"
#include "nn/layout.h"

#include <optional>

namespace nn {

class Parameter {
public:
    enum class StoragePolicy {
        native,
        floating,
    };

    enum class Presence {
        required,
        optional,
    };

    struct Spec {
        std::optional<Shape> logical_shape;
        Presence presence = Presence::required;
        StoragePolicy storage_policy = StoragePolicy::native;
    };

    Parameter() = default;
    explicit Parameter(Spec spec) : spec_(std::move(spec)) {}

    static Parameter required(
        std::optional<Shape> shape = std::nullopt,
        StoragePolicy storage_policy = StoragePolicy::native);
    static Parameter optional(
        std::optional<Shape> shape = std::nullopt,
        StoragePolicy storage_policy = StoragePolicy::native);

    bool is_bound() const noexcept { return tensor_ != nullptr; }
    bool is_required() const noexcept { return spec_.presence == Presence::required; }
    bool is_tied() const noexcept { return tied_to_ != nullptr; }

    const Spec& spec() const noexcept { return spec_; }
    ggml_tensor* local_tensor() const noexcept { return tensor_; }
    ggml_tensor* tensor() const;
    const Shape& logical_shape() const noexcept { return logical_shape_; }
    const Layout& layout() const noexcept { return layout_; }
    ggml_type storage_type() const noexcept { return tensor_ ? tensor_->type : GGML_TYPE_COUNT; }

    void tie(Parameter& parameter);
    void resolve_tie();
    void bind(ggml_tensor* tensor);
    void bind(ggml_tensor* tensor, Shape logical_shape, Layout layout = Layout::identity());
    void clear() noexcept;

private:
    Spec spec_;
    ggml_tensor* tensor_ = nullptr;
    Shape logical_shape_;
    Layout layout_ = Layout::identity();
    Parameter* tied_to_ = nullptr;
};

} // namespace nn
