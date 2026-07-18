#pragma once

#include "ggml.h"
#include "nn/layout.h"
#include "ops/contracts/quantization.h"

#include <optional>

namespace nn {

class Parameter {
public:
    using Usage = ggml_ops_ext::ops_parameter_usage;

    enum class Presence {
        required,
        optional,
    };

    struct Spec {
        std::optional<Shape> logical_shape;
        Presence presence = Presence::required;
        Usage usage = Usage::generic;
    };

    Parameter() = default;
    explicit Parameter(Spec spec) : spec_(std::move(spec)) {}

    static Parameter required(
        std::optional<Shape> shape = std::nullopt,
        Usage usage = Usage::generic);
    static Parameter optional(
        std::optional<Shape> shape = std::nullopt,
        Usage usage = Usage::generic);

    bool is_bound() const noexcept { return tensor_ != nullptr; }
    bool is_required() const noexcept { return spec_.presence == Presence::required; }
    bool is_tied() const noexcept { return tied_to_ != nullptr; }

    const Spec& spec() const noexcept { return spec_; }
    ggml_tensor* local_tensor() const noexcept { return tensor_; }
    ggml_tensor* tensor() const;
    const Shape& logical_shape() const noexcept { return logical_shape_; }
    const Layout& layout() const noexcept { return layout_; }
    ggml_type storage_type() const noexcept { return tensor_ ? tensor_->type : GGML_TYPE_COUNT; }
    ggml_ops_ext::ops_storage_capability storage_capability() const noexcept;
    bool supports_direct_storage(ggml_type type) const noexcept;

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
