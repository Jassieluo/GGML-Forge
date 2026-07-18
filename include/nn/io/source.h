#pragma once

#include "ggml.h"
#include "nn/core/layout.h"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace nn::io {

struct TensorInfo {
    std::string name;
    ggml_type storage_type = GGML_TYPE_COUNT;
    Shape storage_shape;
    Shape logical_shape;
    Layout layout = Layout::identity();
    size_t bytes = 0;
};

class Source {
public:
    virtual ~Source() = default;
    virtual size_t size() const = 0;
    virtual const TensorInfo& info(size_t index) const = 0;
    virtual std::optional<size_t> find(std::string_view name) const = 0;
    virtual bool read(size_t index, size_t offset, void* destination, size_t bytes) = 0;
};

} // namespace nn::io
