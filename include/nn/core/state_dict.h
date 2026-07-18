#pragma once

#include "ggml-backend.h"
#include "nn/core/context.h"
#include "nn/core/layout.h"

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>

namespace nn {

class StateDict {
public:
    struct Entry {
        std::string parameter_path;
        std::string source_name;
        ggml_tensor* tensor = nullptr;
        Shape logical_shape;
        Layout layout = Layout::identity();
    };

    ~StateDict();
    StateDict(const StateDict&) = delete;
    StateDict& operator=(const StateDict&) = delete;

    const Entry* find(std::string_view parameter_path) const;
    size_t size() const noexcept { return entries_.size(); }

private:
    friend class StateDictBuilder;
    explicit StateDict(size_t context_size);

    std::unique_ptr<Context> context_;
    ggml_backend_buffer_t buffer_ = nullptr;
    std::unordered_map<std::string, Entry> entries_;
};

} // namespace nn
