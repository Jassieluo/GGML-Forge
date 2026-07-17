#pragma once

#include "nn/module.h"
#include "nn/parameter_dict.h"
#include "nn/state_dict.h"
#include "nn/io/source.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace nn::io {

using NameMapper = std::function<std::optional<std::string>(std::string_view, const Parameter&)>;

struct LoadResult {
    std::shared_ptr<const StateDict> state;
    std::string error;

    explicit operator bool() const noexcept { return static_cast<bool>(state); }
};

LoadResult load_into(Module& module, Source& source, ggml_backend_t backend, NameMapper mapper = {});
std::string bind_from(Module& module, const ParameterDict& source, NameMapper mapper = {});

} // namespace nn::io
