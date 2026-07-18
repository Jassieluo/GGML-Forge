#pragma once

#include "nn/core/module.h"
#include "nn/core/state_dict.h"
#include "nn/io/source.h"

#include <memory>
#include <string>

namespace nn::io {

struct LoadResult {
    std::shared_ptr<const StateDict> state;
    std::string error;

    explicit operator bool() const noexcept { return static_cast<bool>(state); }
};

LoadResult load_into(ModuleBase& module, Source& source, ggml_backend_t backend);

} // namespace nn::io
