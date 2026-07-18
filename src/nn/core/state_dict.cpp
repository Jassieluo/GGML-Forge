#include "nn/core/state_dict.h"

namespace nn {

StateDict::StateDict(size_t context_size)
    : context_(std::make_unique<Context>(context_size)) {}

StateDict::~StateDict() {
    if (buffer_) ggml_backend_buffer_free(buffer_);
}

const StateDict::Entry* StateDict::find(std::string_view parameter_path) const {
    auto it = entries_.find(std::string(parameter_path));
    return it == entries_.end() ? nullptr : &it->second;
}

} // namespace nn
