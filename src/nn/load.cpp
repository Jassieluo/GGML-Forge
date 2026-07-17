#include "nn/io/load.h"

#include <algorithm>
#include <limits>
#include <unordered_set>
#include <vector>

namespace nn {

class StateDictBuilder {
public:
    static std::shared_ptr<StateDict> create(size_t context_size) {
        return std::shared_ptr<StateDict>(new StateDict(context_size));
    }
    static Context& context(StateDict& state) { return *state.context_; }
    static ggml_backend_buffer_t& buffer(StateDict& state) { return state.buffer_; }
    static std::unordered_map<std::string, StateDict::Entry>& entries(StateDict& state) { return state.entries_; }
};

namespace io {

namespace {

constexpr size_t kReadChunkSize = 16 * 1024 * 1024;

struct PendingBinding {
    std::string path;
    Parameter* parameter = nullptr;
    size_t source_index = 0;
    ggml_type destination_type = GGML_TYPE_COUNT;
};

bool upload_native(Source& source, const PendingBinding& binding, const TensorInfo& info,
                   ggml_tensor* destination) {
    if (ggml_nbytes(destination) != info.bytes) return false;
    std::vector<uint8_t> bytes(std::min(info.bytes, kReadChunkSize));
    for (size_t offset = 0; offset < info.bytes;) {
        const size_t chunk = std::min(bytes.size(), info.bytes - offset);
        if (!source.read(binding.source_index, offset, bytes.data(), chunk)) return false;
        ggml_backend_tensor_set(destination, bytes.data(), offset, chunk);
        offset += chunk;
    }
    return true;
}

bool upload_as_f16(Source& source, const PendingBinding& binding, const TensorInfo& info,
                   ggml_tensor* destination) {
    if (info.storage_shape.empty() || info.storage_shape[0] <= 0) return false;
    const ggml_type_traits* traits = ggml_get_type_traits(info.storage_type);
    if (!traits || !traits->to_float) return false;

    const int64_t row_elements = info.storage_shape[0];
    const int64_t elements = ggml_nelements(destination);
    if (elements <= 0 || elements % row_elements != 0) return false;
    const int64_t rows = elements / row_elements;
    const size_t source_row_bytes = ggml_row_size(info.storage_type, row_elements);
    if (source_row_bytes == 0 || static_cast<uint64_t>(rows) > std::numeric_limits<size_t>::max() / source_row_bytes ||
        static_cast<size_t>(rows) * source_row_bytes != info.bytes) return false;

    std::vector<uint8_t> source_row(source_row_bytes);
    std::vector<float> float_row(static_cast<size_t>(row_elements));
    std::vector<ggml_fp16_t> f16_row(static_cast<size_t>(row_elements));
    const size_t destination_row_bytes = f16_row.size() * sizeof(ggml_fp16_t);
    for (int64_t row = 0; row < rows; ++row) {
        const size_t source_offset = static_cast<size_t>(row) * source_row_bytes;
        if (!source.read(binding.source_index, source_offset, source_row.data(), source_row.size())) return false;
        traits->to_float(source_row.data(), float_row.data(), row_elements);
        ggml_fp32_to_fp16_row(float_row.data(), f16_row.data(), row_elements);
        ggml_backend_tensor_set(destination, f16_row.data(), static_cast<size_t>(row) * destination_row_bytes,
                                destination_row_bytes);
    }
    return true;
}

} // namespace

LoadResult load_into(Module& module, Source& source, ggml_backend_t backend, NameMapper mapper) {
    if (!backend) return {{}, "backend cannot be null"};

    std::unordered_set<std::string> source_names;
    for (size_t i = 0; i < source.size(); ++i) {
        const TensorInfo& info = source.info(i);
        if (info.name.empty() || !source_names.insert(info.name).second) {
            return {{}, "source contains an empty or duplicate tensor name"};
        }
    }

    std::vector<PendingBinding> pending;
    std::unordered_set<size_t> used;
    std::string validation_error;
    module.for_each_parameter([&](std::string_view path, Parameter& parameter) {
        if (!validation_error.empty() || parameter.is_tied()) return;
        if (parameter.is_bound()) {
            validation_error = "parameter is already bound: " + std::string(path);
            return;
        }
        std::optional<std::string> source_name = mapper ? mapper(path, parameter) : std::optional<std::string>(path);
        if (!source_name) {
            if (parameter.is_required()) validation_error = "required parameter has no source mapping: " + std::string(path);
            return;
        }
        auto index = source.find(*source_name);
        if (!index) {
            if (parameter.is_required()) validation_error = "required parameter is missing: " + std::string(path);
            return;
        }
        if (!used.insert(*index).second) {
            validation_error = "source tensor is bound more than once: " + *source_name;
            return;
        }
        const TensorInfo& info = source.info(*index);
        if (info.storage_type < 0 || info.storage_type >= GGML_TYPE_COUNT) {
            validation_error = "invalid source tensor type: " + info.name;
            return;
        }
        const Shape logical = info.logical_shape.empty() ? info.layout.logical_shape(info.storage_shape) : info.logical_shape;
        if (parameter.spec().logical_shape && *parameter.spec().logical_shape != logical) {
            validation_error = "parameter shape mismatch: " + std::string(path);
            return;
        }
        ggml_type destination_type = info.storage_type;
        const ggml_type_traits* traits = ggml_get_type_traits(info.storage_type);
        if (parameter.spec().storage_policy == Parameter::StoragePolicy::floating && traits->is_quantized) {
            if (!traits->to_float) {
                validation_error = "source tensor cannot be converted to floating storage: " + info.name;
                return;
            }
            destination_type = GGML_TYPE_F16;
        }
        pending.push_back({std::string(path), &parameter, *index, destination_type});
    });
    if (!validation_error.empty()) return {{}, validation_error};

    size_t context_size = 1024 * 1024;
    if (pending.size() > (std::numeric_limits<size_t>::max() - context_size) / ggml_tensor_overhead()) {
        return {{}, "parameter context size overflow"};
    }
    context_size += pending.size() * ggml_tensor_overhead();
    auto state = StateDictBuilder::create(context_size);
    StateDict& writable = *state;

    for (const PendingBinding& binding : pending) {
        const TensorInfo& info = source.info(binding.source_index);
        ggml_tensor* tensor = StateDictBuilder::context(writable).empty(binding.path, info.storage_shape, binding.destination_type);
        StateDict::Entry entry;
        entry.parameter_path = binding.path;
        entry.source_name = info.name;
        entry.tensor = tensor;
        entry.layout = info.layout;
        entry.logical_shape = info.logical_shape.empty() ? info.layout.logical_shape(info.storage_shape) : info.logical_shape;
        StateDictBuilder::entries(writable).emplace(binding.path, std::move(entry));
    }
    StateDictBuilder::buffer(writable) = ggml_backend_alloc_ctx_tensors(StateDictBuilder::context(writable).native_handle(), backend);
    if (!StateDictBuilder::buffer(writable)) return {{}, "failed to allocate parameter storage on backend"};

    for (const PendingBinding& binding : pending) {
        const TensorInfo& info = source.info(binding.source_index);
        const StateDict::Entry* entry = writable.find(binding.path);
        if (!entry) return {{}, "missing destination tensor: " + binding.path};
        const bool uploaded = binding.destination_type == info.storage_type
            ? upload_native(source, binding, info, entry->tensor)
            : binding.destination_type == GGML_TYPE_F16 && upload_as_f16(source, binding, info, entry->tensor);
        if (!uploaded) return {{}, "failed to upload source tensor: " + info.name};
    }

    for (const PendingBinding& binding : pending) {
        const StateDict::Entry* entry = writable.find(binding.path);
        binding.parameter->bind(entry->tensor, entry->logical_shape, entry->layout);
    }
    module.for_each_parameter([&](std::string_view, Parameter& parameter) {
        if (parameter.is_tied()) parameter.resolve_tie();
    });
    module.attach_state_dict(state);
    return {std::move(state), {}};
}

std::string bind_from(Module& module, const ParameterDict& source, NameMapper mapper) {
    std::string error;
    module.for_each_parameter([&](std::string_view path, Parameter& parameter) {
        if (!error.empty() || parameter.is_bound() || parameter.is_tied()) return;
        const std::optional<std::string> source_name = mapper
            ? mapper(path, parameter)
            : std::optional<std::string>(path);
        if (!source_name) {
            if (parameter.is_required()) error = "required parameter has no source mapping: " + std::string(path);
            return;
        }
        const Parameter* source_parameter = source.find(*source_name);
        if (!source_parameter || !source_parameter->is_bound()) {
            if (parameter.is_required()) error = "required parameter is missing: " + std::string(path);
            return;
        }
        if (parameter.spec().logical_shape &&
            *parameter.spec().logical_shape != source_parameter->logical_shape()) {
            error = "parameter shape mismatch: " + std::string(path);
            return;
        }
        parameter.bind(
            source_parameter->tensor(),
            source_parameter->logical_shape(),
            source_parameter->layout());
    });
    if (!error.empty()) return error;
    module.for_each_parameter([&](std::string_view, Parameter& parameter) {
        if (parameter.is_tied()) parameter.resolve_tie();
    });
    return {};
}

} // namespace io
} // namespace nn
