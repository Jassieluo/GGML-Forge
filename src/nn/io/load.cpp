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

} // namespace

LoadResult load_into(ModuleBase& module, Source& source, ggml_backend_t backend) {
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
        const std::string source_name(path);
        auto index = source.find(source_name);
        if (!index) {
            if (parameter.is_required()) validation_error = "required parameter is missing: " + std::string(path);
            return;
        }
        if (!used.insert(*index).second) {
            validation_error = "source tensor is bound more than once: " + source_name;
            return;
        }
        const TensorInfo& info = source.info(*index);
        if (info.storage_type < 0 || info.storage_type >= GGML_TYPE_COUNT) {
            validation_error = "invalid source tensor type: " + info.name;
            return;
        }
        if (!parameter.supports_direct_storage(info.storage_type)) {
            validation_error = "parameter storage is not directly executable: " +
                               std::string(path) + " (" + ggml_type_name(info.storage_type) + ")";
            return;
        }
        const bool quantized = ggml_is_quantized(info.storage_type);
        if (quantized) {
            if (info.storage_shape.empty() || info.storage_shape[0] <= 0 ||
                info.storage_shape[0] % ggml_blck_size(info.storage_type) != 0) {
                validation_error = "quantized parameter row is incompatible with its block size: " +
                                   std::string(path);
                return;
            }
            const auto capability = parameter.storage_capability();
            if (capability.quantized_layout == ggml_ops_ext::ops_weight_layout::channel_rows &&
                info.layout.axes() != std::vector<int>({1, 0, 2})) {
                validation_error = "quantized convolution parameter requires channel-row storage: " +
                                   std::string(path);
                return;
            }
            if (capability.quantized_layout == ggml_ops_ext::ops_weight_layout::flattened_rows &&
                (info.storage_shape.size() != 2 || info.logical_shape.size() != 4 ||
                 info.storage_shape[0] != info.logical_shape[0] * info.logical_shape[1] * info.logical_shape[2] ||
                 info.storage_shape[1] != info.logical_shape[3])) {
                validation_error = "quantized 2D convolution requires flattened-row storage: " +
                                   std::string(path);
                return;
            }
            if (capability.quantized_layout == ggml_ops_ext::ops_weight_layout::flexible_rows) {
                const bool channel_rows = info.storage_shape.size() == 3 && info.logical_shape.size() == 4 &&
                    info.storage_shape[0] == info.logical_shape[2] &&
                    info.storage_shape[1] == info.logical_shape[0] * info.logical_shape[1] &&
                    info.storage_shape[2] == info.logical_shape[3];
                const bool flattened_rows = info.storage_shape.size() == 2 && info.logical_shape.size() == 4 &&
                    info.storage_shape[0] == info.logical_shape[0] * info.logical_shape[1] * info.logical_shape[2] &&
                    info.storage_shape[1] == info.logical_shape[3];
                if (!channel_rows && !flattened_rows) {
                    validation_error = "quantized 2D convolution requires channel-row or flattened-row storage: " +
                                       std::string(path);
                    return;
                }
            }
        }
        const Shape logical = info.logical_shape.empty() ? info.layout.logical_shape(info.storage_shape) : info.logical_shape;
        if (parameter.spec().logical_shape && *parameter.spec().logical_shape != logical) {
            validation_error = "parameter shape mismatch: " + std::string(path);
            return;
        }
        pending.push_back({std::string(path), &parameter, *index, info.storage_type});
    });
    if (!validation_error.empty()) return {{}, validation_error};
    if (used.size() != source.size()) {
        std::string unexpected;
        for (size_t index = 0; index < source.size(); ++index) {
            if (used.find(index) != used.end()) continue;
            if (!unexpected.empty()) unexpected += ", ";
            unexpected += source.info(index).name;
        }
        return {{}, "source contains unexpected parameters: " + unexpected};
    }

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
        const bool uploaded = upload_native(source, binding, info, entry->tensor);
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

} // namespace io
} // namespace nn
