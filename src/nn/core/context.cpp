#include "nn/core/context.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace nn {

namespace {

void validate_shape(const Shape& shape) {
    if (shape.size() > GGML_MAX_DIMS) {
        throw std::invalid_argument("tensor shape exceeds GGML_MAX_DIMS");
    }
    for (int64_t dimension : shape) {
        if (dimension < 0) {
            throw std::invalid_argument("tensor dimensions cannot be negative");
        }
    }
}

Shape normalize_shape(const Shape& shape) {
    validate_shape(shape);
    return shape.empty() ? Shape{1} : shape;
}

size_t checked_bytes(size_t count, size_t element_size) {
    if (element_size != 0 && count > std::numeric_limits<size_t>::max() / element_size) {
        throw std::overflow_error("tensor byte count overflow");
    }
    return count * element_size;
}

} // namespace

Context::Context(size_t memory_size, bool no_alloc)
    : Context(nullptr, memory_size, no_alloc) {}

Context::Context(void* memory_buffer, size_t memory_size, bool no_alloc) {
    if (memory_size == 0) {
        throw std::invalid_argument("nn::Context memory size must be greater than zero");
    }
    context_ = ggml_init({memory_size, memory_buffer, no_alloc});
    if (context_ == nullptr) {
        throw std::runtime_error("failed to initialize ggml context");
    }
}

Context::Context(ggml_context* context, bool owns_context)
    : context_(context), owns_context_(owns_context) {
    if (context_ == nullptr) {
        throw std::invalid_argument("cannot borrow a null ggml context");
    }
}

Context Context::borrow(ggml_context* context) {
    return Context(context, false);
}

Context::~Context() {
    if (context_ != nullptr && owns_context_) {
        ggml_free(context_);
    }
}

Context::Context(Context&& other) noexcept
    : context_(std::exchange(other.context_, nullptr)),
      owns_context_(std::exchange(other.owns_context_, false)),
      tensors_(std::move(other.tensors_)),
      bindings_(std::move(other.bindings_)) {}

Context& Context::operator=(Context&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (context_ != nullptr && owns_context_) {
        ggml_free(context_);
    }
    context_ = std::exchange(other.context_, nullptr);
    owns_context_ = std::exchange(other.owns_context_, false);
    tensors_ = std::move(other.tensors_);
    bindings_ = std::move(other.bindings_);
    return *this;
}

ggml_tensor* Context::create_tensor(const std::string& name, const Shape& requested_shape, ggml_type type) {
    if (context_ == nullptr) {
        throw std::logic_error("cannot use a moved-from nn::Context");
    }
    if (name.size() >= GGML_MAX_NAME) {
        throw std::invalid_argument("tensor name exceeds GGML_MAX_NAME");
    }

    const Shape shape = normalize_shape(requested_shape);
    ggml_tensor* result = ggml_new_tensor(context_, type, static_cast<int>(shape.size()), shape.data());
    if (result == nullptr) {
        throw std::runtime_error("failed to create ggml tensor");
    }
    if (!name.empty()) {
        ggml_set_name(result, name.c_str());
    }
    tensors_.insert(result);
    return result;
}

void Context::check_tensor(const ggml_tensor* tensor) const {
    if (tensor == nullptr) {
        throw std::invalid_argument("tensor cannot be null");
    }
    if (tensors_.find(const_cast<ggml_tensor*>(tensor)) == tensors_.end()) {
        throw std::invalid_argument("tensor was not created by this nn::Context");
    }
}

size_t Context::element_count(const ggml_tensor* tensor) {
    const int64_t count = ggml_nelements(tensor);
    if (count < 0 || static_cast<uint64_t>(count) > std::numeric_limits<size_t>::max()) {
        throw std::overflow_error("tensor element count overflow");
    }
    return static_cast<size_t>(count);
}

void Context::require_binding(ggml_tensor* target) {
    check_tensor(target);
    ggml_set_input(target);
    Binding& binding = bindings_[target];
    binding.required = true;
    binding.bytes = ggml_nbytes(target);
    binding.dirty = true;
}

void Context::bind_borrow(ggml_tensor* target, const void* data, size_t count, size_t element_size) {
    check_tensor(target);
    const size_t expected_count = element_count(target);
    const size_t actual_count = count == 0 ? expected_count : count;
    if (actual_count != expected_count) {
        throw std::invalid_argument("bound data count does not match tensor shape");
    }
    const size_t bytes = checked_bytes(actual_count, element_size);
    if (bytes != ggml_nbytes(target)) {
        throw std::invalid_argument("bound data byte size does not match tensor storage");
    }
    if (bytes != 0 && data == nullptr) {
        throw std::invalid_argument("bound data cannot be null for a non-empty tensor");
    }

    Binding& binding = bindings_[target];
    binding.kind = BindingKind::borrowed;
    binding.borrowed_data = data;
    binding.owned_data.clear();
    binding.bytes = bytes;
    binding.required = true;
    binding.dirty = true;
}

void Context::bind_copy(ggml_tensor* target, const void* data, size_t count, size_t element_size) {
    check_tensor(target);
    const size_t expected_count = element_count(target);
    const size_t actual_count = count == 0 ? expected_count : count;
    if (actual_count != expected_count) {
        throw std::invalid_argument("copied data count does not match tensor shape");
    }
    const size_t bytes = checked_bytes(actual_count, element_size);
    if (bytes != ggml_nbytes(target)) {
        throw std::invalid_argument("copied data byte size does not match tensor storage");
    }
    if (bytes != 0 && data == nullptr) {
        throw std::invalid_argument("copied data cannot be null for a non-empty tensor");
    }

    Binding& binding = bindings_[target];
    binding.kind = BindingKind::owned;
    binding.borrowed_data = nullptr;
    binding.owned_data.resize(bytes);
    if (bytes != 0) {
        std::memcpy(binding.owned_data.data(), data, bytes);
    }
    binding.bytes = bytes;
    binding.required = true;
    binding.dirty = true;
}

void Context::bind_zero(ggml_tensor* target) {
    check_tensor(target);
    Binding& binding = bindings_[target];
    binding.kind = BindingKind::owned;
    binding.borrowed_data = nullptr;
    binding.bytes = ggml_nbytes(target);
    binding.owned_data.assign(binding.bytes, uint8_t{0});
    binding.required = true;
    binding.dirty = true;
}

void Context::materialize() {
    for (auto& entry : bindings_) {
        ggml_tensor* target = entry.first;
        Binding& binding = entry.second;
        if (!binding.dirty) {
            continue;
        }
        if (target->buffer == nullptr && target->data == nullptr) {
            continue;
        }
        if (binding.required && binding.kind == BindingKind::unbound) {
            const char* name = ggml_get_name(target);
            throw std::logic_error(std::string("input tensor is not bound: ") + (name ? name : "<unnamed>"));
        }
        if (binding.bytes == 0) {
            binding.dirty = false;
            continue;
        }

        const void* source = binding.kind == BindingKind::borrowed
            ? binding.borrowed_data
            : binding.owned_data.data();
        if (source == nullptr) {
            throw std::logic_error("tensor binding has no source data");
        }
        if (target->buffer != nullptr) {
            ggml_backend_tensor_set(target, source, 0, binding.bytes);
        } else if (target->data != nullptr) {
            std::memcpy(target->data, source, binding.bytes);
        } else {
            throw std::logic_error("tensor must be allocated before materialize()");
        }
        binding.dirty = false;
    }
}

void Context::reset() {
    if (context_ == nullptr) throw std::logic_error("cannot reset a moved-from nn::Context");
    tensors_.clear();
    bindings_.clear();
    ggml_reset(context_);
}

void Context::write_bytes(
    ggml_tensor* target,
    const void* data,
    size_t count,
    size_t element_offset,
    size_t element_size) {
    if (target == nullptr) {
        throw std::invalid_argument("tensor cannot be null");
    }
    const size_t total = element_count(target);
    if (element_offset > total || count > total - element_offset) {
        throw std::out_of_range("tensor write exceeds tensor bounds");
    }
    if (count != 0 && data == nullptr) {
        throw std::invalid_argument("tensor write data cannot be null");
    }
    const size_t offset = checked_bytes(element_offset, element_size);
    const size_t bytes = checked_bytes(count, element_size);
    if (target->buffer != nullptr) {
        ggml_backend_tensor_set(target, data, offset, bytes);
    } else if (target->data != nullptr) {
        std::memcpy(static_cast<uint8_t*>(target->data) + offset, data, bytes);
    } else {
        throw std::logic_error("tensor must be allocated before write()");
    }
}

void Context::read_bytes(
    const ggml_tensor* source,
    void* data,
    size_t count,
    size_t element_offset,
    size_t element_size) const {
    if (source == nullptr) {
        throw std::invalid_argument("tensor cannot be null");
    }
    const size_t total = element_count(source);
    if (element_offset > total || count > total - element_offset) {
        throw std::out_of_range("tensor read exceeds tensor bounds");
    }
    if (count != 0 && data == nullptr) {
        throw std::invalid_argument("tensor read destination cannot be null");
    }
    const size_t offset = checked_bytes(element_offset, element_size);
    const size_t bytes = checked_bytes(count, element_size);
    if (source->buffer != nullptr) {
        ggml_backend_tensor_get(source, data, offset, bytes);
    } else if (source->data != nullptr) {
        std::memcpy(data, static_cast<const uint8_t*>(source->data) + offset, bytes);
    } else {
        throw std::logic_error("tensor must be allocated before read()");
    }
}

ggml_tensor* Context::view(
    ggml_tensor* source,
    const Shape& shape,
    size_t element_offset,
    const std::string& name) {
    if (source == nullptr) {
        throw std::invalid_argument("tensor cannot be null");
    }
    if (ggml_is_quantized(source->type)) {
        throw std::invalid_argument("element offsets are ambiguous for quantized tensors; use view_bytes()");
    }
    return view_bytes(source, shape, checked_bytes(element_offset, ggml_type_size(source->type)), {}, name);
}

ggml_tensor* Context::view_bytes(
    ggml_tensor* source,
    const Shape& requested_shape,
    size_t byte_offset,
    const Strides& byte_strides,
    const std::string& name) {
    if (source == nullptr) {
        throw std::invalid_argument("tensor cannot be null");
    }
    const Shape shape = normalize_shape(requested_shape);
    if (!byte_strides.empty() && byte_strides.size() != shape.size() - 1) {
        throw std::invalid_argument("view byte strides must contain one stride for each dimension after dimension 0");
    }

    Strides strides = byte_strides;
    if (strides.empty() && shape.size() > 1) {
        strides.resize(shape.size() - 1);
        strides[0] = ggml_row_size(source->type, shape[0]);
        for (size_t i = 1; i < strides.size(); ++i) {
            if (shape[i] != 0 && strides[i - 1] > std::numeric_limits<size_t>::max() / static_cast<size_t>(shape[i])) {
                throw std::overflow_error("view stride overflow");
            }
            strides[i] = strides[i - 1] * static_cast<size_t>(shape[i]);
        }
    }

    size_t span = ggml_row_size(source->type, shape[0]);
    for (size_t i = 1; i < shape.size(); ++i) {
        if (shape[i] > 0) {
            const size_t extent = static_cast<size_t>(shape[i] - 1);
            if (extent != 0 && strides[i - 1] > (std::numeric_limits<size_t>::max() - span) / extent) {
                throw std::overflow_error("view byte span overflow");
            }
            span += extent * strides[i - 1];
        }
    }
    if (byte_offset > ggml_nbytes(source) || span > ggml_nbytes(source) - byte_offset) {
        throw std::out_of_range("view exceeds source tensor storage");
    }

    ggml_tensor* result = nullptr;
    switch (shape.size()) {
        case 1:
            result = ggml_view_1d(context_, source, shape[0], byte_offset);
            break;
        case 2:
            result = ggml_view_2d(context_, source, shape[0], shape[1], strides[0], byte_offset);
            break;
        case 3:
            result = ggml_view_3d(context_, source, shape[0], shape[1], shape[2], strides[0], strides[1], byte_offset);
            break;
        case 4:
            result = ggml_view_4d(context_, source, shape[0], shape[1], shape[2], shape[3], strides[0], strides[1], strides[2], byte_offset);
            break;
        default:
            throw std::logic_error("normalized tensor shape must have between one and four dimensions");
    }
    if (!name.empty()) {
        if (name.size() >= GGML_MAX_NAME) {
            throw std::invalid_argument("tensor name exceeds GGML_MAX_NAME");
        }
        ggml_set_name(result, name.c_str());
    }
    tensors_.insert(result);
    return result;
}

ggml_cgraph* Context::create_graph(size_t capacity, bool gradients) {
    if (context_ == nullptr) {
        throw std::logic_error("cannot use a moved-from nn::Context");
    }
    if (capacity == 0) {
        throw std::invalid_argument("graph capacity must be greater than zero");
    }
    return ggml_new_graph_custom(context_, capacity, gradients);
}

ggml_cgraph* Context::build(ggml_tensor* output, size_t capacity) {
    if (output == nullptr) {
        throw std::invalid_argument("graph output cannot be null");
    }
    ggml_cgraph* graph = create_graph(capacity, false);
    ggml_set_output(output);
    expand(graph, output);
    return graph;
}

ggml_cgraph* Context::build(std::initializer_list<ggml_tensor*> outputs, size_t capacity) {
    if (outputs.size() == 0) {
        throw std::invalid_argument("cannot build a graph without outputs");
    }
    ggml_cgraph* graph = create_graph(capacity, false);
    for (ggml_tensor* output : outputs) {
        if (output == nullptr) {
            throw std::invalid_argument("graph output cannot be null");
        }
        ggml_set_output(output);
        expand(graph, output);
    }
    return graph;
}

void Context::expand(ggml_cgraph* graph, ggml_tensor* output) {
    if (graph == nullptr) {
        throw std::invalid_argument("graph cannot be null");
    }
    if (output == nullptr) {
        throw std::invalid_argument("graph output cannot be null");
    }
    ggml_build_forward_expand(graph, output);
}

} // namespace nn
