#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "nn/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nn {

namespace data {

template <typename T>
struct Borrowed {
    const T* data = nullptr;
    size_t count = 0;
};

template <typename T>
struct Copied {
    const T* data = nullptr;
    size_t count = 0;
};

template <typename T>
Borrowed<T> borrow(const T* data, size_t count = 0) {
    return {data, count};
}

template <typename T, size_t N>
Borrowed<T> borrow(const T (&data)[N]) {
    return {data, N};
}

template <typename T, size_t N>
Borrowed<T> borrow(const std::array<T, N>& data) {
    return {data.data(), N};
}

template <typename T, typename Allocator>
Borrowed<T> borrow(const std::vector<T, Allocator>& data) {
    return {data.data(), data.size()};
}

template <typename T, typename Allocator>
Borrowed<T> borrow(std::vector<T, Allocator>&&) = delete;

template <typename T>
Copied<T> copy(const T* data, size_t count = 0) {
    return {data, count};
}

template <typename T, size_t N>
Copied<T> copy(const T (&data)[N]) {
    return {data, N};
}

template <typename T, size_t N>
Copied<T> copy(const std::array<T, N>& data) {
    return {data.data(), N};
}

template <typename T, typename Allocator>
Copied<T> copy(const std::vector<T, Allocator>& data) {
    return {data.data(), data.size()};
}

} // namespace data

class Context {
public:
    explicit Context(size_t memory_size, bool no_alloc = true);
    Context(void* memory_buffer, size_t memory_size, bool no_alloc = true);
    static Context borrow(ggml_context* context);
    ~Context();

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&& other) noexcept;
    Context& operator=(Context&& other) noexcept;

    ggml_context* native_handle() noexcept { return context_; }
    const ggml_context* native_handle() const noexcept { return context_; }

    ggml_tensor* empty(const std::string& name, const Shape& shape, ggml_type type) {
        return create_tensor(name, shape, type);
    }

    template <typename T>
    ggml_tensor* empty(const std::string& name, const Shape& shape) {
        return create_tensor(name, shape, type_of<T>());
    }

    template <typename T>
    ggml_tensor* empty(const std::string& name, std::initializer_list<int64_t> shape) {
        return empty<T>(name, Shape(shape));
    }

    template <typename T>
    ggml_tensor* input(const std::string& name, const Shape& shape) {
        ggml_tensor* result = create_tensor(name, shape, type_of<T>());
        require_binding(result);
        return result;
    }

    template <typename T>
    ggml_tensor* input(const std::string& name, std::initializer_list<int64_t> shape) {
        return input<T>(name, Shape(shape));
    }

    template <typename T>
    ggml_tensor* input(const std::string& name, const Shape& shape, data::Borrowed<T> source) {
        ggml_tensor* result = input<T>(name, shape);
        bind(result, source);
        return result;
    }

    template <typename T>
    ggml_tensor* input(const std::string& name, std::initializer_list<int64_t> shape, data::Borrowed<T> source) {
        return input<T>(name, Shape(shape), source);
    }

    template <typename T>
    ggml_tensor* constant(const std::string& name, const Shape& shape, data::Copied<T> source) {
        ggml_tensor* result = create_tensor(name, shape, type_of<T>());
        bind(result, source);
        return result;
    }

    template <typename T>
    ggml_tensor* constant(const std::string& name, std::initializer_list<int64_t> shape, data::Copied<T> source) {
        return constant<T>(name, Shape(shape), source);
    }

    template <typename T>
    ggml_tensor* zeros(const std::string& name, const Shape& shape) {
        ggml_tensor* result = create_tensor(name, shape, type_of<T>());
        bind_zero(result);
        return result;
    }

    template <typename T>
    ggml_tensor* zeros(const std::string& name, std::initializer_list<int64_t> shape) {
        return zeros<T>(name, Shape(shape));
    }

    template <typename T>
    ggml_tensor* full(const std::string& name, const Shape& shape, T value) {
        ggml_tensor* result = create_tensor(name, shape, type_of<T>());
        const size_t count = element_count(result);
        std::vector<T> values(count, value);
        bind_copy(result, values.data(), values.size(), sizeof(T));
        return result;
    }

    template <typename T>
    ggml_tensor* full(const std::string& name, std::initializer_list<int64_t> shape, T value) {
        return full<T>(name, Shape(shape), value);
    }

    template <typename T>
    ggml_tensor* tensor(const std::string& name, const Shape& shape, data::Borrowed<T> source) {
        return input<T>(name, shape, source);
    }

    template <typename T>
    ggml_tensor* tensor(const std::string& name, const Shape& shape, data::Copied<T> source) {
        return constant<T>(name, shape, source);
    }

    template <typename T>
    void bind(ggml_tensor* target, data::Borrowed<T> source) {
        check_type<T>(target);
        bind_borrow(target, source.data, source.count, sizeof(T));
    }

    template <typename T>
    void bind(ggml_tensor* target, data::Copied<T> source) {
        check_type<T>(target);
        bind_copy(target, source.data, source.count, sizeof(T));
    }

    void materialize();
    void reset();

    template <typename T>
    void write(ggml_tensor* target, const T* data, size_t count, size_t element_offset = 0) {
        check_type<T>(target);
        write_bytes(target, data, count, element_offset, sizeof(T));
    }

    template <typename T>
    void read(const ggml_tensor* source, T* data, size_t count, size_t element_offset = 0) const {
        check_type<T>(source);
        read_bytes(source, data, count, element_offset, sizeof(T));
    }

    ggml_tensor* view(
        ggml_tensor* source,
        const Shape& shape,
        size_t element_offset = 0,
        const std::string& name = "");

    ggml_tensor* view_bytes(
        ggml_tensor* source,
        const Shape& shape,
        size_t byte_offset = 0,
        const Strides& byte_strides = {},
        const std::string& name = "");

    ggml_cgraph* create_graph(size_t capacity = GGML_DEFAULT_GRAPH_SIZE, bool gradients = false);
    ggml_cgraph* build(ggml_tensor* output, size_t capacity = GGML_DEFAULT_GRAPH_SIZE);
    ggml_cgraph* build(std::initializer_list<ggml_tensor*> outputs, size_t capacity = GGML_DEFAULT_GRAPH_SIZE);
    void expand(ggml_cgraph* graph, ggml_tensor* output);

private:
    Context(ggml_context* context, bool owns_context);

    enum class BindingKind {
        unbound,
        borrowed,
        owned,
    };

    struct Binding {
        BindingKind kind = BindingKind::unbound;
        const void* borrowed_data = nullptr;
        std::vector<uint8_t> owned_data;
        size_t bytes = 0;
        bool required = false;
        bool dirty = true;
    };

    ggml_context* context_ = nullptr;
    bool owns_context_ = true;
    std::unordered_set<ggml_tensor*> tensors_;
    std::unordered_map<ggml_tensor*, Binding> bindings_;

    ggml_tensor* create_tensor(const std::string& name, const Shape& shape, ggml_type type);
    void require_binding(ggml_tensor* target);
    void bind_borrow(ggml_tensor* target, const void* data, size_t count, size_t element_size);
    void bind_copy(ggml_tensor* target, const void* data, size_t count, size_t element_size);
    void bind_zero(ggml_tensor* target);
    void write_bytes(ggml_tensor* target, const void* data, size_t count, size_t element_offset, size_t element_size);
    void read_bytes(const ggml_tensor* source, void* data, size_t count, size_t element_offset, size_t element_size) const;
    void check_tensor(const ggml_tensor* tensor) const;
    static size_t element_count(const ggml_tensor* tensor);

    template <typename T>
    static constexpr ggml_type type_of() {
        if constexpr (std::is_same_v<T, float>) {
            return GGML_TYPE_F32;
        } else if constexpr (std::is_same_v<T, double>) {
            return GGML_TYPE_F64;
        } else if constexpr (std::is_same_v<T, ggml_fp16_t>) {
            return GGML_TYPE_F16;
        } else if constexpr (std::is_same_v<T, int8_t>) {
            return GGML_TYPE_I8;
        } else if constexpr (std::is_same_v<T, int16_t>) {
            return GGML_TYPE_I16;
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return GGML_TYPE_I32;
        } else if constexpr (std::is_same_v<T, int64_t>) {
            return GGML_TYPE_I64;
        } else {
            static_assert(!std::is_same_v<T, T>, "unsupported nn::Context tensor element type");
        }
    }

    template <typename T>
    void check_type(const ggml_tensor* tensor) const {
        if (tensor == nullptr) {
            throw std::invalid_argument("tensor cannot be null");
        }
        if (tensor->type != type_of<T>()) {
            throw std::invalid_argument("tensor element type does not match the requested C++ type");
        }
    }
};

} // namespace nn
