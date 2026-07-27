#include "nn/io/gguf.h"

#include "gguf.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nn::io {

namespace {

using GgufPtr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
using GgmlPtr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;

bool read_integer_array(const gguf_context* context, int64_t key, std::vector<int64_t>& values) {
    if (key < 0 || gguf_get_kv_type(context, key) != GGUF_TYPE_ARRAY) return false;
    const size_t count = gguf_get_arr_n(context, key);
    const void* data = gguf_get_arr_data(context, key);
    if (count > 0 && !data) return false;
    values.resize(count);

#define COPY_INTEGER_ARRAY(type_id, cpp_type) \
    case type_id: { \
        const auto* source = static_cast<const cpp_type*>(data); \
        for (size_t i = 0; i < count; ++i) values[i] = static_cast<int64_t>(source[i]); \
        return true; \
    }

    switch (gguf_get_arr_type(context, key)) {
        COPY_INTEGER_ARRAY(GGUF_TYPE_UINT8, uint8_t)
        COPY_INTEGER_ARRAY(GGUF_TYPE_INT8, int8_t)
        COPY_INTEGER_ARRAY(GGUF_TYPE_UINT16, uint16_t)
        COPY_INTEGER_ARRAY(GGUF_TYPE_INT16, int16_t)
        COPY_INTEGER_ARRAY(GGUF_TYPE_UINT32, uint32_t)
        COPY_INTEGER_ARRAY(GGUF_TYPE_INT32, int32_t)
        COPY_INTEGER_ARRAY(GGUF_TYPE_UINT64, uint64_t)
        COPY_INTEGER_ARRAY(GGUF_TYPE_INT64, int64_t)
        default: return false;
    }

#undef COPY_INTEGER_ARRAY
}

std::unordered_map<std::string, Layout> read_layouts(const gguf_context* context) {
    const int64_t names_key = gguf_find_key(context, "nn.storage_layout.names");
    const int64_t offsets_key = gguf_find_key(context, "nn.storage_layout.offsets");
    const int64_t axes_key = gguf_find_key(context, "nn.storage_layout.axes");
    if (names_key < 0 && offsets_key < 0 && axes_key < 0) return {};
    if (names_key < 0 || offsets_key < 0 || axes_key < 0 ||
        gguf_get_kv_type(context, names_key) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(context, names_key) != GGUF_TYPE_STRING) {
        throw std::runtime_error("incomplete nn.storage_layout metadata");
    }

    std::vector<int64_t> offsets;
    std::vector<int64_t> axes;
    if (!read_integer_array(context, offsets_key, offsets) ||
        !read_integer_array(context, axes_key, axes)) {
        throw std::runtime_error("nn.storage_layout offsets and axes must be integer arrays");
    }

    const size_t count = gguf_get_arr_n(context, names_key);
    if (offsets.size() != count + 1 || offsets.empty() || offsets.front() != 0 ||
        offsets.back() < 0 || static_cast<size_t>(offsets.back()) != axes.size()) {
        throw std::runtime_error("invalid nn.storage_layout offsets");
    }

    std::unordered_map<std::string, Layout> layouts;
    for (size_t i = 0; i < count; ++i) {
        const char* name = gguf_get_arr_str(context, names_key, i);
        const int64_t begin = offsets[i];
        const int64_t end = offsets[i + 1];
        if (!name || !name[0] || begin < 0 || end <= begin ||
            end > static_cast<int64_t>(axes.size()) || end - begin > GGML_MAX_DIMS) {
            throw std::runtime_error("invalid nn.storage_layout entry");
        }
        std::vector<int> permutation;
        permutation.reserve(static_cast<size_t>(end - begin));
        for (int64_t j = begin; j < end; ++j) {
            if (axes[static_cast<size_t>(j)] < 0 || axes[static_cast<size_t>(j)] > std::numeric_limits<int>::max()) {
                throw std::runtime_error("invalid nn.storage_layout axis");
            }
            permutation.push_back(static_cast<int>(axes[static_cast<size_t>(j)]));
        }
        if (!layouts.emplace(name, Layout::permuted(permutation)).second) {
            throw std::runtime_error("duplicate nn.storage_layout tensor name");
        }
    }
    return layouts;
}

std::unordered_map<std::string, Shape> read_logical_shapes(const gguf_context* context) {
    const int64_t names_key = gguf_find_key(context, "nn.logical_shape.names");
    const int64_t offsets_key = gguf_find_key(context, "nn.logical_shape.offsets");
    const int64_t dimensions_key = gguf_find_key(context, "nn.logical_shape.dimensions");
    if (names_key < 0 && offsets_key < 0 && dimensions_key < 0) return {};
    if (names_key < 0 || offsets_key < 0 || dimensions_key < 0 ||
        gguf_get_kv_type(context, names_key) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(context, names_key) != GGUF_TYPE_STRING) {
        throw std::runtime_error("incomplete nn.logical_shape metadata");
    }
    std::vector<int64_t> offsets;
    std::vector<int64_t> dimensions;
    if (!read_integer_array(context, offsets_key, offsets) ||
        !read_integer_array(context, dimensions_key, dimensions)) {
        throw std::runtime_error("nn.logical_shape offsets and dimensions must be integer arrays");
    }
    const size_t count = gguf_get_arr_n(context, names_key);
    if (offsets.size() != count + 1 || offsets.empty() || offsets.front() != 0 ||
        offsets.back() < 0 || static_cast<size_t>(offsets.back()) != dimensions.size()) {
        throw std::runtime_error("invalid nn.logical_shape offsets");
    }
    std::unordered_map<std::string, Shape> shapes;
    for (size_t i = 0; i < count; ++i) {
        const char* name = gguf_get_arr_str(context, names_key, i);
        const int64_t begin = offsets[i];
        const int64_t end = offsets[i + 1];
        if (!name || !name[0] || begin < 0 || end <= begin ||
            end > static_cast<int64_t>(dimensions.size()) || end - begin > GGML_MAX_DIMS) {
            throw std::runtime_error("invalid nn.logical_shape entry");
        }
        Shape shape;
        shape.reserve(static_cast<size_t>(end - begin));
        for (int64_t j = begin; j < end; ++j) {
            const int64_t dimension = dimensions[static_cast<size_t>(j)];
            if (dimension <= 0) throw std::runtime_error("invalid nn.logical_shape dimension");
            shape.push_back(dimension);
        }
        if (!shapes.emplace(name, std::move(shape)).second) {
            throw std::runtime_error("duplicate nn.logical_shape tensor name");
        }
    }
    return shapes;
}

} // namespace

struct GGUFSource::Impl {
    explicit Impl(const std::string& path) : file(path, std::ios::binary) {}

    GgufPtr gguf{nullptr, gguf_free};
    GgmlPtr tensors{nullptr, ggml_free};
    std::ifstream file;
    size_t data_offset = 0;
    std::vector<TensorInfo> infos;
    std::vector<size_t> offsets;
    std::unordered_map<std::string, size_t> indices;
};

GGUFSource::GGUFSource(const std::string& path) : impl_(std::make_unique<Impl>(path)) {
    if (!impl_->file) throw std::runtime_error("failed to open GGUF file: " + path);

    ggml_context* tensor_context = nullptr;
    gguf_init_params params = {/*.no_alloc =*/ true, /*.ctx =*/ &tensor_context};
    impl_->gguf.reset(gguf_init_from_file(path.c_str(), params));
    impl_->tensors.reset(tensor_context);
    if (!impl_->gguf || !impl_->tensors) throw std::runtime_error("failed to read GGUF metadata: " + path);

    impl_->data_offset = gguf_get_data_offset(impl_->gguf.get());
    const auto layouts = read_layouts(impl_->gguf.get());
    const auto logical_shapes = read_logical_shapes(impl_->gguf.get());
    const int64_t tensor_count = gguf_get_n_tensors(impl_->gguf.get());
    impl_->infos.reserve(static_cast<size_t>(tensor_count));
    impl_->offsets.reserve(static_cast<size_t>(tensor_count));
    impl_->indices.reserve(static_cast<size_t>(tensor_count));

    for (int64_t i = 0; i < tensor_count; ++i) {
        const char* name = gguf_get_tensor_name(impl_->gguf.get(), i);
        ggml_tensor* tensor = name ? ggml_get_tensor(impl_->tensors.get(), name) : nullptr;
        if (!name || !name[0] || !tensor) throw std::runtime_error("invalid GGUF tensor metadata");

        TensorInfo info;
        info.name = name;
        info.storage_type = gguf_get_tensor_type(impl_->gguf.get(), i);
        const auto logical_shape = logical_shapes.find(name);
        // ggml_n_dims() drops trailing singleton dimensions. Logical-shape
        // metadata distinguishes flattened quantized matrices from native
        // floating-point convolution tensors when the output count is one.
        const int rank = logical_shape != logical_shapes.end() &&
                         logical_shape->second.size() == 4
            ? (ggml_is_quantized(info.storage_type)
                   ? std::max(2, ggml_n_dims(tensor)) : 4)
            : ggml_n_dims(tensor);
        info.storage_shape.reserve(static_cast<size_t>(rank));
        for (int dim = 0; dim < rank; ++dim) info.storage_shape.push_back(tensor->ne[dim]);
        info.bytes = gguf_get_tensor_size(impl_->gguf.get(), i);

        auto layout = layouts.find(info.name);
        if (logical_shape != logical_shapes.end()) {
            if (layout != layouts.end()) {
                throw std::runtime_error("tensor combines logical-shape and storage-layout metadata: " + info.name);
            }
            info.logical_shape = logical_shape->second;
        } else if (layout != layouts.end()) {
            info.layout = layout->second;
            info.logical_shape = info.layout.logical_shape(info.storage_shape);
        } else {
            info.logical_shape = info.storage_shape;
        }

        const size_t relative_offset = gguf_get_tensor_offset(impl_->gguf.get(), i);
        if (relative_offset > std::numeric_limits<size_t>::max() - impl_->data_offset) {
            throw std::runtime_error("GGUF tensor offset overflow: " + info.name);
        }
        const size_t index = impl_->infos.size();
        if (!impl_->indices.emplace(info.name, index).second) {
            throw std::runtime_error("duplicate GGUF tensor name: " + info.name);
        }
        impl_->infos.push_back(std::move(info));
        impl_->offsets.push_back(impl_->data_offset + relative_offset);
    }

    for (const auto& [name, layout] : layouts) {
        (void)layout;
        if (impl_->indices.find(name) == impl_->indices.end()) {
            throw std::runtime_error("nn.storage_layout refers to an unknown tensor: " + name);
        }
    }
    for (const auto& [name, shape] : logical_shapes) {
        (void)shape;
        if (impl_->indices.find(name) == impl_->indices.end()) {
            throw std::runtime_error("nn.logical_shape refers to an unknown tensor: " + name);
        }
    }
}

GGUFSource::~GGUFSource() = default;
GGUFSource::GGUFSource(GGUFSource&&) noexcept = default;
GGUFSource& GGUFSource::operator=(GGUFSource&&) noexcept = default;

size_t GGUFSource::size() const {
    return impl_->infos.size();
}

const TensorInfo& GGUFSource::info(size_t index) const {
    return impl_->infos.at(index);
}

std::optional<size_t> GGUFSource::find(std::string_view name) const {
    const auto found = impl_->indices.find(std::string(name));
    if (found == impl_->indices.end()) return std::nullopt;
    return found->second;
}

bool GGUFSource::read(size_t index, size_t offset, void* destination, size_t bytes) {
    if (index >= impl_->infos.size() || offset > impl_->infos[index].bytes ||
        bytes > impl_->infos[index].bytes - offset || (!destination && bytes > 0)) return false;
    if (offset > std::numeric_limits<size_t>::max() - impl_->offsets[index]) return false;
    const size_t absolute_offset = impl_->offsets[index] + offset;
    if (absolute_offset > static_cast<size_t>(std::numeric_limits<std::streamoff>::max()) ||
        bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) return false;
    impl_->file.clear();
    impl_->file.seekg(static_cast<std::streamoff>(absolute_offset), std::ios::beg);
    if (!impl_->file) return false;
    impl_->file.read(static_cast<char*>(destination), static_cast<std::streamsize>(bytes));
    return impl_->file.good() || static_cast<size_t>(impl_->file.gcount()) == bytes;
}

const gguf_context* GGUFSource::metadata_context() const noexcept {
    return impl_->gguf.get();
}

} // namespace nn::io
