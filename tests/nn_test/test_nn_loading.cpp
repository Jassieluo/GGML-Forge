#include "nn/io/load.h"
#include "nn/io/gguf.h"
#include "nn/nn.h"

#include "ggml-backend.h"
#include "gguf.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

class MemorySource final : public nn::io::Source {
public:
    void add(nn::io::TensorInfo info, std::vector<uint8_t> bytes) {
        if (info.bytes != bytes.size()) throw std::invalid_argument("invalid test tensor bytes");
        indices_[info.name] = infos_.size();
        infos_.push_back(std::move(info));
        bytes_.push_back(std::move(bytes));
    }

    size_t size() const override { return infos_.size(); }
    const nn::io::TensorInfo& info(size_t index) const override { return infos_.at(index); }
    std::optional<size_t> find(std::string_view name) const override {
        auto it = indices_.find(std::string(name));
        return it == indices_.end() ? std::nullopt : std::optional<size_t>(it->second);
    }
    bool read(size_t index, size_t offset, void* destination, size_t bytes) override {
        if (index >= bytes_.size() || offset > bytes_[index].size() || bytes > bytes_[index].size() - offset) return false;
        if (bytes > 0) std::memcpy(destination, bytes_[index].data() + offset, bytes);
        return true;
    }

private:
    std::vector<nn::io::TensorInfo> infos_;
    std::vector<std::vector<uint8_t>> bytes_;
    std::unordered_map<std::string, size_t> indices_;
};

class Layer final : public nn::Module<Layer> {
public:
    nn::Parameter& weight = parameter("weight", nn::Parameter::required(nn::Shape{2, 3}));
    nn::Parameter& bias = parameter("bias", nn::Parameter::optional(nn::Shape{2}));
};

class Model final : public nn::Module<Model> {
public:
    Layer& layer = submodule<Layer>("layer");
    nn::Parameter& tied = parameter("tied", nn::Parameter::required(nn::Shape{2, 3}));

    Model() { tied.tie(layer.weight); }
};

class EmbeddingModel final : public nn::Module<EmbeddingModel> {
public:
    nn::Embedding& embedding = submodule<nn::Embedding>("embedding");
};

class CallableLeaf final : public nn::Module<CallableLeaf> {
public:
    int forward(int value) { return value + 1; }
};

class OwnedModel final : public nn::Module<OwnedModel> {
public:
    CallableLeaf& leaf = submodule<CallableLeaf>("leaf");
    nn::Parameter& scale = parameter("scale", nn::Parameter::required(nn::Shape{1}));
};

class CapabilityModel final : public nn::Module<CapabilityModel> {
public:
    nn::Linear& linear = submodule<nn::Linear>("linear");
    nn::Embedding& embedding = submodule<nn::Embedding>("embedding");
    nn::Conv1d& conv = submodule<nn::Conv1d>("conv");
    nn::LayerNorm& norm = submodule<nn::LayerNorm>("norm");
};

std::vector<uint8_t> bytes(const std::vector<float>& values) {
    std::vector<uint8_t> result(values.size() * sizeof(float));
    std::memcpy(result.data(), values.data(), result.size());
    return result;
}

std::vector<uint8_t> quantize_q4_0(const std::vector<float>& values, int64_t rows, int64_t row_elements) {
    std::vector<uint8_t> result(ggml_row_size(GGML_TYPE_Q4_0, row_elements) * static_cast<size_t>(rows));
    const size_t written = ggml_quantize_chunk(
        GGML_TYPE_Q4_0, values.data(), result.data(), 0, rows, row_elements, nullptr);
    if (written != result.size()) throw std::runtime_error("Q4_0 test quantization size mismatch");
    return result;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct TemporaryFile {
    std::filesystem::path path;

    explicit TemporaryFile(std::filesystem::path value) : path(std::move(value)) {}
    TemporaryFile(const TemporaryFile&) = delete;
    TemporaryFile& operator=(const TemporaryFile&) = delete;
    TemporaryFile(TemporaryFile&& other) noexcept : path(std::move(other.path)) { other.path.clear(); }
    TemporaryFile& operator=(TemporaryFile&&) = delete;

    ~TemporaryFile() {
        if (path.empty()) return;
        std::error_code error;
        std::filesystem::remove(path, error);
    }
};

TemporaryFile write_test_gguf() {
    const auto unique = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    TemporaryFile file(std::filesystem::temp_directory_path() /
                       ("nn-gguf-source-" + std::to_string(unique) + ".gguf"));

    ggml_init_params params = {/*.mem_size =*/ 1024 * 1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors(ggml_init(params), ggml_free);
    require(tensors != nullptr, "failed to create GGUF test tensor context");
    ggml_tensor* weight = ggml_new_tensor_2d(tensors.get(), GGML_TYPE_F32, 3, 2);
    ggml_set_name(weight, "layer.weight");
    const std::vector<float> values{1, 2, 3, 4, 5, 6};
    std::memcpy(weight->data, values.data(), values.size() * sizeof(float));

    std::unique_ptr<gguf_context, decltype(&gguf_free)> output(gguf_init_empty(), gguf_free);
    require(output != nullptr, "failed to create GGUF test metadata");
    gguf_set_val_str(output.get(), "general.architecture", "nn_test");
    const char* names[] = {"layer.weight"};
    const int32_t offsets[] = {0, 2};
    const int32_t axes[] = {1, 0};
    gguf_set_arr_str(output.get(), "nn.storage_layout.names", names, 1);
    gguf_set_arr_data(output.get(), "nn.storage_layout.offsets", GGUF_TYPE_INT32, offsets, 2);
    gguf_set_arr_data(output.get(), "nn.storage_layout.axes", GGUF_TYPE_INT32, axes, 2);
    gguf_add_tensor(output.get(), weight);
    require(gguf_write_to_file(output.get(), file.path.string().c_str(), false), "failed to write GGUF test file");
    return file;
}

} // namespace

int main() {
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    require(backend != nullptr, "failed to create CPU backend");

    try {
        OwnedModel owned;
        require(owned.leaf(41) == 42, "CRTP module call did not forward to forward()");
        require(owned.parameter_count() == 1, "owned parameter was not registered");

        CapabilityModel capability_model;
        const nn::ModelSchema schema = capability_model.schema();
        require(schema.to_json().find("\"path\":\"embedding.weight\"") != std::string::npos,
                "model schema JSON omitted a canonical parameter path");
        require(schema.to_json().find("\"Q4_0\"") != std::string::npos,
                "model schema JSON did not use canonical storage type names");
        const nn::ParameterSchema* linear_weight = schema.find("linear.weight");
        const nn::ParameterSchema* embedding_weight = schema.find("embedding.weight");
        const nn::ParameterSchema* conv_weight = schema.find("conv.weight");
        const nn::ParameterSchema* norm_weight = schema.find("norm.weight");
        require(linear_weight && linear_weight->supports_direct_storage(GGML_TYPE_Q4_K),
                "linear schema omitted direct Q4_K storage");
        require(embedding_weight && embedding_weight->supports_direct_storage(GGML_TYPE_Q4_0) &&
                    !embedding_weight->supports_direct_storage(GGML_TYPE_Q4_K),
                "embedding schema does not reflect portable direct storage support");
        require(conv_weight && conv_weight->quantized_layout ==
                    ggml_ops_ext::ops_weight_layout::channel_rows,
                "convolution schema omitted channel-row quantized layout");
        require(norm_weight && norm_weight->direct_storage_types == std::vector<ggml_type>({GGML_TYPE_F32}),
                "normalization schema exposed unsupported storage types");

        Model model;
        MemorySource source;
        source.add({"layer.weight", GGML_TYPE_F32, {3, 2}, {}, nn::Layout::permuted({1, 0}), 6 * sizeof(float)},
                   bytes({1, 2, 3, 4, 5, 6}));

        nn::io::LoadResult result = nn::io::load_into(model, source, backend);
        require(static_cast<bool>(result), result.error.c_str());
        require(model.parameter_count() == 3, "recursive parameter count mismatch");
        require(model.layer.weight.is_bound(), "required parameter was not bound");
        require(!model.layer.bias.is_bound(), "missing optional parameter was unexpectedly bound");
        require(model.tied.local_tensor() == model.layer.weight.local_tensor(), "tied parameter did not share storage");
        require(result.state->size() == 1, "state dict should contain one physical tensor");
        require(model.layer.weight.logical_shape() == nn::Shape({2, 3}), "logical layout was not applied");

        Model strict_model;
        MemorySource source_with_extra;
        source_with_extra.add(
            {"layer.weight", GGML_TYPE_F32, {3, 2}, {}, nn::Layout::permuted({1, 0}), 6 * sizeof(float)},
            bytes({1, 2, 3, 4, 5, 6}));
        source_with_extra.add(
            {"unused.weight", GGML_TYPE_F32, {1}, {}, nn::Layout::identity(), sizeof(float)},
            bytes({1}));
        nn::io::LoadResult strict_result = nn::io::load_into(strict_model, source_with_extra, backend);
        require(!strict_result && strict_result.error.find("unexpected parameters") != std::string::npos,
                "strict loading accepted an unexpected source tensor");

        TemporaryFile file = write_test_gguf();
        nn::io::GGUFSource gguf_source(file.path.string());
        require(gguf_source.size() == 1, "GGUF source tensor count mismatch");
        require(gguf_source.info(0).storage_shape == nn::Shape({3, 2}), "GGUF storage shape mismatch");
        require(gguf_source.info(0).logical_shape == nn::Shape({2, 3}), "GGUF logical shape mismatch");
        const int64_t architecture = gguf_find_key(gguf_source.metadata_context(), "general.architecture");
        require(architecture >= 0, "GGUF metadata context is unavailable");

        Model gguf_model;
        nn::io::LoadResult gguf_result = nn::io::load_into(gguf_model, gguf_source, backend);
        require(static_cast<bool>(gguf_result), gguf_result.error.c_str());
        std::vector<float> loaded(6);
        ggml_backend_tensor_get(gguf_model.layer.weight.local_tensor(), loaded.data(), 0, loaded.size() * sizeof(float));
        require(loaded == std::vector<float>({1, 2, 3, 4, 5, 6}), "GGUF tensor contents mismatch");

        std::vector<float> embedding_values(64);
        for (size_t i = 0; i < embedding_values.size(); ++i) {
            embedding_values[i] = std::sin(static_cast<float>(i) * 0.1f);
        }
        std::vector<uint8_t> quantized = quantize_q4_0(embedding_values, 2, 32);
        const size_t quantized_bytes = quantized.size();
        MemorySource quantized_source;
        quantized_source.add({"embedding.weight", GGML_TYPE_Q4_0, {32, 2}, {32, 2},
                              nn::Layout::identity(), quantized_bytes}, std::move(quantized));
        EmbeddingModel embedding_model;
        nn::io::LoadResult embedding_result = nn::io::load_into(embedding_model, quantized_source, backend);
        require(static_cast<bool>(embedding_result), embedding_result.error.c_str());
        require(embedding_model.embedding.weight.storage_type() == GGML_TYPE_Q4_0,
                "quantized embedding was not preserved for direct execution");

    } catch (const std::exception& error) {
        std::cerr << "nn loading test failed: " << error.what() << "\n";
        ggml_backend_free(backend);
        return 1;
    }

    ggml_backend_free(backend);
    std::cout << "nn loading tests passed\n";
    return 0;
}
