#include "models/ssl/hubert/hubert.h"

#include "ggml-backend.h"
#include "gguf.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct TemporaryFile {
    std::filesystem::path path;
    explicit TemporaryFile(std::filesystem::path value) : path(std::move(value)) {}
    TemporaryFile(const TemporaryFile&) = delete;
    TemporaryFile& operator=(const TemporaryFile&) = delete;
    TemporaryFile(TemporaryFile&& other) noexcept : path(std::move(other.path)) { other.path.clear(); }
    ~TemporaryFile() {
        if (path.empty()) return;
        std::error_code error;
        std::filesystem::remove(path, error);
    }
};

void add_q4(ggml_context* context, gguf_context* output, const std::string& name, bool convolution) {
    constexpr int64_t width = 32;
    ggml_tensor* tensor = convolution
        ? ggml_new_tensor_3d(context, GGML_TYPE_Q4_0, width, 1, 2)
        : ggml_new_tensor_2d(context, GGML_TYPE_Q4_0, width, 1);
    ggml_set_name(tensor, name.c_str());
    std::vector<float> values(static_cast<size_t>(ggml_nelements(tensor)));
    for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin(static_cast<float>(i) * 0.1f);
    const int64_t rows = ggml_nelements(tensor) / width;
    require(ggml_quantize_chunk(GGML_TYPE_Q4_0, values.data(), tensor->data, 0, rows, width, nullptr) ==
                ggml_nbytes(tensor),
            "HuBERT fixture quantization size mismatch");
    gguf_add_tensor(output, tensor);
}

void add_f32(ggml_context* context, gguf_context* output, const std::string& name) {
    ggml_tensor* tensor = ggml_new_tensor_1d(context, GGML_TYPE_F32, 32);
    ggml_set_name(tensor, name.c_str());
    ggml_set_zero(tensor);
    gguf_add_tensor(output, tensor);
}

TemporaryFile write_hubert_fixture() {
    const auto unique = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    TemporaryFile file(std::filesystem::temp_directory_path() /
                       ("nn-hubert-load-" + std::to_string(unique) + ".gguf"));
    std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors(
        ggml_init({8 * 1024 * 1024, nullptr, false}), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> output(gguf_init_empty(), gguf_free);
    require(tensors != nullptr && output != nullptr, "failed to create HuBERT fixture contexts");

    gguf_set_val_str(output.get(), "general.architecture", "gpt_sovits_hubert");
    gguf_set_val_str(output.get(), "gpt_sovits.version", "v2");
    std::vector<std::string> channel_row_names;
    gpt_sovits::HubertModel model;
    model.for_each_parameter([&](std::string_view path, const nn::Parameter& parameter) {
        if (!parameter.is_required()) return;
        const std::string source_name(path);
        const auto usage = parameter.spec().usage;
        const bool convolution = usage == nn::Parameter::Usage::conv1d_weight ||
                                 usage == nn::Parameter::Usage::conv_transpose1d_weight;
        if (parameter.supports_direct_storage(GGML_TYPE_Q4_0)) {
            add_q4(tensors.get(), output.get(), source_name, convolution);
            if (convolution) channel_row_names.push_back(source_name);
        } else {
            add_f32(tensors.get(), output.get(), source_name);
        }
    });

    if (!channel_row_names.empty()) {
        std::vector<const char*> layout_names;
        std::vector<int32_t> offsets{0};
        std::vector<int32_t> axes;
        for (const std::string& name : channel_row_names) {
            layout_names.push_back(name.c_str());
            axes.insert(axes.end(), {1, 0, 2});
            offsets.push_back(static_cast<int32_t>(axes.size()));
        }
        gguf_set_arr_str(output.get(), "nn.storage_layout.names", layout_names.data(), layout_names.size());
        gguf_set_arr_data(output.get(), "nn.storage_layout.offsets", GGUF_TYPE_INT32,
                          offsets.data(), offsets.size());
        gguf_set_arr_data(output.get(), "nn.storage_layout.axes", GGUF_TYPE_INT32,
                          axes.data(), axes.size());
    }

    require(gguf_write_to_file(output.get(), file.path.string().c_str(), false),
            "failed to write HuBERT fixture");
    return file;
}

} // namespace

int main(int argc, char** argv) {
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    require(backend != nullptr, "failed to initialize CPU backend");
    try {
        if (argc == 1) {
            TemporaryFile file = write_hubert_fixture();
            gpt_sovits::HubertModel model;
            require(model.load(file.path.string(), backend), "HuBERT failed to load through nn::io");
            require(model.state_dict() != nullptr, "HuBERT did not retain StateDict ownership");
            require(model.position_encoder.weight.storage_type() == GGML_TYPE_Q4_0,
                    "HuBERT positional convolution did not preserve Q4 storage");
            require(model.feature_extractor.layers[0].weight.storage_type() == GGML_TYPE_Q4_0,
                    "HuBERT feature convolution did not preserve Q4 storage");
            require(model.encoder.layers[0].self_attn.q_proj.weight.storage_type() == GGML_TYPE_Q4_0,
                    "HuBERT linear weight did not preserve Q4 storage");
        } else {
            for (int index = 1; index < argc; ++index) {
                gpt_sovits::HubertModel model;
                require(model.load(argv[index], backend), "real HuBERT artifact failed to load through nn::io");
                require(model.state_dict() != nullptr, "real HuBERT artifact did not retain StateDict ownership");
                require(model.feature_extractor.layers[0].weight.is_bound() && !model.encoder.layers.empty(),
                        "real HuBERT topology is invalid");
                std::cout << "HuBERT artifact passed: " << argv[index] << '\n';
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "HuBERT loading test failed: " << error.what() << "\n";
        ggml_backend_free(backend);
        return 1;
    }
    ggml_backend_free(backend);
    std::cout << "HuBERT loading test passed\n";
    return 0;
}
