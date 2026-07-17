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
#include <unordered_map>
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

std::unordered_map<std::string, std::string> hubert_names() {
    std::unordered_map<std::string, std::string> names = {
        {"pos_conv_weight", "encoder.pos_conv_embed.conv.weight"},
        {"pos_conv_bias", "encoder.pos_conv_embed.conv.bias"},
        {"ln0.weight", "feature_extractor.conv_layers.0.layer_norm.weight"},
        {"ln0.bias", "feature_extractor.conv_layers.0.layer_norm.bias"},
        {"proj_ln.weight", "feature_projection.layer_norm.weight"},
        {"proj_ln.bias", "feature_projection.layer_norm.bias"},
        {"proj_dense.weight", "feature_projection.projection.weight"},
        {"proj_dense.bias", "feature_projection.projection.bias"},
        {"encoder_ln.weight", "encoder.layer_norm.weight"},
        {"encoder_ln.bias", "encoder.layer_norm.bias"},
    };
    for (int i = 0; i < 7; ++i) {
        names["conv_layers." + std::to_string(i) + ".weight"] =
            "feature_extractor.conv_layers." + std::to_string(i) + ".conv.weight";
        names["conv_layers." + std::to_string(i) + ".bias"] = "";
    }
    for (int i = 0; i < 12; ++i) {
        const std::string cpp = "encoder.layers." + std::to_string(i) + ".";
        const std::string source = "encoder.layers." + std::to_string(i) + ".";
        names[cpp + "self_attn.q_proj.weight"] = source + "attention.q_proj.weight";
        names[cpp + "self_attn.q_proj.bias"] = source + "attention.q_proj.bias";
        names[cpp + "self_attn.k_proj.weight"] = source + "attention.k_proj.weight";
        names[cpp + "self_attn.k_proj.bias"] = source + "attention.k_proj.bias";
        names[cpp + "self_attn.v_proj.weight"] = source + "attention.v_proj.weight";
        names[cpp + "self_attn.v_proj.bias"] = source + "attention.v_proj.bias";
        names[cpp + "self_attn.out_proj.weight"] = source + "attention.out_proj.weight";
        names[cpp + "self_attn.out_proj.bias"] = source + "attention.out_proj.bias";
        names[cpp + "norm1.weight"] = source + "layer_norm.weight";
        names[cpp + "norm1.bias"] = source + "layer_norm.bias";
        names[cpp + "ffn.w1.weight"] = source + "feed_forward.intermediate_dense.weight";
        names[cpp + "ffn.w1.bias"] = source + "feed_forward.intermediate_dense.bias";
        names[cpp + "ffn.w2.weight"] = source + "feed_forward.output_dense.weight";
        names[cpp + "ffn.w2.bias"] = source + "feed_forward.output_dense.bias";
        names[cpp + "norm2.weight"] = source + "final_layer_norm.weight";
        names[cpp + "norm2.bias"] = source + "final_layer_norm.bias";
    }
    return names;
}

void add_q4(ggml_context* context, gguf_context* output, const std::string& name) {
    constexpr int64_t width = 32;
    ggml_tensor* tensor = ggml_new_tensor_2d(context, GGML_TYPE_Q4_0, width, 1);
    ggml_set_name(tensor, name.c_str());
    std::vector<float> values(width);
    for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin(static_cast<float>(i) * 0.1f);
    require(ggml_quantize_chunk(GGML_TYPE_Q4_0, values.data(), tensor->data, 0, 1, width, nullptr) ==
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
    const auto names = hubert_names();
    gpt_sovits::HubertModel model;
    model.for_each_parameter([&](std::string_view path, const nn::Parameter& parameter) {
        if (!parameter.is_required()) return;
        const auto found = names.find(std::string(path));
        const std::string source_name = found == names.end() ? std::string(path) : found->second;
        require(!source_name.empty(), "required HuBERT fixture parameter has no source name");
        if (path == "pos_conv_bias") add_f32(tensors.get(), output.get(), source_name);
        else add_q4(tensors.get(), output.get(), source_name);
    });

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
            require(model.pos_conv_weight.storage_type() == GGML_TYPE_Q4_0,
                    "HuBERT positional convolution did not preserve Q4 storage");
            require(model.conv_layers[0].weight.storage_type() == GGML_TYPE_Q4_0,
                    "HuBERT feature convolution did not preserve Q4 storage");
            require(model.encoder.layers[0]->self_attn.q_proj.weight.storage_type() == GGML_TYPE_Q4_0,
                    "HuBERT linear weight did not preserve Q4 storage");
        } else {
            for (int index = 1; index < argc; ++index) {
                gpt_sovits::HubertModel model;
                require(model.load(argv[index], backend), "real HuBERT artifact failed to load through nn::io");
                require(model.state_dict() != nullptr, "real HuBERT artifact did not retain StateDict ownership");
                require(model.conv_layers[0].weight.is_bound() && !model.encoder.layers.empty(),
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
