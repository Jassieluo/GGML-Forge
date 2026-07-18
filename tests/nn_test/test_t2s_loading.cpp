#include "models/lm/gpt_t2s/gpt_t2s.h"

#include "ggml-backend.h"
#include "gguf.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
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
    TemporaryFile& operator=(TemporaryFile&&) = delete;

    ~TemporaryFile() {
        if (path.empty()) return;
        std::error_code error;
        std::filesystem::remove(path, error);
    }
};

ggml_tensor* add_tensor(ggml_context* context, gguf_context* output, const char* name,
                        const std::vector<int64_t>& shape) {
    ggml_tensor* tensor = ggml_new_tensor(context, GGML_TYPE_F32, static_cast<int>(shape.size()), shape.data());
    ggml_set_name(tensor, name);
    std::memset(tensor->data, 0, ggml_nbytes(tensor));
    gguf_add_tensor(output, tensor);
    return tensor;
}

TemporaryFile write_t2s_fixture() {
    const auto unique = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    TemporaryFile file(std::filesystem::temp_directory_path() /
                       ("nn-t2s-load-" + std::to_string(unique) + ".gguf"));

    std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors(
        ggml_init({4 * 1024 * 1024, nullptr, false}), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> output(gguf_init_empty(), gguf_free);
    require(tensors != nullptr && output != nullptr, "failed to create T2S fixture contexts");

    gguf_set_val_str(output.get(), "general.architecture", "gpt_sovits_t2s");
    gguf_set_val_str(output.get(), "gpt_sovits.version", "v2");
    gguf_set_val_u32(output.get(), "gpt_sovits.t2s.family", 1);
    gguf_set_val_u32(output.get(), "gpt_sovits.t2s.n_layers", 1);
    gguf_set_val_u32(output.get(), "gpt_sovits.t2s.hidden_dim", 4);
    gguf_set_val_u32(output.get(), "attention.head_count", 2);

    add_tensor(tensors.get(), output.get(), "word_embeddings.weight", {4, 8});
    add_tensor(tensors.get(), output.get(), "audio_embeddings.weight", {4, 8});
    add_tensor(tensors.get(), output.get(), "bert_proj.weight", {1024, 4});
    add_tensor(tensors.get(), output.get(), "bert_proj.bias", {4});
    add_tensor(tensors.get(), output.get(), "predict.weight", {4, 8});

    const std::string layer = "decoder.layers.0.";
    for (const char* projection : {"q", "k", "v"}) {
        add_tensor(tensors.get(), output.get(), (layer + "self_attn." + projection + "_proj.weight").c_str(), {4, 4});
    }
    add_tensor(tensors.get(), output.get(), (layer + "self_attn.out_proj.weight").c_str(), {4, 4});
    add_tensor(tensors.get(), output.get(), (layer + "ln1.weight").c_str(), {4});
    add_tensor(tensors.get(), output.get(), (layer + "ln1.bias").c_str(), {4});
    add_tensor(tensors.get(), output.get(), (layer + "ln2.weight").c_str(), {4});
    add_tensor(tensors.get(), output.get(), (layer + "ln2.bias").c_str(), {4});
    add_tensor(tensors.get(), output.get(), (layer + "ffn.w1.weight").c_str(), {4, 8});
    add_tensor(tensors.get(), output.get(), (layer + "ffn.w1.bias").c_str(), {8});
    add_tensor(tensors.get(), output.get(), (layer + "ffn.w2.weight").c_str(), {8, 4});
    add_tensor(tensors.get(), output.get(), (layer + "ffn.w2.bias").c_str(), {4});

    ggml_tensor* text_alpha = add_tensor(tensors.get(), output.get(), "text_position_alpha", {1});
    ggml_tensor* audio_alpha = add_tensor(tensors.get(), output.get(), "audio_position_alpha", {1});
    *static_cast<float*>(text_alpha->data) = 1.25f;
    *static_cast<float*>(audio_alpha->data) = 0.75f;

    require(gguf_write_to_file(output.get(), file.path.string().c_str(), false), "failed to write T2S fixture");
    return file;
}

} // namespace

int main(int argc, char** argv) {
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    require(backend != nullptr, "failed to initialize CPU backend");

    try {
        if (argc == 1) {
            TemporaryFile file = write_t2s_fixture();
            gpt_sovits::T2SModel model;
            require(model.load(file.path.string(), backend), "T2S failed to load through nn::io");
            require(model.state_dict() != nullptr, "T2S did not retain StateDict ownership");
            require(model.n_layers == 1 && model.n_heads == 2 && model.head_dim == 2, "T2S topology mismatch");
            require(model.decoder.layers.size() == 1, "T2S decoder was not rebuilt from metadata");
            require(model.text_position_alpha.is_bound() && model.audio_position_alpha.is_bound(),
                    "T2S position alpha parameters were not bound");
            require(model.text_alpha == 1.25f && model.audio_alpha == 0.75f,
                    "T2S position alpha values mismatch");
        } else {
            for (int index = 1; index < argc; ++index) {
                gpt_sovits::T2SModel model;
                require(model.load(argv[index], backend), "real T2S artifact failed to load through nn::io");
                require(model.state_dict() != nullptr, "real T2S artifact did not retain StateDict ownership");
                require(model.n_layers > 0 && model.n_heads > 0 && model.head_dim > 0,
                        "real T2S artifact has invalid topology");
                require(static_cast<int>(model.decoder.layers.size()) == model.n_layers,
                        "real T2S decoder layer count mismatch");
                std::cout << "T2S artifact passed: " << argv[index]
                          << ", version=" << model.version_string
                          << ", layers=" << model.n_layers
                          << ", heads=" << model.n_heads << '\n';
            }
        }
    } catch (...) {
        ggml_backend_free(backend);
        throw;
    }

    ggml_backend_free(backend);
    std::cout << "T2S loading test passed\n";
    return 0;
}
