#include "models/text/bert/bert.h"

#include "ggml-backend.h"
#include "gguf.h"

#include <chrono>
#include <cmath>
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

void add_f32(ggml_context* context, gguf_context* output, const std::string& name,
             const std::vector<int64_t>& shape) {
    ggml_tensor* tensor = ggml_new_tensor(context, GGML_TYPE_F32, static_cast<int>(shape.size()), shape.data());
    ggml_set_name(tensor, name.c_str());
    std::memset(tensor->data, 0, ggml_nbytes(tensor));
    gguf_add_tensor(output, tensor);
}

void add_q4_embedding(ggml_context* context, gguf_context* output, const char* name, int64_t rows) {
    constexpr int64_t width = 32;
    ggml_tensor* tensor = ggml_new_tensor_2d(context, GGML_TYPE_Q4_0, width, rows);
    ggml_set_name(tensor, name);
    std::vector<float> values(static_cast<size_t>(width * rows));
    for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin(static_cast<float>(i) * 0.03f);
    const size_t written = ggml_quantize_chunk(GGML_TYPE_Q4_0, values.data(), tensor->data, 0, rows, width, nullptr);
    require(written == ggml_nbytes(tensor), "BERT fixture quantization size mismatch");
    gguf_add_tensor(output, tensor);
}

TemporaryFile write_bert_fixture() {
    const auto unique = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    TemporaryFile file(std::filesystem::temp_directory_path() /
                       ("nn-bert-load-" + std::to_string(unique) + ".gguf"));
    std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors(
        ggml_init({16 * 1024 * 1024, nullptr, false}), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> output(gguf_init_empty(), gguf_free);
    require(tensors != nullptr && output != nullptr, "failed to create BERT fixture contexts");

    gguf_set_val_str(output.get(), "general.architecture", "gpt_sovits_bert");
    gguf_set_val_str(output.get(), "gpt_sovits.version", "v2");
    gguf_set_val_u32(output.get(), "attention.head_count", 4);
    add_q4_embedding(tensors.get(), output.get(), "word_embeddings.weight", 8);
    add_q4_embedding(tensors.get(), output.get(), "position_embeddings.weight", 16);
    add_q4_embedding(tensors.get(), output.get(), "token_type_embeddings.weight", 2);

    for (int layer = 0; layer < 22; ++layer) {
        const std::string prefix = "encoder.layers." + std::to_string(layer) + ".";
        add_f32(tensors.get(), output.get(), prefix + "self_attn.q_proj.weight", {32, 32});
        add_f32(tensors.get(), output.get(), prefix + "self_attn.k_proj.weight", {32, 32});
        add_f32(tensors.get(), output.get(), prefix + "self_attn.v_proj.weight", {32, 32});
        add_f32(tensors.get(), output.get(), prefix + "self_attn.out_proj.weight", {32, 32});
        add_f32(tensors.get(), output.get(), prefix + "ffn.w1.weight", {32, 64});
        add_f32(tensors.get(), output.get(), prefix + "ffn.w2.weight", {64, 32});
    }

    require(gguf_write_to_file(output.get(), file.path.string().c_str(), false), "failed to write BERT fixture");
    return file;
}

} // namespace

int main(int argc, char** argv) {
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    require(backend != nullptr, "failed to initialize CPU backend");

    try {
        if (argc == 1) {
            TemporaryFile file = write_bert_fixture();
            gpt_sovits::BertModel model;
            require(model.load(file.path.string(), backend), "BERT failed to load through nn::io");
            require(model.state_dict() != nullptr, "BERT did not retain StateDict ownership");
            require(model.n_heads == 4, "BERT attention metadata mismatch");
            require(model.word_embeddings.weight.storage_type() == GGML_TYPE_Q4_0,
                    "BERT Q4 embedding was not preserved for direct execution");
            require(model.encoder.layers[0].self_attn.q_proj.weight.storage_type() == GGML_TYPE_F32,
                    "BERT linear weight did not preserve native storage");
            require(model.encoder.layers[0].self_attn.head_dim == 8, "BERT head dimension mismatch");
        } else {
            for (int index = 1; index < argc; ++index) {
                gpt_sovits::BertModel model;
                require(model.load(argv[index], backend), "real BERT artifact failed to load through nn::io");
                require(model.state_dict() != nullptr, "real BERT artifact did not retain StateDict ownership");
                require(model.n_heads > 0 && !model.encoder.layers.empty(), "real BERT topology is invalid");
                std::cout << "BERT artifact passed: " << argv[index]
                          << ", heads=" << model.n_heads << '\n';
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "BERT loading test failed: " << error.what() << "\n";
        ggml_backend_free(backend);
        return 1;
    }

    ggml_backend_free(backend);
    std::cout << "BERT loading test passed\n";
    return 0;
}
