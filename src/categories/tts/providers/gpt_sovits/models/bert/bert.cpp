#include "providers/gpt_sovits/models/bert/bert.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include "ops/ops.h"
#include "nn/nn.h"
#include "nn/io/gguf.h"
#include "nn/io/load.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <exception>
#include <memory>

namespace gpt_sovits {

bool BertModel::read_metadata(const struct gguf_context* ctx_gguf) {
    int kid = gguf_find_key(ctx_gguf, "attention.head_count");
    n_heads = kid >= 0 && gguf_get_kv_type(ctx_gguf, kid) == GGUF_TYPE_UINT32
        ? static_cast<int>(gguf_get_val_u32(ctx_gguf, kid)) : 0;
    return n_heads > 0;
}

bool BertModel::load(const std::string& path, ggml_backend_t backend) {
    if (!backend) return false;

    std::unique_ptr<nn::io::GGUFSource> source;
    try {
        source = std::make_unique<nn::io::GGUFSource>(path);
    } catch (const std::exception& error) {
        std::cerr << "[BERT] " << error.what() << "\n";
        return false;
    }

    const gguf_context* metadata = source->metadata_context();
    const int64_t architecture_key = gguf_find_key(metadata, "general.architecture");
    const int64_t version_key = gguf_find_key(metadata, "gpt_sovits.version");
    if (architecture_key < 0 || gguf_get_kv_type(metadata, architecture_key) != GGUF_TYPE_STRING ||
        std::string(gguf_get_val_str(metadata, architecture_key)) != "gpt_sovits_bert" ||
        version_key < 0 || gguf_get_kv_type(metadata, version_key) != GGUF_TYPE_STRING ||
        coarse_version_from_string(gguf_get_val_str(metadata, version_key)) == 0 || !read_metadata(metadata)) {
        std::cerr << "[BERT] Invalid architecture, version, or attention metadata.\n";
        return false;
    }

    const auto embedding_index = source->find("word_embeddings.weight");
    if (!embedding_index) {
        std::cerr << "[BERT] Word embedding tensor is missing.\n";
        return false;
    }
    const nn::Shape& embedding_shape = source->info(*embedding_index).logical_shape;
    if (embedding_shape.empty() || embedding_shape[0] <= 0 || embedding_shape[0] % n_heads != 0) {
        std::cerr << "[BERT] Invalid embedding shape or attention head count.\n";
        return false;
    }
    const int head_dim = static_cast<int>(embedding_shape[0] / n_heads);
    for (auto& layer : encoder.layers) {
        layer.self_attn.n_heads = n_heads;
        layer.self_attn.head_dim = head_dim;
    }

    nn::io::LoadResult loaded = nn::io::load_into(*this, *source, backend);
    if (!loaded) {
        std::cerr << "[BERT] " << loaded.error << "\n";
        return false;
    }
    this->to(backend);

    return true;
}

struct ggml_tensor* BertModel::forward(
    nn::Context& context,
    struct ggml_tensor* input_ids,
    struct ggml_tensor* position_ids,
    struct ggml_tensor* token_type_ids,
    ggml_backend_t backend
) {
    struct ggml_context* ctx = context.native_handle();
    struct ggml_tensor* x = ggml_add(
        ctx,
        ggml_add(ctx, word_embeddings(context, input_ids), position_embeddings(context, position_ids)),
        token_type_embeddings(context, token_type_ids));
    if (x->type != GGML_TYPE_F32) {
        x = ggml_cont(ctx, ggml_cast(ctx, x, GGML_TYPE_F32));
    }
    x = embeddings_ln(context, x);
    return encoder(context, x, nullptr, backend);
}

struct ggml_tensor* BertRunner::forward(struct ggml_context* ctx_graph, const std::vector<int32_t>& input_ids) {
    int seq_len = (int)input_ids.size();
    nn::Context output_context = nn::Context::borrow(ctx_graph);
    if (seq_len == 0) {
        return output_context.empty<float>("bert.output", {1024, 0});
    }

    nn::Context execution_context(128 * 1024 * 1024, true);
    
    // Inputs belong to this graph execution, not the shared model replica.
    std::vector<int32_t> position_ids(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        position_ids[i] = i + 2;
    }
    std::vector<int32_t> token_type_ids(seq_len, 0);
    struct ggml_tensor* input_ids_tensor = execution_context.input<int32_t>("bert.input_ids", {seq_len}, nn::data::borrow(input_ids));
    struct ggml_tensor* position_ids_tensor = execution_context.input<int32_t>("bert.position_ids", {seq_len}, nn::data::borrow(position_ids));
    struct ggml_tensor* token_type_ids_tensor = execution_context.input<int32_t>("bert.token_type_ids", {seq_len}, nn::data::borrow(token_type_ids));
    
    struct ggml_tensor* x = model_.forward(
        execution_context, input_ids_tensor, position_ids_tensor, token_type_ids_tensor, backend_);
    
    struct ggml_cgraph* gf = execution_context.build(x);
 
    // Create and use the graph allocator (ggml_gallocr) for memory planning and alignment
    bool is_local_galloc = false;
    ggml_gallocr_t galloc = allocator_;
    if (!galloc) {
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        is_local_galloc = true;
        if (!galloc) {
            std::cerr << "[BERT] Error: Failed to create graph allocator (gallocr)!\n";
            return nullptr;
        }
    }
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        std::cerr << "[BERT] Error: Failed to allocate graph using gallocr!\n";
        if (is_local_galloc) {
            ggml_gallocr_free(galloc);
        }
        return nullptr;
    }

    execution_context.materialize();
    
    ggml_ops_ext::ops_backend_graph_compute(backend_, gf);
    
    // Retrieve computed output features back to CPU
    int out_elements = 1024 * seq_len;
    std::vector<float> host_out(out_elements);
    ggml_backend_tensor_get(x, host_out.data(), 0, out_elements * sizeof(float));
    if (is_local_galloc) {
        ggml_gallocr_free(galloc);
    }
    struct ggml_tensor* out_tensor = output_context.empty<float>("bert.output", {1024, seq_len});
    output_context.write(out_tensor, host_out.data(), host_out.size());
    return out_tensor;
}

} // namespace gpt_sovits
