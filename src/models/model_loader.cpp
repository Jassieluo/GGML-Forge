#include "gguf_model.h"
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <cctype>



namespace gpt_sovits {

bool load_gguf_model(const std::string& path, GGUFModel& model, ggml_backend_t backend) {
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[load_gguf_model] Loading GGUF model: " << path << std::endl;

    // 1. Load weights metadata only (no_alloc = true, very fast, minimal memory)
    struct ggml_context* ggml_ctx_backend = nullptr;
    struct gguf_init_params params_backend = {
        /* .no_alloc = */ true,
        /* .ctx      = */ &ggml_ctx_backend
    };
    struct gguf_context* ctx_gguf = gguf_init_from_file(path.c_str(), params_backend);
    if (!ctx_gguf) {
        fprintf(stderr, "[GPT-SoVITS] Failed to load GGUF metadata from %s\n", path.c_str());
        return false;
    }

    // Read version metadata if present
    int kid_ver = gguf_find_key(ctx_gguf, "gpt_sovits.version");
    if (kid_ver != -1) {
        enum gguf_type type = gguf_get_kv_type(ctx_gguf, kid_ver);
        if (type == GGUF_TYPE_STRING) {
            std::string ver_str = gguf_get_val_str(ctx_gguf, kid_ver);
            model.version_string = ver_str;
            if (ver_str.find("v1") != std::string::npos || ver_str == "1") {
                model.version = 1;
            } else if (ver_str.find("v2") != std::string::npos || ver_str == "2") {
                model.version = 2;
            } else if (ver_str.find("v3") != std::string::npos || ver_str == "3") {
                model.version = 3;
            } else if (ver_str.find("v4") != std::string::npos || ver_str == "4") {
                model.version = 4;
            }
        } else if (type == GGUF_TYPE_UINT32) {
            uint32_t val = gguf_get_val_u32(ctx_gguf, kid_ver);
            model.version = (int)val;
            model.version_string = std::to_string(val);
        } else if (type == GGUF_TYPE_INT32) {
            int32_t val = gguf_get_val_i32(ctx_gguf, kid_ver);
            model.version = (int)val;
            model.version_string = std::to_string(val);
        }
    }

    // Trigger metadata lifecycle hook (e.g. read n_heads for BERT)
    model.on_read_metadata(ctx_gguf);

    // Pre-adjust tensor shapes in metadata context (e.g. transpose VITS convolutional weights)
    int n_tensors = (int)gguf_get_n_tensors(ctx_gguf);
    for (int i = 0; i < n_tensors; ++i) {
        std::string name = gguf_get_tensor_name(ctx_gguf, i);
        struct ggml_tensor* t_backend = ggml_get_tensor(ggml_ctx_backend, name.c_str());
        if (t_backend) {
            model.on_prepare_tensor(t_backend, name);
        }
    }

    // 2. Allocate the tensors on the backend
    std::cout << "[load_gguf_model Debug] Allocating backend tensors for " << path << ", ctx: " << ggml_ctx_backend << ", no_alloc: " << (ggml_get_no_alloc(ggml_ctx_backend) ? "true" : "false") << std::endl;
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ggml_ctx_backend, backend);
    if (!buffer) {
        fprintf(stderr, "[GPT-SoVITS] Failed to allocate backend buffer for GGUF: %s\n", path.c_str());
        gguf_free(ctx_gguf);
        return false;
    }
    model.backend_buffer = buffer;

    // 3. Open GGUF file once as raw binary to read and stream weight data directly
    FILE* file = fopen(path.c_str(), "rb");
    if (!file) {
        fprintf(stderr, "[GPT-SoVITS] Failed to open GGUF file for binary reading: %s\n", path.c_str());
        gguf_free(ctx_gguf);
        return false;
    }

    size_t data_offset = gguf_get_data_offset(ctx_gguf);

    // 4. Stream weight data directly from file to backend tensors
    for (int i = 0; i < n_tensors; ++i) {
        std::string name = gguf_get_tensor_name(ctx_gguf, i);
        struct ggml_tensor* t_backend = ggml_get_tensor(ggml_ctx_backend, name.c_str());
        if (!t_backend) continue;

        size_t tensor_offset = data_offset + gguf_get_tensor_offset(ctx_gguf, i);
        size_t tensor_size = gguf_get_tensor_size(ctx_gguf, i);
        enum ggml_type tensor_type = gguf_get_tensor_type(ctx_gguf, i);

        std::vector<uint8_t> temp_buf(tensor_size);
        if (fseek(file, (long)tensor_offset, SEEK_SET) != 0) {
            fprintf(stderr, "[GPT-SoVITS] Failed to seek to tensor offset for %s\n", name.c_str());
            fclose(file);
            gguf_free(ctx_gguf);
            return false;
        }
        if (fread(temp_buf.data(), 1, tensor_size, file) != tensor_size) {
            fprintf(stderr, "[GPT-SoVITS] Failed to read tensor data for %s\n", name.c_str());
            fclose(file);
            gguf_free(ctx_gguf);
            return false;
        }

        // Delegate weight copy / transposition to model sub-class. If not handled, copy raw bytes.
        if (!model.on_upload_tensor(t_backend, temp_buf.data(), tensor_size, tensor_type, name)) {
            ggml_backend_tensor_set(t_backend, temp_buf.data(), 0, tensor_size);
        }
    }

    fclose(file);

    // 5. Populate model tensors map and save context
    model.ctx = ggml_ctx_backend;
    for (int i = 0; i < n_tensors; ++i) {
        std::string name = gguf_get_tensor_name(ctx_gguf, i);
        struct ggml_tensor* tensor = ggml_get_tensor(ggml_ctx_backend, name.c_str());
        if (tensor) {
            model.tensors[name] = tensor;
        }
    }

    gguf_free(ctx_gguf);
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[load_gguf_model] GGUF loaded successfully." << std::endl;
    return true;
}

bool dequantize_tensor_to_f32(struct ggml_tensor* tensor, std::vector<float>& out_data, ggml_backend_t backend) {
    if (!tensor) return false;
    int64_t nelems = ggml_nelements(tensor);
    out_data.resize(nelems);
    
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, out_data.data(), 0, nelems * sizeof(float));
        return true;
    }
    
    // Create a temporary graph to evaluate the cast on backend
    struct ggml_init_params params = {
        /* .mem_size   = */ 10 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true
    };
    struct ggml_context* ctx_cast = ggml_init(params);
    if (!ctx_cast) return false;
    
    struct ggml_tensor* cast_node = ggml_cast(ctx_cast, tensor, GGML_TYPE_F32);
    struct ggml_cgraph* graph_cast = ggml_new_graph(ctx_cast);
    ggml_build_forward_expand(graph_cast, cast_node);
    
    // Allocate the output node on backend
    ggml_backend_buffer_t cast_buf = ggml_backend_alloc_ctx_tensors(ctx_cast, backend);
    if (!cast_buf) {
        ggml_free(ctx_cast);
        return false;
    }
    
    ggml_backend_graph_compute(backend, graph_cast);
    
    // Get F32 data back
    ggml_backend_tensor_get(cast_node, out_data.data(), 0, nelems * sizeof(float));
    
    ggml_backend_buffer_free(cast_buf);
    ggml_free(ctx_cast);
    return true;
}

} // namespace gpt_sovits
