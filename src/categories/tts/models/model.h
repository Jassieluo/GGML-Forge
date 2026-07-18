#pragma once

#include "ggml.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace tts {

// Forward declarations
struct tts_model;
struct tts_session;

// Represent a single model layer/tensor mappings
struct model_tensor {
    struct ggml_tensor* tensor = nullptr;
    std::string name;
};

// Abstract base class representing a model architecture loaded from a GGUF file
class ModelArch {
public:
    virtual ~ModelArch() = default;

    // Load tensor mapping and metadata from GGUF context
    virtual bool load_tensors(
        struct ggml_context* ctx,
        const std::unordered_map<std::string, struct ggml_tensor*>& tensors
    ) = 0;

    // Build the GGML computational graph for inference
    // - ctx: ggml context used for nodes creation
    // - inputs: map of named input tensors (e.g., "tokens", "latents")
    virtual struct ggml_cgraph* build_graph(
        struct ggml_context* ctx,
        const std::unordered_map<std::string, struct ggml_tensor*>& inputs
    ) = 0;
};

// Model registry factory function
// Creates the correct ModelArch instance based on the GGUF "general.architecture" metadata string
std::unique_ptr<ModelArch> create_model_arch(const std::string& arch_name);

} // namespace tts
