#pragma once

#include <string>
#include <vector>
#include <memory>

struct ggml_context;
typedef struct ggml_backend* ggml_backend_t;
typedef struct ggml_gallocr* ggml_gallocr_t;

namespace gpt_sovits {

// Provider-specific frontend contract: processing includes GPT-SoVITS BERT features.

struct BertModel;

struct FrontendResult {
    std::vector<std::string> phones;
    std::vector<int32_t> phone_ids;
    std::vector<int> word2ph;
    std::vector<float> bert_features;
};

class ITextFrontend {
public:
    virtual ~ITextFrontend() = default;

    // Initialize phonemizer/G2P dictionaries
    virtual bool initialize() = 0;

    // Map phone symbol to its index ID
    virtual int32_t phone_to_id(const std::string& phone) const = 0;

    // Split text paragraph into synthesis segments
    virtual std::vector<std::string> split_text(const std::string& text, const std::string& split_method) const = 0;

    // Set model version (if supported, e.g. V1 or V2)
    virtual void set_symbol_version(int version) = 0;

    // Full text preprocessing pipeline (Mixed-mode phonemization + BERT features extraction & alignment)
    virtual bool process(
        const std::string& text,
        const std::string& language,
        BertModel* bert_model,
        struct ggml_context* ctx_graph,
        ggml_backend_t bert_backend,
        FrontendResult& out_result,
        ggml_gallocr_t galloc = nullptr
    ) = 0;
};

} // namespace gpt_sovits
