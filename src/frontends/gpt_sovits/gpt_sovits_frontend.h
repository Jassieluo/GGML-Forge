#pragma once

#include "text_frontend.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

// Forward declarations of external libraries & types
namespace phonemizer {
class Phonemizer;
struct PhonemizerResult;
}

struct ggml_context;
typedef struct ggml_backend* ggml_backend_t;
typedef struct ggml_gallocr* ggml_gallocr_t;

namespace gpt_sovits {

struct BertModel; // Forward declaration

class GPTSoVITSFrontend : public ITextFrontend {
public:
    GPTSoVITSFrontend(const std::string& dict_dir);
    ~GPTSoVITSFrontend() override;

    // Initialize phonemizer and pre-populate phone symbol dictionary
    bool initialize() override;

    // Set model version (1 or 2)
    void set_symbol_version(int version) override;

    // Map phone symbol to its index in get_phone_symbols()
    int32_t phone_to_id(const std::string& phone) const override;

    // Load custom BERT vocabulary from file
    bool load_bert_vocab(const std::string& vocab_path);

    // Convert clean UTF-8 text into BERT token IDs (with [CLS] and [SEP])
    std::vector<int32_t> bert_tokenize(const std::string& text) const;

    // Split text paragraph into synthesis segments using target split method
    std::vector<std::string> split_text(const std::string& text, const std::string& split_method) const override;

    // Full text preprocessing pipeline (Mixed-mode phonemization + BERT features extraction & alignment)
    bool process(
        const std::string& text,
        const std::string& language,
        BertModel* bert_model,
        struct ggml_context* ctx_graph,
        ggml_backend_t bert_backend,
        FrontendResult& out_result,
        ggml_gallocr_t galloc = nullptr
    ) override;

private:
    std::string dict_dir_;
    std::unique_ptr<phonemizer::Phonemizer> phonemizer_;
    std::unordered_map<std::string, int32_t> phone_to_id_map_;
    std::unordered_map<std::string, int32_t> bert_vocab_;
    int version_ = 2;
};

} // namespace gpt_sovits
