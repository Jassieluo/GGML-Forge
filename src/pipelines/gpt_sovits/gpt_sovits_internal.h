#pragma once

#include "gpt_sovits.h"
#include "models/models.h"
#include "ops/ops.h"
#include "pipeline_types.h"
#include "frontends/gpt_sovits/gpt_sovits_frontend.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <sstream>

namespace gpt_sovits {

struct CoutSilencer {
    std::streambuf* old_buf;
    std::stringstream null_stream;
    bool active;

    CoutSilencer(bool silencer_active);
    ~CoutSilencer();
};

class Impl {
public:
    struct InternalParams {
        std::string dict_dir;
        std::string hubert_model_path;
        std::string bert_model_path;
        std::string t2s_model_path;
        std::string vits_model_path;
        int n_threads = 4;
        bool use_gpu = true;
    } params;

    std::unique_ptr<GPTSoVITSFrontend> frontend;

    // GGML Backends
    ggml_backend_t static_backend = nullptr;
    ggml_backend_t dynamic_backend = nullptr;
    ggml_backend_t vits_backend = nullptr;
    ggml_backend_t vits_target_backend = nullptr;
    ggml_backend_t bert_backend = nullptr;
    ggml_backend_t t2s_backend = nullptr;

    struct ggml_threadpool* static_threadpool = nullptr;
    struct ggml_threadpool* dynamic_threadpool = nullptr;
    struct ggml_threadpool* vits_threadpool = nullptr;
    struct ggml_threadpool* bert_threadpool = nullptr;
    struct ggml_threadpool* t2s_threadpool = nullptr;

    // persistent Static base models (CNHuBERT and RoBERTa BERT)
    std::unique_ptr<HubertModel> hubert;
    std::unique_ptr<BertModel> bert;

    // Dynamic speaker weights (hot-swapped)
    std::unique_ptr<T2SModel> t2s;
    std::unique_ptr<VITSModel> vits;

    // Double buffers for glitch-free speaker hot-swapping
    std::unique_ptr<T2SModel> t2s_standby;
    std::unique_ptr<VITSModel> vits_standby;

    // Resident Cache in VRAM
    std::unordered_map<std::string, PromptCache> prompt_caches;

    // Engine-resident output buffer for stable C-pointer access
    std::vector<float> last_synthesized_audio;

    // Persistent VITS graph allocator
    ggml_gallocr_t vits_galloc = nullptr;

    // Model Slots and Backend Registry
    struct ModelSlot {
        std::string path;
        std::string device = "cpu";
        bool is_resident = true;
        bool is_loaded = false;
    };
    ModelSlot slots[4];
    std::unordered_map<std::string, ggml_backend_t> device_backends;
    struct ggml_threadpool* shared_cpu_threadpool = nullptr;
    bool bypass_offload = false;

    Impl(
        const char* dict_dir,
        const char* hubert_model_path,
        const char* bert_model_path,
        const char* t2s_model_path,
        const char* vits_model_path,
        int n_threads,
        int backend_mode,
        const char* device_name = nullptr
    );
    ~Impl();

    // Member helper functions
    ggml_backend_t get_backend_for_device(const std::string& device_name);
    void create_and_bind_shared_threadpool(ggml_backend_t backend);
    bool load_model(int model_type);
    void offload_model(int model_type);
    void configure_sycl_cache_impl();
};

} // namespace gpt_sovits
