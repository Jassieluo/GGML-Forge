#pragma once

#include "categories/tts/gpt_sovits.h"
#include "providers/gpt_sovits/models/models.h"
#include "ops/ops.h"
#include "provider_types.h"
#include "providers/gpt_sovits/frontend/text_frontend.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <sstream>
#include <random>
#include <mutex>

namespace tts { struct RuntimeContext; }

namespace gpt_sovits {

struct SharedStaticArtifacts {
    std::mutex mutex;
    std::unordered_map<std::string, std::weak_ptr<HubertModel>> hubert;
    std::unordered_map<std::string, std::weak_ptr<BertModel>> bert;
    std::unordered_map<std::string, std::weak_ptr<ERes2NetV2>> speaker_encoder;
};

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
        std::string speaker_encoder_model_path;
        int n_threads = 1;
        bool use_gpu = true;
    } params;

    std::unique_ptr<ITextFrontend> frontend;

    // GGML Backends
    ggml_backend_t static_backend = nullptr;
    ggml_backend_t dynamic_backend = nullptr;
    ggml_backend_t vits_backend = nullptr;
    ggml_backend_t vits_target_backend = nullptr;
    ggml_backend_t bert_backend = nullptr;
    ggml_backend_t t2s_backend = nullptr;
    ggml_backend_t speaker_encoder_backend = nullptr;

    struct ggml_threadpool* static_threadpool = nullptr;
    struct ggml_threadpool* dynamic_threadpool = nullptr;
    struct ggml_threadpool* vits_threadpool = nullptr;
    struct ggml_threadpool* bert_threadpool = nullptr;
    struct ggml_threadpool* t2s_threadpool = nullptr;

    // persistent Static base models (CNHuBERT and RoBERTa BERT)
    std::shared_ptr<HubertModel> hubert;
    std::shared_ptr<BertModel> bert;
    std::shared_ptr<ERes2NetV2> speaker_encoder;
    std::shared_ptr<SharedStaticArtifacts> shared_static_artifacts;

    // Dynamic speaker weights (hot-swapped)
    std::unique_ptr<T2SModel> t2s;
    std::unique_ptr<VITSModel> vits;
    int vits_version = 2;

    // Double buffers for glitch-free speaker hot-swapping
    std::unique_ptr<T2SModel> t2s_standby;
    std::unique_ptr<VITSModel> vits_standby;

    // Resident Cache in VRAM
    std::unordered_map<std::string, PromptCache> prompt_caches;

    // Host-side prompt VQ constants, refreshed when the active VITS profile changes.
    std::string prompt_vq_profile_id;
    std::vector<float> prompt_vq_codebook;
    std::vector<float> prompt_vq_codebook_norms;
    std::vector<float> prompt_vq_projection_weights;
    std::vector<float> prompt_vq_projection_bias;

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
    ModelSlot slots[5];
    std::unordered_map<std::string, ggml_backend_t> device_backends;
    struct ggml_threadpool* shared_cpu_threadpool = nullptr;
    bool bypass_offload = false;
    bool defer_model_compat_validation = false;
    bool initialized = false;
    std::mt19937 default_rng{42u};
    std::mt19937* active_rng = nullptr;

    // Best-effort progress reporting, attached per request by the session
    // layer. progress_base/progress_range window the active segment inside the
    // whole request so multi-segment synthesis reports one monotonic ramp.
    std::function<void(float)> progress_fn;
    float progress_base = 0.0f;
    float progress_range = 1.0f;

    Impl(
        const char* dict_dir,
        const char* hubert_model_path,
        const char* bert_model_path,
        const char* t2s_model_path,
        const char* vits_model_path,
        const char* speaker_encoder_model_path,
        int n_threads,
        int backend_mode,
        const char* device_name = nullptr,
        const tts::RuntimeContext* runtime_context = nullptr,
        std::shared_ptr<SharedStaticArtifacts> shared_artifacts = nullptr
    );
    ~Impl();

    // Member helper functions
    ggml_backend_t get_backend_for_device(const std::string& device_name);
    void create_and_bind_shared_threadpool(ggml_backend_t backend);
    bool validate_t2s_vits_compatibility() const;
    bool load_model(int model_type);
    void offload_model(int model_type);
    void configure_sycl_cache_impl();
};

} // namespace gpt_sovits
