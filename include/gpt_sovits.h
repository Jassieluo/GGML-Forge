#pragma once

#include <cstddef>

#if defined(_WIN32)
#  if defined(GPT_SOVITS_BUILD_SHARED)
#    define GPT_SOVITS_API __declspec(dllexport)
#  elif defined(GPT_SOVITS_USE_SHARED)
#    define GPT_SOVITS_API __declspec(dllimport)
#  else
#    define GPT_SOVITS_API
#  endif
#else
#  define GPT_SOVITS_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle representing the GPT-SoVITS engine instance
typedef void* gpt_sovits_engine_t;

// Configure SYCL JIT persistent cache settings programmatically.
// Must be called BEFORE initializing the engine.
// If cache_dir is NULL or empty, it defaults to "sycl_cache" in the current project directory.
GPT_SOVITS_API void gpt_sovits_configure_sycl_cache(bool enable_cache, const char* cache_dir);

// Enable or disable console log output (stdout) during inference programmatically
GPT_SOVITS_API void gpt_sovits_set_log_enabled(bool enabled);

// Set the number of ODE integration steps for the CFM model (V3/V4). Default is 10.
GPT_SOVITS_API void gpt_sovits_set_cfm_steps(gpt_sovits_engine_t engine, int steps);

// Get target version from loaded VITS model
GPT_SOVITS_API int gpt_sovits_get_version(gpt_sovits_engine_t engine);

// Get output sampling rate of the engine based on loaded VITS model
GPT_SOVITS_API int gpt_sovits_get_sampling_rate(gpt_sovits_engine_t engine);

// Initialize the provider engine with explicit device selection.
// backend_mode: 0 = CPU Only, 1 = GPU Only, 2 = Hybrid Mode, 3 = Inverse Hybrid Mode
// device_name: name of the specific device (e.g. "CUDA0", "SYCL0"). If NULL or empty, defaults to first found GPU.
GPT_SOVITS_API gpt_sovits_engine_t gpt_sovits_init_with_device(
    const char* dict_dir,
    const char* hubert_model_path,
    const char* bert_model_path,
    const char* t2s_model_path,
    const char* vits_model_path,
    int n_threads,
    int backend_mode,
    const char* device_name
);

// Free engine resources
GPT_SOVITS_API void gpt_sovits_free(gpt_sovits_engine_t engine);

// Set model path, target backend device, and residency policy for a specific model slot.
// model_type: 0 = Hubert, 1 = BERT, 2 = T2S, 3 = VITS
// model_path: path to the GGUF model file
// device_name: name of the specific device (e.g. "CPU", "CUDA0", "SYCL0"). If NULL/empty, defaults to CPU or first found GPU.
// is_resident: if true, the model stays resident in memory. If false, it is loaded dynamically during inference and offloaded immediately after.
GPT_SOVITS_API void gpt_sovits_set_model_config(
    gpt_sovits_engine_t engine,
    int model_type,
    const char* model_path,
    const char* device_name,
    bool is_resident
);

// Manually load a specific model slot
GPT_SOVITS_API bool gpt_sovits_load_model(gpt_sovits_engine_t engine, int model_type);

// Manually offload a specific model slot to reclaim memory/VRAM
GPT_SOVITS_API void gpt_sovits_offload_model(gpt_sovits_engine_t engine, int model_type);

// Check if a specific model slot is currently loaded
GPT_SOVITS_API bool gpt_sovits_is_model_loaded(gpt_sovits_engine_t engine, int model_type);

// Hot-swap speaker weights dynamically
GPT_SOVITS_API bool gpt_sovits_load_speaker(
    gpt_sovits_engine_t engine,
    const char* t2s_model_path,
    const char* vits_model_path
);

// Pre-compute speaker prompt style cache
GPT_SOVITS_API void gpt_sovits_get_or_create_prompt_cache(
    gpt_sovits_engine_t engine,
    const char* cache_id,
    const float* ref_audio_data,
    size_t ref_audio_len,
    int ref_audio_sample_rate,
    const char* ref_text,
    const char* ref_language,
    const float* sv_emb_data,
    size_t sv_emb_len
);

// Synthesize target speech using cached prompt, returns pointer to samples (resident in engine)
GPT_SOVITS_API const float* gpt_sovits_synthesize_with_cache(
    gpt_sovits_engine_t engine,
    const char* text,
    const char* language,
    const char* cache_id,
    float speed,
    int* out_num_samples
);

// Synthesize target speech with on-the-fly reference processing.
GPT_SOVITS_API const float* gpt_sovits_synthesize(
    gpt_sovits_engine_t engine,
    const char* text,
    const char* language,
    const float* ref_audio_data,
    size_t ref_audio_len,
    int ref_audio_sample_rate,
    const char* ref_text,
    const char* ref_language,
    float speed,
    int* out_num_samples
);

// VoiceManager high-level service wrapper C API

typedef void* gpt_sovits_voice_manager_t;

GPT_SOVITS_API gpt_sovits_voice_manager_t gpt_sovits_voice_manager_init(gpt_sovits_engine_t engine);

GPT_SOVITS_API void gpt_sovits_voice_manager_free(gpt_sovits_voice_manager_t manager);

GPT_SOVITS_API bool gpt_sovits_voice_manager_register_character(
    gpt_sovits_voice_manager_t manager,
    const char* voices_root_dir,
    const char* character_id
);

GPT_SOVITS_API const char* gpt_sovits_voice_manager_get_cache_id(
    gpt_sovits_voice_manager_t manager,
    const char* character_id
);

GPT_SOVITS_API const float* gpt_sovits_voice_manager_synthesize(
    gpt_sovits_voice_manager_t manager,
    const char* character_id,
    const char* text,
    const char* language,
    float speed,
    int* out_num_samples
);

#ifdef __cplusplus
}
#endif
