#pragma once

#include <stddef.h>
#include <stdint.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef _WIN32
#  if defined(TTS_BUILD_SHARED)
#    define TTS_API __declspec(dllexport)
#  elif defined(TTS_USE_SHARED)
#    define TTS_API __declspec(dllimport)
#  else
#    define TTS_API
#  endif
#else
#  define TTS_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handles representing core structures
typedef struct tts_model* tts_model_ptr;
typedef struct tts_session* tts_session_ptr;
typedef struct tts_runtime* tts_runtime_ptr;
typedef void (*tts_audio_chunk_callback)(const float* audio, size_t sample_count, void* user_data);
// Reports synthesis progress in [0, 1]. Called from the synthesis thread.
typedef void (*tts_progress_callback)(float progress, void* user_data);

// Machine-specific execution settings. These never belong in a model composition.
struct tts_runtime_params {
    const char* device; // "auto", "cpu", or an exact backend device name such as "CUDA0"
    uint32_t n_threads; // CPU worker count per execution lane.
    uint32_t max_concurrency; // Maximum number of provider execution lanes.
};

enum tts_component_residency {
    TTS_COMPONENT_RESIDENT = 0,
    TTS_COMPONENT_ON_DEMAND = 1,
};

struct tts_capabilities {
    bool streaming;
    bool voice_cloning;
    bool speaker_id;
    bool speaker_embedding;
    bool emotion;
    bool deterministic_seed;
    bool speed_control;
};

// 1. Runtime Lifecycle Management
TTS_API struct tts_runtime_params tts_runtime_default_params(void);
TTS_API tts_runtime_ptr tts_runtime_create(struct tts_runtime_params params);
TTS_API void            tts_runtime_free(tts_runtime_ptr runtime);
TTS_API const char*     tts_runtime_get_device(tts_runtime_ptr runtime);
TTS_API uint32_t        tts_runtime_get_thread_count(tts_runtime_ptr runtime);
TTS_API uint32_t        tts_runtime_get_max_concurrency(tts_runtime_ptr runtime);
// Sets a machine-specific policy for models loaded after this call. A null or
// empty device inherits the runtime default device.
TTS_API bool tts_runtime_set_component_policy(
    tts_runtime_ptr runtime,
    const char* component,
    const char* device,
    enum tts_component_residency residency
);

// 2. Model Lifecycle Management
TTS_API tts_model_ptr tts_load_model(tts_runtime_ptr runtime, const char* path);
TTS_API void          tts_free_model(tts_model_ptr model);
TTS_API const char*   tts_model_get_name(tts_model_ptr model);
TTS_API const char*   tts_model_get_provider(tts_model_ptr model);
TTS_API struct tts_capabilities tts_model_get_capabilities(tts_model_ptr model);

// 3. Session Lifecycle Management
TTS_API tts_session_ptr tts_create_session(tts_model_ptr model);
TTS_API void            tts_free_session(tts_session_ptr session);
TTS_API bool tts_session_set_reference(
    tts_session_ptr session,
    const float* audio,
    size_t sample_count,
    int32_t sample_rate,
    const char* text,
    const char* language
);
// Provider-defined options are attached to subsequent requests. Stable generic
// options include "speed" (float); providers may document additional names.
TTS_API bool tts_session_set_float_option(tts_session_ptr session, const char* name, float value);
TTS_API bool tts_session_set_string_option(tts_session_ptr session, const char* name, const char* value);
// Progress applies to subsequent synthesize calls on this session. Pass a null
// callback to detach. Providers report best-effort estimates; values are
// monotonic within one request.
TTS_API bool tts_session_set_progress_callback(
    tts_session_ptr session,
    tts_progress_callback callback,
    void* user_data
);
TTS_API int32_t tts_session_get_output_sample_rate(tts_session_ptr session);

// 4. Global Configuration APIs
TTS_API void tts_set_log_level(int level); // 0 = Info, 1 = Warning, 2 = Error, 3 = Debug

// 5. Speech Synthesis Pipeline API
// Synthesizes text to float32 mono audio waveform (22050Hz or similar, depending on model)
// Returns pointer to internal buffer (owned by context, valid until next synthesis)
TTS_API const float* tts_synthesize(
    tts_session_ptr session,
    const char*     text,
    const char*     lang,
    float           speed,
    int32_t*        out_samples_count
);
TTS_API bool tts_synthesize_streaming(
    tts_session_ptr session,
    const char* text,
    const char* lang,
    float speed,
    tts_audio_chunk_callback callback,
    void* user_data
);

#ifdef __cplusplus
}
#endif
