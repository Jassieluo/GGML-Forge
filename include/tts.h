#pragma once

#include <stddef.h>
#include <stdint.h>

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
typedef struct tts_context* tts_context_ptr;

// Configuration settings for model loading
struct tts_model_params {
    int32_t main_gpu;     // Main GPU device index (e.g. 0)
    bool    use_gpu;      // Toggle GPU acceleration
    bool    vocab_only;   // Only load vocabulary metadata
};

// Configuration settings for inference contexts
struct tts_context_params {
    uint32_t n_threads;    // Thread pool size for CPU execution
};

// 1. Model Lifecycle Management
TTS_API tts_model_ptr tts_load_model_from_file(const char* path, struct tts_model_params params);
TTS_API void          tts_free_model(tts_model_ptr model);

// 2. Inference Context Lifecycle Management
TTS_API tts_context_ptr tts_new_context_with_model(tts_model_ptr model, struct tts_context_params params);
TTS_API void            tts_free_context(tts_context_ptr ctx);

// 3. Global Configuration APIs
TTS_API void tts_set_log_level(int level); // 0 = Info, 1 = Warning, 2 = Error, 3 = Debug

// 4. Speech Synthesis Pipeline API
// Synthesizes text to float32 mono audio waveform (22050Hz or similar, depending on model)
// Returns pointer to internal buffer (owned by context, valid until next synthesis)
TTS_API const float* tts_synthesize(
    tts_context_ptr ctx,
    const char*     text,
    const char*     lang,
    float           speed,
    int32_t*        out_samples_count
);

#ifdef __cplusplus
}
#endif
