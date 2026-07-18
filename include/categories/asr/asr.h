#pragma once

#include <stddef.h>
#include <stdint.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef _WIN32
#  if defined(ASR_BUILD_SHARED)
#    define ASR_API __declspec(dllexport)
#  elif defined(ASR_USE_SHARED)
#    define ASR_API __declspec(dllimport)
#  else
#    define ASR_API
#  endif
#else
#  define ASR_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct asr_runtime* asr_runtime_ptr;
typedef struct asr_model* asr_model_ptr;
typedef struct asr_session* asr_session_ptr;

enum asr_task {
    ASR_TASK_TRANSCRIBE = 0,
    ASR_TASK_TRANSLATE = 1,
};

enum asr_event_type {
    ASR_EVENT_LANGUAGE = 0,
    ASR_EVENT_SEGMENT = 1,
    ASR_EVENT_TOKEN = 2,
};

struct asr_runtime_params {
    const char* device; // "auto", "cpu", or an exact GGML backend device name.
    uint32_t n_threads;
};

struct asr_request_params {
    enum asr_task task;
    const char* language; // null, empty, or "auto" enables language detection.
    bool token_timestamps;
};

struct asr_capabilities {
    bool streaming;
    bool translation;
    bool language_detection;
    bool segment_timestamps;
    bool token_timestamps;
};

// Text and language pointers are borrowed and valid only during the callback.
// Timestamps are milliseconds; -1 means unavailable.
struct asr_event {
    enum asr_event_type type;
    const char* text;
    size_t text_length;
    const char* language;
    int64_t start_ms;
    int64_t end_ms;
    float confidence; // [0, 1], or -1 when unavailable.
};

typedef bool (*asr_event_callback)(const struct asr_event* event, void* user_data);

ASR_API struct asr_runtime_params asr_runtime_default_params(void);
ASR_API asr_runtime_ptr asr_runtime_create(struct asr_runtime_params params);
ASR_API void asr_runtime_free(asr_runtime_ptr runtime);
ASR_API const char* asr_runtime_get_device(asr_runtime_ptr runtime);
ASR_API uint32_t asr_runtime_get_thread_count(asr_runtime_ptr runtime);

ASR_API asr_model_ptr asr_load_model(asr_runtime_ptr runtime, const char* path);
ASR_API void asr_free_model(asr_model_ptr model);
ASR_API const char* asr_model_get_provider(asr_model_ptr model);
ASR_API struct asr_capabilities asr_model_get_capabilities(asr_model_ptr model);

ASR_API asr_session_ptr asr_create_session(asr_model_ptr model);
ASR_API void asr_free_session(asr_session_ptr session);
ASR_API bool asr_session_reset(asr_session_ptr session);

ASR_API struct asr_request_params asr_request_default_params(void);
ASR_API bool asr_transcribe(
    asr_session_ptr session,
    const float* mono_audio,
    size_t sample_count,
    int32_t sample_rate,
    struct asr_request_params params,
    asr_event_callback callback,
    void* user_data
);

#ifdef __cplusplus
}
#endif
