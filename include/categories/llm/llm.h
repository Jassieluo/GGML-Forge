#pragma once

#include <stddef.h>
#include <stdint.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef _WIN32
#  if defined(LLM_BUILD_SHARED)
#    define LLM_API __declspec(dllexport)
#  elif defined(LLM_USE_SHARED)
#    define LLM_API __declspec(dllimport)
#  else
#    define LLM_API
#  endif
#else
#  define LLM_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct llm_runtime* llm_runtime_ptr;
typedef struct llm_model* llm_model_ptr;
typedef struct llm_session* llm_session_ptr;

typedef bool (*llm_text_callback)(const char* utf8, size_t length, void* user_data);

struct llm_runtime_params {
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_threads;
    int32_t n_gpu_layers;
};

struct llm_generation_params {
    int32_t max_tokens;
    float temperature;
    int32_t top_k;
    float top_p;
    uint32_t seed;
};

LLM_API struct llm_runtime_params llm_runtime_default_params(void);
LLM_API llm_runtime_ptr llm_runtime_create(struct llm_runtime_params params);
LLM_API void llm_runtime_free(llm_runtime_ptr runtime);

LLM_API llm_model_ptr llm_load_model(llm_runtime_ptr runtime, const char* path);
LLM_API void llm_free_model(llm_model_ptr model);
LLM_API const char* llm_model_get_provider(llm_model_ptr model);

LLM_API llm_session_ptr llm_create_session(llm_model_ptr model);
LLM_API void llm_free_session(llm_session_ptr session);
LLM_API bool llm_session_reset(llm_session_ptr session);

LLM_API struct llm_generation_params llm_generation_default_params(void);
LLM_API bool llm_generate(
    llm_session_ptr session,
    const char* prompt,
    struct llm_generation_params params,
    llm_text_callback callback,
    void* user_data
);

#ifdef __cplusplus
}
#endif
