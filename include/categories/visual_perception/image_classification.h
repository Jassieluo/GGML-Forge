#pragma once

#include <stddef.h>
#include <stdint.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef _WIN32
#  if defined(CLASSIFICATION_BUILD_SHARED)
#    define CLASSIFICATION_API __declspec(dllexport)
#  elif defined(CLASSIFICATION_USE_SHARED)
#    define CLASSIFICATION_API __declspec(dllimport)
#  else
#    define CLASSIFICATION_API
#  endif
#else
#  define CLASSIFICATION_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct classification_runtime* classification_runtime_ptr;
typedef struct classification_model* classification_model_ptr;
typedef struct classification_session* classification_session_ptr;

struct classification_runtime_params {
    const char* device;
    uint32_t n_threads;
};

struct classification_image {
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    const uint8_t* data;
};

struct classification_request_params {
    uint32_t top_k; // default 5, clamped to the model class count
};

struct classification_capabilities {
    uint32_t class_count;
};

struct classification_score {
    int32_t class_id;
    float score;
};

struct classification_result {
    struct classification_score* scores;
    size_t score_count;
};

CLASSIFICATION_API struct classification_runtime_params classification_runtime_default_params(void);
CLASSIFICATION_API classification_runtime_ptr classification_runtime_create(
    struct classification_runtime_params params);
CLASSIFICATION_API void classification_runtime_free(classification_runtime_ptr runtime);
CLASSIFICATION_API const char* classification_runtime_get_device(classification_runtime_ptr runtime);
CLASSIFICATION_API uint32_t classification_runtime_get_thread_count(classification_runtime_ptr runtime);

CLASSIFICATION_API classification_model_ptr classification_load_model(
    classification_runtime_ptr runtime, const char* path);
CLASSIFICATION_API void classification_free_model(classification_model_ptr model);
CLASSIFICATION_API const char* classification_model_get_provider(classification_model_ptr model);
CLASSIFICATION_API struct classification_capabilities classification_model_get_capabilities(
    classification_model_ptr model);
CLASSIFICATION_API const char* classification_model_get_label(
    classification_model_ptr model, int32_t class_id);

CLASSIFICATION_API classification_session_ptr classification_create_session(
    classification_model_ptr model);
CLASSIFICATION_API void classification_free_session(classification_session_ptr session);

CLASSIFICATION_API struct classification_request_params classification_request_default_params(void);
CLASSIFICATION_API bool classification_classify(
    classification_session_ptr session,
    const struct classification_image* image,
    struct classification_request_params params,
    struct classification_result* result);
CLASSIFICATION_API void classification_free_result(struct classification_result* result);

#ifdef __cplusplus
}
#endif
