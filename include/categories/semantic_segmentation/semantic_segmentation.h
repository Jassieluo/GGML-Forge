#pragma once

#include <stddef.h>
#include <stdint.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef _WIN32
#  if defined(SEGMENTATION_BUILD_SHARED)
#    define SEGMENTATION_API __declspec(dllexport)
#  elif defined(SEGMENTATION_USE_SHARED)
#    define SEGMENTATION_API __declspec(dllimport)
#  else
#    define SEGMENTATION_API
#  endif
#else
#  define SEGMENTATION_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct segmentation_runtime* segmentation_runtime_ptr;
typedef struct segmentation_model* segmentation_model_ptr;
typedef struct segmentation_session* segmentation_session_ptr;

struct segmentation_runtime_params {
    const char* device; // "auto", "cpu", or an exact GGML backend device name.
    uint32_t n_threads;
};

// Interleaved 8-bit image, rows top-to-bottom; channels is 1 (gray), 3 (RGB),
// or 4 (RGBA, alpha ignored). Pixel data is borrowed for the duration of the
// call.
struct segmentation_image {
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    const uint8_t* data;
};

struct segmentation_request_params {
    // Also fill segmentation_result.confidence (per-pixel probability of the
    // winning class). Rejected when the model cannot provide it.
    bool want_confidence;
};

struct segmentation_capabilities {
    uint32_t class_count;
    bool confidence; // can report per-pixel confidence
};

// Owned by the library; release with segmentation_free_result. Row-major,
// same resolution as the input image; index [y * width + x].
struct segmentation_result {
    uint32_t width;
    uint32_t height;
    int32_t* class_map;
    float* confidence; // null unless requested; [0, 1] per pixel
};

SEGMENTATION_API struct segmentation_runtime_params segmentation_runtime_default_params(void);
SEGMENTATION_API segmentation_runtime_ptr segmentation_runtime_create(struct segmentation_runtime_params params);
SEGMENTATION_API void segmentation_runtime_free(segmentation_runtime_ptr runtime);
SEGMENTATION_API const char* segmentation_runtime_get_device(segmentation_runtime_ptr runtime);
SEGMENTATION_API uint32_t segmentation_runtime_get_thread_count(segmentation_runtime_ptr runtime);

// path is a GGUF file; the provider is selected from its
// `general.architecture` metadata.
SEGMENTATION_API segmentation_model_ptr segmentation_load_model(segmentation_runtime_ptr runtime, const char* path);
SEGMENTATION_API void segmentation_free_model(segmentation_model_ptr model);
SEGMENTATION_API const char* segmentation_model_get_provider(segmentation_model_ptr model);
SEGMENTATION_API struct segmentation_capabilities segmentation_model_get_capabilities(segmentation_model_ptr model);
// Class label for [0, class_count); null when out of range or unnamed.
// Borrowed; valid until the model is freed.
SEGMENTATION_API const char* segmentation_model_get_label(segmentation_model_ptr model, int32_t class_id);
// Display color for a class as rgb[3]; false when out of range or the model
// ships no palette (callers should fall back to a generated palette).
SEGMENTATION_API bool segmentation_model_get_color(segmentation_model_ptr model, int32_t class_id, uint8_t rgb[3]);

SEGMENTATION_API segmentation_session_ptr segmentation_create_session(segmentation_model_ptr model);
SEGMENTATION_API void segmentation_free_session(segmentation_session_ptr session);

SEGMENTATION_API struct segmentation_request_params segmentation_request_default_params(void);
// Synchronous; fills *result on success.
SEGMENTATION_API bool segmentation_segment(
    segmentation_session_ptr session,
    const struct segmentation_image* image,
    struct segmentation_request_params params,
    struct segmentation_result* result
);
SEGMENTATION_API void segmentation_free_result(struct segmentation_result* result);

#ifdef __cplusplus
}
#endif
