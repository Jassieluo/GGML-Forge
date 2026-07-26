#pragma once

#include <stddef.h>
#include <stdint.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef _WIN32
#  if defined(DETECTION_BUILD_SHARED)
#    define DETECTION_API __declspec(dllexport)
#  elif defined(DETECTION_USE_SHARED)
#    define DETECTION_API __declspec(dllimport)
#  else
#    define DETECTION_API
#  endif
#else
#  define DETECTION_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct detection_runtime* detection_runtime_ptr;
typedef struct detection_model* detection_model_ptr;
typedef struct detection_session* detection_session_ptr;

// One category, one result contract: every task returns a list of instances.
// Tasks only differ in which optional attachments each instance carries.
enum detection_task {
    DETECTION_TASK_BOXES = 0,          // axis-aligned boxes
    DETECTION_TASK_ORIENTED_BOXES = 1, // adds a rotation angle per instance
    DETECTION_TASK_INSTANCE_MASKS = 2, // adds a per-instance mask
    DETECTION_TASK_KEYPOINTS = 3,      // adds keypoints per instance (pose)
};

struct detection_runtime_params {
    const char* device; // "auto", "cpu", or an exact GGML backend device name.
    uint32_t n_threads;
};

// Interleaved 8-bit image, rows top-to-bottom; channels is 1 (gray), 3 (RGB),
// or 4 (RGBA, alpha ignored). Pixel data is borrowed for the duration of the
// call.
struct detection_image {
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    const uint8_t* data;
};

struct detection_request_params {
    enum detection_task task;
    float score_threshold; // drop instances below this confidence, default 0.25
    float iou_threshold;   // NMS overlap threshold, default 0.45
    int32_t max_instances; // cap on returned instances, default 300
};

struct detection_capabilities {
    bool boxes;
    bool oriented_boxes;
    bool instance_masks;
    bool keypoints;
    uint32_t class_count;
    uint32_t keypoint_count; // keypoints per instance when supported, else 0
};

struct detection_keypoint {
    float x; // input-image pixel coordinates
    float y;
    float score; // [0, 1], or -1 when the model has no per-point confidence
};

// All coordinates are in input-image pixel space (letterboxing and model
// input scaling are undone by the provider before results are returned).
struct detection_instance {
    float x; // box center
    float y;
    float width;
    float height;
    float angle; // radians, counter-clockwise; 0 unless task is ORIENTED_BOXES
    int32_t class_id;
    float score; // [0, 1]
    // INSTANCE_MASKS only: mask_width x mask_height, 0 or 255 per pixel,
    // covering the axis-aligned box region (resize to box size to overlay).
    const uint8_t* mask;
    uint32_t mask_width;
    uint32_t mask_height;
    // KEYPOINTS only.
    const struct detection_keypoint* keypoints;
    size_t keypoint_count;
};

// Owned by the library, including mask and keypoint storage; release with
// detection_free_result.
struct detection_result {
    struct detection_instance* instances;
    size_t instance_count;
};

DETECTION_API struct detection_runtime_params detection_runtime_default_params(void);
DETECTION_API detection_runtime_ptr detection_runtime_create(struct detection_runtime_params params);
DETECTION_API void detection_runtime_free(detection_runtime_ptr runtime);
DETECTION_API const char* detection_runtime_get_device(detection_runtime_ptr runtime);
DETECTION_API uint32_t detection_runtime_get_thread_count(detection_runtime_ptr runtime);

// path is a GGUF file; the provider is selected from its
// `general.architecture` metadata.
DETECTION_API detection_model_ptr detection_load_model(detection_runtime_ptr runtime, const char* path);
DETECTION_API void detection_free_model(detection_model_ptr model);
DETECTION_API const char* detection_model_get_provider(detection_model_ptr model);
DETECTION_API struct detection_capabilities detection_model_get_capabilities(detection_model_ptr model);
// Class label for [0, class_count); null when out of range or unnamed.
// Borrowed; valid until the model is freed.
DETECTION_API const char* detection_model_get_label(detection_model_ptr model, int32_t class_id);

DETECTION_API detection_session_ptr detection_create_session(detection_model_ptr model);
DETECTION_API void detection_free_session(detection_session_ptr session);

DETECTION_API struct detection_request_params detection_request_default_params(void);
// Synchronous; fills *result on success. The requested task must be
// advertised by the model's capabilities.
DETECTION_API bool detection_detect(
    detection_session_ptr session,
    const struct detection_image* image,
    struct detection_request_params params,
    struct detection_result* result
);
DETECTION_API void detection_free_result(struct detection_result* result);

#ifdef __cplusplus
}
#endif
