#pragma once

#include <stddef.h>
#include <stdint.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef _WIN32
#  if defined(INSTANCE_PERCEPTION_BUILD_SHARED)
#    define INSTANCE_PERCEPTION_API __declspec(dllexport)
#  elif defined(INSTANCE_PERCEPTION_USE_SHARED)
#    define INSTANCE_PERCEPTION_API __declspec(dllimport)
#  else
#    define INSTANCE_PERCEPTION_API
#  endif
#else
#  define INSTANCE_PERCEPTION_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct instance_runtime* instance_runtime_ptr;
typedef struct instance_model* instance_model_ptr;
typedef struct instance_session* instance_session_ptr;

// One category, one result contract: every task returns a list of instances.
// Tasks only differ in which optional attachments each instance carries.
enum instance_task {
    INSTANCE_TASK_BOXES = 0,          // axis-aligned boxes
    INSTANCE_TASK_ORIENTED_BOXES = 1, // adds a rotation angle per instance
    INSTANCE_TASK_MASKS = 2,          // adds a per-instance mask
    INSTANCE_TASK_KEYPOINTS = 3,      // adds keypoints per instance (pose)
};

struct instance_runtime_params {
    const char* device; // "auto", "cpu", or an exact GGML backend device name.
    uint32_t n_threads;
};

// Interleaved 8-bit image, rows top-to-bottom; channels is 1 (gray), 3 (RGB),
// or 4 (RGBA, alpha ignored). Pixel data is borrowed for the duration of the
// call.
struct instance_image {
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    const uint8_t* data;
};

struct instance_request_params {
    enum instance_task task;
    float score_threshold; // drop instances below this confidence, default 0.25
    float iou_threshold;   // NMS overlap threshold, default 0.45
    int32_t max_instances; // cap on returned instances, default 300
};

struct instance_capabilities {
    bool boxes;
    bool oriented_boxes;
    bool instance_masks;
    bool keypoints;
    uint32_t class_count;
    uint32_t keypoint_count; // keypoints per instance when supported, else 0
};

struct instance_keypoint {
    float x; // input-image pixel coordinates
    float y;
    float score; // [0, 1], or -1 when the model has no per-point confidence
};

// All coordinates are in input-image pixel space (letterboxing and model
// input scaling are undone by the provider before results are returned).
struct perceived_instance {
    float x; // box center
    float y;
    float width;
    float height;
    // Radians in image coordinates; positive rotates the width axis toward
    // +y. Zero unless task is ORIENTED_BOXES.
    float angle;
    int32_t class_id;
    float score; // [0, 1]
    // INSTANCE_TASK_MASKS only: mask_width x mask_height, 0 or 255 per pixel,
    // covering the axis-aligned box region (resize to box size to overlay).
    const uint8_t* mask;
    uint32_t mask_width;
    uint32_t mask_height;
    // KEYPOINTS only.
    const struct instance_keypoint* keypoints;
    size_t keypoint_count;
};

// Owned by the library, including mask and keypoint storage; release with
// instance_free_result.
struct instance_result {
    struct perceived_instance* instances;
    size_t instance_count;
};

INSTANCE_PERCEPTION_API struct instance_runtime_params instance_runtime_default_params(void);
INSTANCE_PERCEPTION_API instance_runtime_ptr instance_runtime_create(struct instance_runtime_params params);
INSTANCE_PERCEPTION_API void instance_runtime_free(instance_runtime_ptr runtime);
INSTANCE_PERCEPTION_API const char* instance_runtime_get_device(instance_runtime_ptr runtime);
INSTANCE_PERCEPTION_API uint32_t instance_runtime_get_thread_count(instance_runtime_ptr runtime);

// path is a GGUF file; the provider is selected from its
// `general.architecture` metadata.
INSTANCE_PERCEPTION_API instance_model_ptr instance_load_model(instance_runtime_ptr runtime, const char* path);
INSTANCE_PERCEPTION_API void instance_free_model(instance_model_ptr model);
INSTANCE_PERCEPTION_API const char* instance_model_get_provider(instance_model_ptr model);
INSTANCE_PERCEPTION_API struct instance_capabilities instance_model_get_capabilities(instance_model_ptr model);
// Class label for [0, class_count); null when out of range or unnamed.
// Borrowed; valid until the model is freed.
INSTANCE_PERCEPTION_API const char* instance_model_get_label(instance_model_ptr model, int32_t class_id);

INSTANCE_PERCEPTION_API instance_session_ptr instance_create_session(instance_model_ptr model);
INSTANCE_PERCEPTION_API void instance_free_session(instance_session_ptr session);

INSTANCE_PERCEPTION_API struct instance_request_params instance_request_default_params(void);
// Synchronous; fills *result on success. The requested task must be
// advertised by the model's capabilities.
INSTANCE_PERCEPTION_API bool instance_perceive(
    instance_session_ptr session,
    const struct instance_image* image,
    struct instance_request_params params,
    struct instance_result* result
);
INSTANCE_PERCEPTION_API void instance_free_result(struct instance_result* result);

#ifdef __cplusplus
}
#endif
