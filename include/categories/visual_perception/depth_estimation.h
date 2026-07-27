#pragma once

#include <stddef.h>
#include <stdint.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef _WIN32
#  if defined(DEPTH_BUILD_SHARED)
#    define DEPTH_API __declspec(dllexport)
#  elif defined(DEPTH_USE_SHARED)
#    define DEPTH_API __declspec(dllimport)
#  else
#    define DEPTH_API
#  endif
#else
#  define DEPTH_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct depth_runtime* depth_runtime_ptr;
typedef struct depth_model* depth_model_ptr;
typedef struct depth_session* depth_session_ptr;

// One result contract — a per-pixel depth/disparity map. Tasks differ only in
// how many input views they consume.
enum depth_task {
    DEPTH_TASK_MONOCULAR = 0, // single image
    DEPTH_TASK_STEREO = 1,    // rectified left/right pair
};

// What the values in a depth_map mean.
enum depth_map_kind {
    DEPTH_MAP_RELATIVE = 0,  // affine-invariant: larger = closer, no unit
    DEPTH_MAP_METRIC = 1,    // meters
    DEPTH_MAP_DISPARITY = 2, // pixels between the rectified pair (stereo)
};

struct depth_runtime_params {
    const char* device; // "auto", "cpu", or an exact GGML backend device name.
    uint32_t n_threads;
};

// Interleaved 8-bit image, rows top-to-bottom; channels is 1 (gray), 3 (RGB),
// or 4 (RGBA, alpha ignored). Pixel data is borrowed for the duration of the
// call.
struct depth_image {
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    const uint8_t* data;
};

struct depth_request_params {
    enum depth_task task;
};

struct depth_capabilities {
    bool monocular;
    bool stereo;
    bool metric; // model outputs meters rather than relative depth
};

// Owned by the library; release with depth_free_map. Row-major, same
// resolution as the input image(s); data[y * width + x].
struct depth_map {
    uint32_t width;
    uint32_t height;
    enum depth_map_kind kind;
    float* data;
};

// Rectified stereo calibration. focal_length_px uses the horizontal focal
// length of the rectified images; baseline_m is the camera-center distance.
struct depth_stereo_calibration {
    float focal_length_px;
    float baseline_m;
};

DEPTH_API struct depth_runtime_params depth_runtime_default_params(void);
DEPTH_API depth_runtime_ptr depth_runtime_create(struct depth_runtime_params params);
DEPTH_API void depth_runtime_free(depth_runtime_ptr runtime);
DEPTH_API const char* depth_runtime_get_device(depth_runtime_ptr runtime);
DEPTH_API uint32_t depth_runtime_get_thread_count(depth_runtime_ptr runtime);

// path is a GGUF file; the provider is selected from its
// `general.architecture` metadata.
DEPTH_API depth_model_ptr depth_load_model(depth_runtime_ptr runtime, const char* path);
DEPTH_API void depth_free_model(depth_model_ptr model);
DEPTH_API const char* depth_model_get_provider(depth_model_ptr model);
DEPTH_API struct depth_capabilities depth_model_get_capabilities(depth_model_ptr model);

DEPTH_API depth_session_ptr depth_create_session(depth_model_ptr model);
DEPTH_API void depth_free_session(depth_session_ptr session);

DEPTH_API struct depth_request_params depth_request_default_params(void);
// Synchronous; fills *map on success. right must be null for MONOCULAR and a
// rectified pair partner of image for STEREO (same dimensions). The requested
// task must be advertised by the model's capabilities.
DEPTH_API bool depth_estimate(
    depth_session_ptr session,
    const struct depth_image* image,
    const struct depth_image* right,
    struct depth_request_params params,
    struct depth_map* map
);
// Convert disparity pixels to metric depth using Z = focal_length_px *
// baseline_m / disparity. Non-positive or non-finite disparities map to 0.
// `metric` owns a new buffer and must be released with depth_free_map.
DEPTH_API bool depth_disparity_to_metric(
    const struct depth_map* disparity,
    struct depth_stereo_calibration calibration,
    struct depth_map* metric
);
DEPTH_API void depth_free_map(struct depth_map* map);

#ifdef __cplusplus
}
#endif
