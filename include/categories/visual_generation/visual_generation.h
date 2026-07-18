#pragma once

#include <stddef.h>
#include <stdint.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef _WIN32
#  if defined(VISUAL_BUILD_SHARED)
#    define VISUAL_API __declspec(dllexport)
#  elif defined(VISUAL_USE_SHARED)
#    define VISUAL_API __declspec(dllimport)
#  else
#    define VISUAL_API
#  endif
#else
#  define VISUAL_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct visual_runtime* visual_runtime_ptr;
typedef struct visual_model* visual_model_ptr;
typedef struct visual_session* visual_session_ptr;

struct visual_image {
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint8_t* data;
};

struct visual_audio {
    uint32_t sample_rate;
    uint32_t channels;
    uint64_t sample_count;
    float* data;
};

struct visual_video {
    struct visual_image* frames;
    size_t frame_count;
    uint32_t fps;
    struct visual_audio audio;
};

struct visual_runtime_params {
    const char* backend;
    const char* params_backend;
    const char* max_vram;
    uint32_t n_threads;
    bool stream_layers;
    bool eager_load;
    bool flash_attention;
};

struct visual_model_params {
    const char* model;
    const char* diffusion_model;
    const char* high_noise_diffusion_model;
    const char* unconditioned_diffusion_model;
    const char* clip_l;
    const char* clip_g;
    const char* clip_vision;
    const char* t5xxl;
    const char* llm;
    const char* llm_vision;
    const char* vae;
    const char* audio_vae;
    const char* taesd;
    const char* control_net;
    const char* motion_module;
    const char* photo_maker;
    const char* pulid;
    const char* upscaler;
    const char* adetailer;
    const char* weight_type;
};

struct visual_capabilities {
    bool text_to_image;
    bool image_to_image;
    bool inpainting;
    bool control_image;
    bool reference_images;
    bool lora;
    bool video;
    bool video_audio;
    bool upscale;
    bool adetailer;
    bool preview;
    bool cancellation;
};

struct visual_lora {
    const char* path;
    float strength;
    bool high_noise;
};

struct visual_sample_params {
    const char* sampler;
    const char* scheduler;
    int32_t steps;
    float text_guidance;
    float image_guidance;
    float distilled_guidance;
    float eta;
    float flow_shift;
};

struct visual_hires_params {
    bool enabled;
    const char* upscaler;
    float scale;
    int32_t target_width;
    int32_t target_height;
    int32_t steps;
    float denoising_strength;
    int32_t tile_size;
};

struct visual_tiling_params {
    bool enabled;
    bool temporal;
    int32_t tile_width;
    int32_t tile_height;
    float overlap;
};

struct visual_cache_params {
    const char* mode;
    float reuse_threshold;
    float start_percent;
    float end_percent;
};

struct visual_image_request {
    const char* prompt;
    const char* negative_prompt;
    int32_t width;
    int32_t height;
    int64_t seed;
    int32_t batch_count;
    int32_t clip_skip;
    float strength;
    struct visual_sample_params sample;
    struct visual_image init_image;
    struct visual_image mask_image;
    struct visual_image control_image;
    float control_strength;
    const struct visual_image* reference_images;
    size_t reference_image_count;
    const char* reference_image_args;
    const struct visual_lora* loras;
    size_t lora_count;
    const struct visual_image* identity_images;
    size_t identity_image_count;
    const char* identity_embedding;
    float identity_style_strength;
    const char* pulid_embedding;
    float pulid_weight;
    struct visual_tiling_params vae_tiling;
    struct visual_cache_params cache;
    struct visual_hires_params hires;
    int32_t qwen_image_layers;
    bool circular_x;
    bool circular_y;
};

struct visual_video_request {
    const char* prompt;
    const char* negative_prompt;
    int32_t width;
    int32_t height;
    int64_t seed;
    int32_t frame_count;
    int32_t fps;
    int32_t clip_skip;
    float strength;
    float vace_strength;
    float moe_boundary;
    struct visual_sample_params sample;
    struct visual_sample_params high_noise_sample;
    struct visual_image init_image;
    struct visual_image end_image;
    const struct visual_image* control_frames;
    size_t control_frame_count;
    const struct visual_lora* loras;
    size_t lora_count;
    struct visual_tiling_params vae_tiling;
    struct visual_cache_params cache;
    struct visual_hires_params hires;
    bool circular_x;
    bool circular_y;
};

enum visual_cancel_mode {
    VISUAL_CANCEL_ALL = 0,
    VISUAL_CANCEL_NEW_BATCH_ITEMS = 1,
    VISUAL_CANCEL_RESET = 2,
};

typedef void (*visual_progress_callback)(int32_t step, int32_t steps, float seconds, void* user_data);
// Preview images are borrowed and valid only during the callback.
typedef void (*visual_preview_callback)(
    int32_t step,
    const struct visual_image* frames,
    size_t frame_count,
    bool noisy,
    void* user_data);

VISUAL_API struct visual_runtime_params visual_runtime_default_params(void);
VISUAL_API visual_runtime_ptr visual_runtime_create(struct visual_runtime_params params);
VISUAL_API void visual_runtime_free(visual_runtime_ptr runtime);

VISUAL_API struct visual_model_params visual_model_default_params(void);
VISUAL_API visual_model_ptr visual_load_model(visual_runtime_ptr runtime, const struct visual_model_params* params);
VISUAL_API void visual_free_model(visual_model_ptr model);
VISUAL_API const char* visual_model_get_provider(visual_model_ptr model);
VISUAL_API struct visual_capabilities visual_model_get_capabilities(visual_model_ptr model);

VISUAL_API visual_session_ptr visual_create_session(visual_model_ptr model);
VISUAL_API void visual_free_session(visual_session_ptr session);
VISUAL_API void visual_session_set_callbacks(
    visual_session_ptr session,
    visual_progress_callback progress,
    visual_preview_callback preview,
    void* user_data);
VISUAL_API bool visual_session_cancel(visual_session_ptr session, enum visual_cancel_mode mode);

VISUAL_API struct visual_sample_params visual_sample_default_params(void);
VISUAL_API struct visual_image_request visual_image_request_default_params(void);
VISUAL_API struct visual_video_request visual_video_request_default_params(void);

VISUAL_API bool visual_generate_images(
    visual_session_ptr session,
    const struct visual_image_request* request,
    struct visual_image** images,
    size_t* image_count);
VISUAL_API bool visual_generate_video(
    visual_session_ptr session,
    const struct visual_video_request* request,
    struct visual_video* video);
VISUAL_API bool visual_upscale(
    visual_session_ptr session,
    const struct visual_image* input,
    uint32_t factor,
    struct visual_image** images,
    size_t* image_count);
VISUAL_API bool visual_adetail(
    visual_session_ptr session,
    const struct visual_image* input,
    const char* prompt,
    const char* negative_prompt,
    const struct visual_image_request* inpaint_request,
    struct visual_image** images,
    size_t* image_count);
VISUAL_API bool visual_preprocess_canny(
    struct visual_image* image,
    float high_threshold,
    float low_threshold,
    bool inverse);

VISUAL_API void visual_free_images(struct visual_image* images, size_t image_count);
VISUAL_API void visual_free_video(struct visual_video* video);

#ifdef __cplusplus
}
#endif
