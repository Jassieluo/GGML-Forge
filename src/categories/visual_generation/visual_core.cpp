#include "categories/visual_generation/visual_generation.h"
#include "visual_internal.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <thread>

namespace {

std::string copy_string(const char* value) { return value ? value : ""; }

bool copy_images(
    const std::vector<visual::ImageBuffer>& source,
    visual_image** output,
    size_t* count
) {
    if (!output || !count) return false;
    *output = nullptr;
    *count = 0;
    if (source.empty()) return true;
    auto images = std::unique_ptr<visual_image[]>(new (std::nothrow) visual_image[source.size()]{});
    if (!images) return false;
    for (size_t i = 0; i < source.size(); ++i) {
        const auto& item = source[i];
        images[i] = {item.width, item.height, item.channels, nullptr};
        if (!item.data.empty()) {
            images[i].data = new (std::nothrow) uint8_t[item.data.size()];
            if (!images[i].data) {
                for (size_t j = 0; j < i; ++j) delete[] images[j].data;
                return false;
            }
            std::memcpy(images[i].data, item.data.data(), item.data.size());
        }
    }
    *count = source.size();
    *output = images.release();
    return true;
}

visual::ModelConfig make_model_config(const visual_model_params& input) {
    visual::ModelConfig result;
#define VISUAL_COPY_FIELD(name) result.name = copy_string(input.name)
    VISUAL_COPY_FIELD(model);
    VISUAL_COPY_FIELD(diffusion_model);
    VISUAL_COPY_FIELD(high_noise_diffusion_model);
    VISUAL_COPY_FIELD(unconditioned_diffusion_model);
    VISUAL_COPY_FIELD(clip_l);
    VISUAL_COPY_FIELD(clip_g);
    VISUAL_COPY_FIELD(clip_vision);
    VISUAL_COPY_FIELD(t5xxl);
    VISUAL_COPY_FIELD(llm);
    VISUAL_COPY_FIELD(llm_vision);
    VISUAL_COPY_FIELD(vae);
    VISUAL_COPY_FIELD(audio_vae);
    VISUAL_COPY_FIELD(taesd);
    VISUAL_COPY_FIELD(control_net);
    VISUAL_COPY_FIELD(motion_module);
    VISUAL_COPY_FIELD(photo_maker);
    VISUAL_COPY_FIELD(pulid);
    VISUAL_COPY_FIELD(upscaler);
    VISUAL_COPY_FIELD(adetailer);
    VISUAL_COPY_FIELD(weight_type);
#undef VISUAL_COPY_FIELD
    return result;
}

} // namespace

visual_runtime_params visual_runtime_default_params(void) {
    const auto threads = std::thread::hardware_concurrency();
    return {"auto", "", "", threads ? threads : 4u, false, false, true};
}

visual_runtime_ptr visual_runtime_create(visual_runtime_params params) {
    auto runtime = std::make_unique<visual_runtime>();
    runtime->config.backend = copy_string(params.backend);
    if (runtime->config.backend.empty()) runtime->config.backend = "auto";
    runtime->config.params_backend = copy_string(params.params_backend);
    runtime->config.max_vram = copy_string(params.max_vram);
    runtime->config.n_threads = std::max<uint32_t>(1, params.n_threads);
    runtime->config.stream_layers = params.stream_layers;
    runtime->config.eager_load = params.eager_load;
    runtime->config.flash_attention = params.flash_attention;
    return runtime.release();
}

void visual_runtime_free(visual_runtime_ptr runtime) { delete runtime; }

visual_model_params visual_model_default_params(void) {
    visual_model_params params{};
    params.weight_type = "f16";
    return params;
}

visual_model_ptr visual_load_model(visual_runtime_ptr runtime, const visual_model_params* params) {
    if (!runtime || !params) return nullptr;
    auto provider = visual::ProviderRegistry::get().create("stable-diffusion.cpp");
    if (!provider) return nullptr;
    auto implementation = provider->load(make_model_config(*params), runtime->config);
    if (!implementation) return nullptr;
    auto model = std::make_unique<visual_model>();
    model->provider_name = provider->name();
    model->implementation = std::move(implementation);
    return model.release();
}

void visual_free_model(visual_model_ptr model) { delete model; }

const char* visual_model_get_provider(visual_model_ptr model) {
    return model ? model->provider_name.c_str() : nullptr;
}

visual_capabilities visual_model_get_capabilities(visual_model_ptr model) {
    return model && model->implementation ? model->implementation->capabilities() : visual_capabilities{};
}

visual_session_ptr visual_create_session(visual_model_ptr model) {
    if (!model || !model->implementation) return nullptr;
    auto session = std::make_unique<visual_session>();
    session->model = model->implementation;
    session->implementation = session->model->create_session();
    return session->implementation ? session.release() : nullptr;
}

void visual_free_session(visual_session_ptr session) { delete session; }

void visual_session_set_callbacks(
    visual_session_ptr session,
    visual_progress_callback progress,
    visual_preview_callback preview,
    void* user_data
) {
    if (session && session->implementation) session->implementation->set_callbacks({progress, preview, user_data});
}

bool visual_session_cancel(visual_session_ptr session, visual_cancel_mode mode) {
    return session && session->implementation && session->implementation->cancel(mode);
}

visual_sample_params visual_sample_default_params(void) {
    return {"euler_a", "discrete", 20, 7.0f, 1.0f, 3.5f, 0.0f, 0.0f};
}

visual_image_request visual_image_request_default_params(void) {
    visual_image_request request{};
    request.width = 512;
    request.height = 512;
    request.seed = 42;
    request.batch_count = 1;
    request.clip_skip = -1;
    request.strength = 0.75f;
    request.control_strength = 0.9f;
    request.sample = visual_sample_default_params();
    request.hires.scale = 2.0f;
    request.hires.denoising_strength = 0.7f;
    request.identity_style_strength = 1.0f;
    request.pulid_weight = 1.0f;
    request.vae_tiling.overlap = 0.5f;
    request.cache.mode = "disabled";
    return request;
}

visual_video_request visual_video_request_default_params(void) {
    visual_video_request request{};
    request.width = 512;
    request.height = 512;
    request.seed = 42;
    request.frame_count = 33;
    request.fps = 16;
    request.clip_skip = -1;
    request.strength = 0.75f;
    request.sample = visual_sample_default_params();
    request.high_noise_sample = visual_sample_default_params();
    request.hires.scale = 2.0f;
    request.hires.denoising_strength = 0.7f;
    request.vae_tiling.overlap = 0.5f;
    request.cache.mode = "disabled";
    return request;
}

bool visual_generate_images(
    visual_session_ptr session,
    const visual_image_request* request,
    visual_image** images,
    size_t* image_count
) {
    if (!session || !session->implementation || !request || !images || !image_count) return false;
    std::vector<visual::ImageBuffer> result;
    return session->implementation->generate_images(*request, result) && copy_images(result, images, image_count);
}

bool visual_generate_video(
    visual_session_ptr session,
    const visual_video_request* request,
    visual_video* video
) {
    if (!session || !session->implementation || !request || !video) return false;
    *video = {};
    visual::VideoBuffer result;
    if (!session->implementation->generate_video(*request, result)) return false;
    if (!copy_images(result.frames, &video->frames, &video->frame_count)) return false;
    video->fps = result.fps;
    video->audio.sample_rate = result.audio.sample_rate;
    video->audio.channels = result.audio.channels;
    video->audio.sample_count = result.audio.channels
        ? result.audio.data.size() / result.audio.channels
        : 0;
    if (!result.audio.data.empty()) {
        video->audio.data = new (std::nothrow) float[result.audio.data.size()];
        if (!video->audio.data) {
            visual_free_video(video);
            return false;
        }
        std::copy(result.audio.data.begin(), result.audio.data.end(), video->audio.data);
    }
    return true;
}

bool visual_upscale(
    visual_session_ptr session,
    const visual_image* input,
    uint32_t factor,
    visual_image** images,
    size_t* image_count
) {
    if (!session || !session->implementation || !input || !images || !image_count) return false;
    std::vector<visual::ImageBuffer> result;
    return session->implementation->upscale(*input, factor, result) && copy_images(result, images, image_count);
}

bool visual_adetail(
    visual_session_ptr session,
    const visual_image* input,
    const char* prompt,
    const char* negative_prompt,
    const visual_image_request* request,
    visual_image** images,
    size_t* image_count
) {
    if (!session || !session->implementation || !input || !request || !images || !image_count) return false;
    std::vector<visual::ImageBuffer> result;
    return session->implementation->adetail(
        *input, prompt, negative_prompt, *request, result) && copy_images(result, images, image_count);
}

bool visual_preprocess_canny(visual_image* image, float high, float low, bool inverse) {
    return image && visual::preprocess_canny(*image, high, low, inverse);
}

void visual_free_images(visual_image* images, size_t image_count) {
    if (!images) return;
    for (size_t i = 0; i < image_count; ++i) delete[] images[i].data;
    delete[] images;
}

void visual_free_video(visual_video* video) {
    if (!video) return;
    visual_free_images(video->frames, video->frame_count);
    delete[] video->audio.data;
    *video = {};
}
