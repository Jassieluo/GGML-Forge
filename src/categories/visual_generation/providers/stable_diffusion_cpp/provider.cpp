#include "providers/visual_provider.h"

#include "stable-diffusion.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace visual {
namespace {

const char* nullable(const std::string& value) { return value.empty() ? nullptr : value.c_str(); }

sd_image_t to_sd_image(const visual_image& image) {
    return {image.width, image.height, image.channels, image.data};
}

bool image_bytes(const sd_image_t& image, size_t& bytes) {
    if (!image.data || image.width == 0 || image.height == 0 || image.channel == 0) return false;
    const uint64_t count = static_cast<uint64_t>(image.width) * image.height * image.channel;
    if (count > std::numeric_limits<size_t>::max()) return false;
    bytes = static_cast<size_t>(count);
    return true;
}

bool copy_sd_images(sd_image_t* images, int count, std::vector<ImageBuffer>& output) {
    if (!images || count <= 0) return false;
    output.clear();
    output.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        size_t bytes = 0;
        if (!image_bytes(images[i], bytes)) return false;
        ImageBuffer item;
        item.width = images[i].width;
        item.height = images[i].height;
        item.channels = images[i].channel;
        item.data.assign(images[i].data, images[i].data + bytes);
        output.push_back(std::move(item));
    }
    return true;
}

sd_cache_mode_t cache_mode(const char* value) {
    if (!value || std::strcmp(value, "disabled") == 0) return SD_CACHE_DISABLED;
    if (std::strcmp(value, "easycache") == 0) return SD_CACHE_EASYCACHE;
    if (std::strcmp(value, "ucache") == 0) return SD_CACHE_UCACHE;
    if (std::strcmp(value, "dbcache") == 0) return SD_CACHE_DBCACHE;
    if (std::strcmp(value, "taylorseer") == 0) return SD_CACHE_TAYLORSEER;
    if (std::strcmp(value, "cache_dit") == 0) return SD_CACHE_CACHE_DIT;
    if (std::strcmp(value, "spectrum") == 0) return SD_CACHE_SPECTRUM;
    return SD_CACHE_DISABLED;
}

void apply_sample(sd_ctx_t* context, const visual_sample_params& input, sd_sample_params_t& output) {
    sd_sample_params_init(&output);
    output.sample_steps = std::max(1, input.steps);
    output.guidance.txt_cfg = input.text_guidance;
    output.guidance.img_cfg = input.image_guidance;
    output.guidance.distilled_guidance = input.distilled_guidance;
    output.eta = input.eta;
    output.flow_shift = input.flow_shift;
    if (input.sampler && input.sampler[0]) {
        const sample_method_t method = str_to_sample_method(input.sampler);
        if (method != SAMPLE_METHOD_COUNT) output.sample_method = method;
    } else {
        output.sample_method = sd_get_default_sample_method(context);
    }
    if (input.scheduler && input.scheduler[0]) {
        const scheduler_t scheduler = str_to_scheduler(input.scheduler);
        if (scheduler != SCHEDULER_COUNT) output.scheduler = scheduler;
    } else {
        output.scheduler = sd_get_default_scheduler(context, output.sample_method);
    }
}

void apply_tiling(const visual_tiling_params& input, sd_tiling_params_t& output) {
    output.enabled = input.enabled;
    output.temporal_tiling = input.temporal;
    output.tile_size_x = input.tile_width;
    output.tile_size_y = input.tile_height;
    output.target_overlap = input.overlap;
}

void apply_cache(const visual_cache_params& input, sd_cache_params_t& output) {
    sd_cache_params_init(&output);
    output.mode = cache_mode(input.mode);
    output.reuse_threshold = input.reuse_threshold;
    output.start_percent = input.start_percent;
    output.end_percent = input.end_percent;
}

void apply_hires(const visual_hires_params& input, sd_hires_params_t& output) {
    sd_hires_params_init(&output);
    output.enabled = input.enabled;
    if (input.upscaler && input.upscaler[0]) {
        const sd_hires_upscaler_t method = str_to_sd_hires_upscaler(input.upscaler);
        if (method != SD_HIRES_UPSCALER_COUNT) output.upscaler = method;
    }
    output.scale = input.scale;
    output.target_width = input.target_width;
    output.target_height = input.target_height;
    output.steps = input.steps;
    output.denoising_strength = input.denoising_strength;
    output.upscale_tile_size = input.tile_size;
}

std::vector<sd_lora_t> make_loras(const visual_lora* input, size_t count) {
    std::vector<sd_lora_t> result;
    result.reserve(count);
    for (size_t i = 0; input && i < count; ++i) {
        result.push_back({input[i].high_noise, input[i].strength, input[i].path});
    }
    return result;
}

std::vector<sd_image_t> make_images(const visual_image* input, size_t count) {
    std::vector<sd_image_t> result;
    result.reserve(count);
    for (size_t i = 0; input && i < count; ++i) result.push_back(to_sd_image(input[i]));
    return result;
}

struct CallbackState {
    Callbacks callbacks;
};

void progress_bridge(int step, int steps, float seconds, void* data) {
    auto* state = static_cast<CallbackState*>(data);
    if (state && state->callbacks.progress) {
        state->callbacks.progress(step, steps, seconds, state->callbacks.user_data);
    }
}

void preview_bridge(int step, int frame_count, sd_image_t* frames, bool noisy, void* data) {
    auto* state = static_cast<CallbackState*>(data);
    if (!state || !state->callbacks.preview || !frames || frame_count <= 0) return;
    std::vector<visual_image> views;
    views.reserve(static_cast<size_t>(frame_count));
    for (int i = 0; i < frame_count; ++i) {
        views.push_back({frames[i].width, frames[i].height, frames[i].channel, frames[i].data});
    }
    state->callbacks.preview(step, views.data(), views.size(), noisy, state->callbacks.user_data);
}

std::mutex& callback_mutex() {
    static std::mutex mutex;
    return mutex;
}

class CallbackScope {
public:
    explicit CallbackScope(Callbacks callbacks)
        : lock_(callback_mutex()), state_{callbacks} {
        sd_set_progress_callback(callbacks.progress ? progress_bridge : nullptr, &state_);
        sd_set_preview_callback(
            callbacks.preview ? preview_bridge : nullptr,
            callbacks.preview ? PREVIEW_PROJ : PREVIEW_NONE,
            1, true, false, &state_);
    }

    ~CallbackScope() {
        sd_set_progress_callback(nullptr, nullptr);
        sd_set_preview_callback(nullptr, PREVIEW_NONE, 1, false, false, nullptr);
    }

private:
    std::unique_lock<std::mutex> lock_;
    CallbackState state_;
};

class StableDiffusionModel;

class StableDiffusionSession final : public IVisualSession {
public:
    explicit StableDiffusionSession(std::shared_ptr<StableDiffusionModel> model)
        : model_(std::move(model)) {}

    void set_callbacks(Callbacks callbacks) override { callbacks_ = callbacks; }
    bool cancel(visual_cancel_mode mode) override;
    bool generate_images(const visual_image_request& request, std::vector<ImageBuffer>& output) override;
    bool generate_video(const visual_video_request& request, VideoBuffer& output) override;
    bool upscale(const visual_image& input, uint32_t factor, std::vector<ImageBuffer>& output) override;
    bool adetail(
        const visual_image& input,
        const char* prompt,
        const char* negative_prompt,
        const visual_image_request& request,
        std::vector<ImageBuffer>& output) override;

private:
    sd_img_gen_params_t image_params(
        const visual_image_request& request,
        std::vector<sd_lora_t>& loras,
        std::vector<sd_image_t>& references,
        std::vector<sd_image_t>& identities) const;

    std::shared_ptr<StableDiffusionModel> model_;
    Callbacks callbacks_;
};

class StableDiffusionModel final : public IVisualModel, public std::enable_shared_from_this<StableDiffusionModel> {
public:
    StableDiffusionModel(sd_ctx_t* context, upscaler_ctx_t* upscaler, adetailer_ctx_t* adetailer)
        : context_(context), upscaler_(upscaler), adetailer_(adetailer) {}
    ~StableDiffusionModel() override {
        if (adetailer_) free_adetailer_ctx(adetailer_);
        if (upscaler_) free_upscaler_ctx(upscaler_);
        if (context_) free_sd_ctx(context_);
    }

    std::unique_ptr<IVisualSession> create_session() override {
        return std::make_unique<StableDiffusionSession>(shared_from_this());
    }

    visual_capabilities capabilities() const override {
        const bool images = context_ && sd_ctx_supports_image_generation(context_);
        const bool videos = context_ && sd_ctx_supports_video_generation(context_);
        return {
            images, images, images, images, images, images,
            videos, videos, upscaler_ != nullptr, adetailer_ != nullptr,
            context_ != nullptr, context_ != nullptr,
        };
    }

    sd_ctx_t* context() const { return context_; }
    upscaler_ctx_t* upscaler() const { return upscaler_; }
    adetailer_ctx_t* adetailer() const { return adetailer_; }
    std::mutex& mutex() { return mutex_; }

private:
    sd_ctx_t* context_ = nullptr;
    upscaler_ctx_t* upscaler_ = nullptr;
    adetailer_ctx_t* adetailer_ = nullptr;
    std::mutex mutex_;
};

sd_img_gen_params_t StableDiffusionSession::image_params(
    const visual_image_request& request,
    std::vector<sd_lora_t>& loras,
    std::vector<sd_image_t>& references,
    std::vector<sd_image_t>& identities
) const {
    sd_img_gen_params_t params;
    sd_img_gen_params_init(&params);
    loras = make_loras(request.loras, request.lora_count);
    references = make_images(request.reference_images, request.reference_image_count);
    identities = make_images(request.identity_images, request.identity_image_count);
    params.loras = loras.data();
    params.lora_count = static_cast<uint32_t>(loras.size());
    params.prompt = request.prompt;
    params.negative_prompt = request.negative_prompt;
    params.clip_skip = request.clip_skip;
    params.init_image = to_sd_image(request.init_image);
    params.mask_image = to_sd_image(request.mask_image);
    params.control_image = to_sd_image(request.control_image);
    params.control_strength = request.control_strength;
    params.ref_images = references.data();
    params.ref_images_count = static_cast<int>(references.size());
    params.ref_image_args = request.reference_image_args;
    params.pm_params.id_images = identities.data();
    params.pm_params.id_images_count = static_cast<int>(identities.size());
    params.pm_params.id_embed_path = request.identity_embedding;
    params.pm_params.style_strength = request.identity_style_strength;
    params.pulid_params.id_embedding_path = request.pulid_embedding;
    params.pulid_params.id_weight = request.pulid_weight;
    params.width = request.width;
    params.height = request.height;
    params.strength = request.strength;
    params.seed = request.seed;
    params.batch_count = request.batch_count;
    params.qwen_image_layers = request.qwen_image_layers;
    params.circular_x = request.circular_x;
    params.circular_y = request.circular_y;
    apply_sample(model_->context(), request.sample, params.sample_params);
    apply_tiling(request.vae_tiling, params.vae_tiling_params);
    apply_cache(request.cache, params.cache);
    apply_hires(request.hires, params.hires);
    return params;
}

bool StableDiffusionSession::cancel(visual_cancel_mode mode) {
    if (!model_->context()) return false;
    sd_cancel_mode_t native = SD_CANCEL_ALL;
    if (mode == VISUAL_CANCEL_NEW_BATCH_ITEMS) native = SD_CANCEL_NEW_LATENTS;
    if (mode == VISUAL_CANCEL_RESET) native = SD_CANCEL_RESET;
    sd_cancel_generation(model_->context(), native);
    return true;
}

bool StableDiffusionSession::generate_images(
    const visual_image_request& request,
    std::vector<ImageBuffer>& output
) {
    if (!model_->context() || !request.prompt || request.width <= 0 || request.height <= 0) return false;
    std::lock_guard<std::mutex> model_lock(model_->mutex());
    CallbackScope callbacks(callbacks_);
    std::vector<sd_lora_t> loras;
    std::vector<sd_image_t> references;
    std::vector<sd_image_t> identities;
    auto params = image_params(request, loras, references, identities);
    sd_image_t* images = nullptr;
    int count = 0;
    const bool generated = generate_image(model_->context(), &params, &images, &count);
    const bool copied = generated && copy_sd_images(images, count, output);
    if (images) free_sd_images(images, count);
    return copied;
}

bool StableDiffusionSession::generate_video(
    const visual_video_request& request,
    VideoBuffer& output
) {
    if (!model_->context() || !request.prompt || request.width <= 0 || request.height <= 0) return false;
    std::lock_guard<std::mutex> model_lock(model_->mutex());
    CallbackScope callbacks(callbacks_);
    auto loras = make_loras(request.loras, request.lora_count);
    auto controls = make_images(request.control_frames, request.control_frame_count);
    sd_vid_gen_params_t params;
    sd_vid_gen_params_init(&params);
    params.loras = loras.data();
    params.lora_count = static_cast<uint32_t>(loras.size());
    params.prompt = request.prompt;
    params.negative_prompt = request.negative_prompt;
    params.clip_skip = request.clip_skip;
    params.init_image = to_sd_image(request.init_image);
    params.end_image = to_sd_image(request.end_image);
    params.control_frames = controls.data();
    params.control_frames_size = static_cast<int>(controls.size());
    params.width = request.width;
    params.height = request.height;
    params.strength = request.strength;
    params.seed = request.seed;
    params.video_frames = request.frame_count;
    params.fps = request.fps;
    params.vace_strength = request.vace_strength;
    params.moe_boundary = request.moe_boundary;
    params.circular_x = request.circular_x;
    params.circular_y = request.circular_y;
    apply_sample(model_->context(), request.sample, params.sample_params);
    apply_sample(model_->context(), request.high_noise_sample, params.high_noise_sample_params);
    apply_tiling(request.vae_tiling, params.vae_tiling_params);
    apply_cache(request.cache, params.cache);
    apply_hires(request.hires, params.hires);

    sd_image_t* frames = nullptr;
    sd_audio_t* audio = nullptr;
    int frame_count = 0;
    const bool generated = ::generate_video(model_->context(), &params, &frames, &frame_count, &audio);
    bool copied = generated && copy_sd_images(frames, frame_count, output.frames);
    output.fps = static_cast<uint32_t>(std::max(0, request.fps));
    if (copied && audio && audio->data) {
        const uint64_t count = audio->sample_count * audio->channels;
        if (count <= std::numeric_limits<size_t>::max()) {
            output.audio.sample_rate = audio->sample_rate;
            output.audio.channels = audio->channels;
            output.audio.data.assign(audio->data, audio->data + static_cast<size_t>(count));
        } else {
            copied = false;
        }
    }
    if (frames) free_sd_images(frames, frame_count);
    if (audio) free_sd_audio(audio);
    return copied;
}

bool StableDiffusionSession::upscale(
    const visual_image& input,
    uint32_t factor,
    std::vector<ImageBuffer>& output
) {
    if (!model_->upscaler() || !input.data || factor == 0) return false;
    std::lock_guard<std::mutex> model_lock(model_->mutex());
    sd_image_t* images = nullptr;
    int count = 0;
    const bool generated = ::upscale(model_->upscaler(), to_sd_image(input), factor, &images, &count);
    const bool copied = generated && copy_sd_images(images, count, output);
    if (images) free_sd_images(images, count);
    return copied;
}

bool StableDiffusionSession::adetail(
    const visual_image& input,
    const char* prompt,
    const char* negative_prompt,
    const visual_image_request& request,
    std::vector<ImageBuffer>& output
) {
    if (!model_->adetailer() || !model_->context() || !input.data) return false;
    std::lock_guard<std::mutex> model_lock(model_->mutex());
    CallbackScope callbacks(callbacks_);
    std::vector<sd_lora_t> loras;
    std::vector<sd_image_t> references;
    std::vector<sd_image_t> identities;
    auto inpaint = image_params(request, loras, references, identities);
    const sd_adetailer_params_t detail = {prompt, negative_prompt, nullptr};
    sd_image_t* images = nullptr;
    int count = 0;
    const bool generated = adetail_image(
        model_->adetailer(), model_->context(), to_sd_image(input), &detail, &inpaint,
        &images, &count);
    const bool copied = generated && copy_sd_images(images, count, output);
    if (images) free_sd_images(images, count);
    return copied;
}

// stable-diffusion.cpp reports every failure through its log callback; with
// none installed, load errors are invisible. Warnings and errors always reach
// stderr; set FORGE_SD_VERBOSE for info/debug output.
void install_sd_logging() {
    static std::once_flag once;
    std::call_once(once, [] {
        sd_set_log_callback(
            [](enum sd_log_level_t level, const char* text, void*) {
                if (!text) return;
                if (level >= SD_LOG_WARN || std::getenv("FORGE_SD_VERBOSE")) {
                    std::fprintf(stderr, "[stable-diffusion] %s%s",
                                 level == SD_LOG_ERROR ? "Error: " : "",
                                 text);
                }
            },
            nullptr);
    });
}

class StableDiffusionProvider final : public IVisualProvider {
public:
    const char* name() const override { return "stable-diffusion.cpp"; }

    std::shared_ptr<IVisualModel> load(const ModelConfig& model, const RuntimeConfig& runtime) const override {
        install_sd_logging();
        const bool has_generator = !model.model.empty() || !model.diffusion_model.empty();
        sd_ctx_t* context = nullptr;
        if (has_generator) {
            sd_ctx_params_t params;
            sd_ctx_params_init(&params);
            params.model_path = nullable(model.model);
            params.diffusion_model_path = nullable(model.diffusion_model);
            params.high_noise_diffusion_model_path = nullable(model.high_noise_diffusion_model);
            params.uncond_diffusion_model_path = nullable(model.unconditioned_diffusion_model);
            params.clip_l_path = nullable(model.clip_l);
            params.clip_g_path = nullable(model.clip_g);
            params.clip_vision_path = nullable(model.clip_vision);
            params.t5xxl_path = nullable(model.t5xxl);
            params.llm_path = nullable(model.llm);
            params.llm_vision_path = nullable(model.llm_vision);
            params.vae_path = nullable(model.vae);
            params.audio_vae_path = nullable(model.audio_vae);
            params.taesd_path = nullable(model.taesd);
            params.control_net_path = nullable(model.control_net);
            params.motion_module_path = nullable(model.motion_module);
            params.photo_maker_path = nullable(model.photo_maker);
            params.pulid_weights_path = nullable(model.pulid);
            params.n_threads = static_cast<int>(runtime.n_threads);
            params.enable_mmap = true;
            params.flash_attn = runtime.flash_attention;
            params.diffusion_flash_attn = runtime.flash_attention;
            params.max_vram = nullable(runtime.max_vram);
            params.stream_layers = runtime.stream_layers;
            params.eager_load = runtime.eager_load;
            params.backend = runtime.backend == "auto" ? nullptr : nullable(runtime.backend);
            params.params_backend = nullable(runtime.params_backend);
            if (!model.weight_type.empty()) {
                const sd_type_t type = str_to_sd_type(model.weight_type.c_str());
                if (type != SD_TYPE_COUNT) params.wtype = type;
            }
            context = new_sd_ctx(&params);
            if (!context) return nullptr;
        }

        const char* backend = runtime.backend == "auto" ? nullptr : nullable(runtime.backend);
        upscaler_ctx_t* upscaler = model.upscaler.empty()
            ? nullptr
            : new_upscaler_ctx(model.upscaler.c_str(), false, runtime.n_threads, 0, backend, nullable(runtime.params_backend));
        adetailer_ctx_t* adetailer = model.adetailer.empty()
            ? nullptr
            : new_adetailer_ctx(model.adetailer.c_str(), runtime.n_threads, backend, nullable(runtime.params_backend));
        if (!context && !upscaler && !adetailer) return nullptr;
        return std::make_shared<StableDiffusionModel>(context, upscaler, adetailer);
    }
};

[[maybe_unused]] const bool registered = [] {
    ProviderRegistry::get().register_provider("stable-diffusion.cpp", [] {
        return std::make_unique<StableDiffusionProvider>();
    });
    return true;
}();

} // namespace

bool preprocess_canny(visual_image& image, float high, float low, bool inverse) {
    if (!image.data || image.width == 0 || image.height == 0 || image.channels == 0) return false;
    return ::preprocess_canny(to_sd_image(image), high, low, 0.5f, 1.0f, inverse);
}

} // namespace visual
