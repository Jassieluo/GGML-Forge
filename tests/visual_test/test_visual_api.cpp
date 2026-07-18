#include "categories/visual_generation/visual_generation.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cstring>
#include <iostream>
#include <string>

namespace {

struct CallbackState {
    int progress_calls = 0;
    int preview_calls = 0;
};

void progress(int32_t step, int32_t steps, float seconds, void* data) {
    auto* state = static_cast<CallbackState*>(data);
    ++state->progress_calls;
    std::cout << "step " << step << '/' << steps << " (" << seconds << "s)\n";
}

void preview(int32_t, const visual_image* frames, size_t count, bool, void* data) {
    if (frames && count > 0 && frames[0].data) {
        ++static_cast<CallbackState*>(data)->preview_calls;
    }
}

bool write_png(const char* path, const visual_image& image) {
    if (!path || !image.data || image.channels < 3) return false;
    return stbi_write_png(
        path,
        static_cast<int>(image.width),
        static_cast<int>(image.height),
        static_cast<int>(image.channels),
        image.data,
        static_cast<int>(image.width * image.channels)) != 0;
}

} // namespace

int main(int argc, char** argv) {
    const visual_runtime_params defaults = visual_runtime_default_params();
    if (!defaults.backend || std::strcmp(defaults.backend, "auto") != 0 || defaults.n_threads == 0) {
        std::cerr << "invalid visual runtime defaults\n";
        return 1;
    }

    visual_runtime_ptr runtime = visual_runtime_create(defaults);
    if (!runtime) {
        std::cerr << "failed to create visual runtime\n";
        return 1;
    }

    const visual_model_params model_defaults = visual_model_default_params();
    if (!model_defaults.weight_type || std::strcmp(model_defaults.weight_type, "f16") != 0) {
        std::cerr << "invalid visual model defaults\n";
        return 1;
    }

    const visual_image_request image = visual_image_request_default_params();
    const visual_video_request video = visual_video_request_default_params();
    if (image.width <= 0 || image.height <= 0 || image.batch_count <= 0 ||
        video.width <= 0 || video.height <= 0 || video.frame_count <= 0 || video.fps <= 0) {
        std::cerr << "invalid visual request defaults\n";
        return 1;
    }

    visual_model_params empty_model = visual_model_default_params();
    if (visual_load_model(nullptr, &empty_model) || visual_load_model(runtime, nullptr) ||
        visual_load_model(runtime, &empty_model) || visual_create_session(nullptr) ||
        visual_session_cancel(nullptr, VISUAL_CANCEL_ALL)) {
        std::cerr << "visual null/empty contract failed\n";
        return 1;
    }

    const visual_capabilities empty = visual_model_get_capabilities(nullptr);
    if (empty.text_to_image || empty.video || empty.upscale || empty.cancellation) {
        std::cerr << "null visual model reported capabilities\n";
        return 1;
    }

    visual_image* images = nullptr;
    size_t image_count = 0;
    if (visual_generate_images(nullptr, &image, &images, &image_count) || images || image_count != 0) {
        std::cerr << "visual generation accepted a null session\n";
        return 1;
    }

    visual_free_images(nullptr, 0);
    visual_video empty_video{};
    visual_free_video(&empty_video);

    if (argc > 1) {
        visual_model_params params = visual_model_default_params();
        params.model = argv[1];
        visual_model_ptr model = visual_load_model(runtime, &params);
        const visual_capabilities capabilities = visual_model_get_capabilities(model);
        visual_session_ptr session = model ? visual_create_session(model) : nullptr;
        if (!model || !session || !capabilities.text_to_image || !capabilities.preview ||
            !capabilities.cancellation) {
            std::cerr << "failed to load a text-to-image model\n";
            visual_free_session(session);
            visual_free_model(model);
            visual_runtime_free(runtime);
            return 1;
        }

        CallbackState callbacks;
        visual_session_set_callbacks(session, progress, preview, &callbacks);
        visual_image_request request = visual_image_request_default_params();
        request.prompt =
            "masterpiece, cinematic photograph of a small red fox sitting in a snowy pine forest, "
            "soft morning light, detailed fur, shallow depth of field";
        request.negative_prompt =
            "low quality, blurry, deformed, extra limbs, text, watermark";
        request.width = 512;
        request.height = 512;
        request.seed = 42;
        request.batch_count = 1;
        request.sample.steps = 20;
        request.sample.text_guidance = 7.0f;

        visual_image* generated = nullptr;
        size_t generated_count = 0;
        const bool generated_ok = visual_generate_images(
            session, &request, &generated, &generated_count);
        const bool output_ok = argc < 3 ||
            (generated_count > 0 && write_png(argv[2], generated[0]));
        if (!generated_ok || generated_count != 1 || !generated[0].data ||
            generated[0].width != 512 || generated[0].height != 512 ||
            callbacks.progress_calls == 0 || !output_ok) {
            std::cerr << "real visual generation failed\n";
            visual_free_images(generated, generated_count);
            visual_free_session(session);
            visual_free_model(model);
            visual_runtime_free(runtime);
            return 1;
        }
        visual_free_images(generated, generated_count);
        visual_free_session(session);
        visual_free_model(model);
        std::cout << "stable-diffusion.cpp image generation passed"
                  << " (progress=" << callbacks.progress_calls
                  << ", preview=" << callbacks.preview_calls << ")\n";
    }

    visual_runtime_free(runtime);
    std::cout << "visual generation API lifecycle checks passed\n";
    return 0;
}
