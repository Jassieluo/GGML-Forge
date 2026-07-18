#include "categories/visual_generation/visual_generation.h"

#include <cstring>
#include <iostream>

int main() {
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
    visual_runtime_free(runtime);
    std::cout << "visual generation API lifecycle checks passed\n";
    return 0;
}
