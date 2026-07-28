#include "categories/visual_generation/visual_generation.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

void print_progress(int32_t step, int32_t steps, float seconds, void*) {
    std::cerr << "\rStep " << step << '/' << steps << " (" << seconds << "s)" << std::flush;
}

bool write_ppm(const char* path, const visual_image& image) {
    if (!path || !image.data || image.channels < 3) return false;
    const std::filesystem::path output = std::filesystem::u8path(path);
    if (output.has_parent_path()) std::filesystem::create_directories(output.parent_path());
    std::ofstream file(output, std::ios::binary);
    if (!file) return false;
    file << "P6\n" << image.width << ' ' << image.height << "\n255\n";
    for (uint64_t pixel = 0; pixel < static_cast<uint64_t>(image.width) * image.height; ++pixel) {
        file.write(reinterpret_cast<const char*>(image.data + pixel * image.channels), 3);
    }
    return static_cast<bool>(file);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " <model> [output.ppm] [prompt]\n";
        return 2;
    }
    const char* output = argc > 2 ? argv[2] : "outputs/visual_generation/example.ppm";
    const char* prompt = argc > 3
        ? argv[3]
        : "cinematic photograph of a red fox in a snowy pine forest, detailed fur";

    visual_runtime_params runtime_params = visual_runtime_default_params();
    visual_runtime_ptr runtime = visual_runtime_create(runtime_params);
    visual_model_params model_params = visual_model_default_params();
    model_params.model = argv[1];
    visual_model_ptr model = runtime ? visual_load_model(runtime, &model_params) : nullptr;
    visual_session_ptr session = model ? visual_create_session(model) : nullptr;
    const visual_capabilities capabilities = visual_model_get_capabilities(model);
    if (!session || !capabilities.text_to_image) {
        std::cerr << "Failed to initialize a text-to-image model.\n";
        visual_free_session(session);
        visual_free_model(model);
        visual_runtime_free(runtime);
        return 1;
    }

    visual_session_set_callbacks(session, print_progress, nullptr, nullptr);
    visual_image_request request = visual_image_request_default_params();
    request.prompt = prompt;
    request.negative_prompt = "low quality, blurry, deformed, text, watermark";
    request.width = 512;
    request.height = 512;
    request.seed = 42;
    const bool is_sdxs = std::string(argv[1]).find("sdxs") != std::string::npos;
    request.sample.steps = is_sdxs ? 1 : 20;
    request.sample.text_guidance = is_sdxs ? 1.0f : 7.0f;

    visual_image* images = nullptr;
    size_t image_count = 0;
    const bool generated = visual_generate_images(session, &request, &images, &image_count);
    std::cerr << '\n';
    const bool written = generated && image_count > 0 && write_ppm(output, images[0]);
    visual_free_images(images, image_count);
    visual_free_session(session);
    visual_free_model(model);
    visual_runtime_free(runtime);
    if (!written) {
        std::cerr << "Image generation or PPM output failed.\n";
        return 1;
    }
    std::cout << "Wrote " << output << '\n';
    return 0;
}
