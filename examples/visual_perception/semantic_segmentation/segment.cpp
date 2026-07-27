#include "categories/visual_perception/semantic_segmentation.h"
#include "image/image_io.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4 || argc > 5) {
        std::cerr << "usage: segment-image <model.gguf> <image> <mask.png> [CPU|CUDA0|SYCL0]\n";
        return 2;
    }
    forge::media::Image image;
    std::string error;
    if (!forge::media::load_image(argv[2], image, error)) {
        std::cerr << "failed to load image: " << error << '\n';
        return 1;
    }
    segmentation_runtime_params params = segmentation_runtime_default_params();
    if (argc == 5) params.device = argv[4];
    segmentation_runtime_ptr runtime = segmentation_runtime_create(params);
    segmentation_model_ptr model = segmentation_load_model(runtime, argv[1]);
    segmentation_session_ptr session = segmentation_create_session(model);
    if (!runtime || !model || !session) {
        std::cerr << "failed to initialize segmentation model\n";
        segmentation_free_session(session);
        segmentation_free_model(model);
        segmentation_runtime_free(runtime);
        return 1;
    }
    const segmentation_image input{image.width, image.height, 3, image.pixels.data()};
    segmentation_result result{};
    const bool ok = segmentation_segment(
        session, &input, segmentation_request_default_params(), &result);
    if (ok) {
        std::vector<uint8_t> colors(static_cast<size_t>(result.width) * result.height * 3);
        for (size_t i = 0; i < static_cast<size_t>(result.width) * result.height; ++i) {
            uint8_t rgb[3]{};
            if (!segmentation_model_get_color(model, result.class_map[i], rgb)) {
                const uint32_t id = static_cast<uint32_t>(result.class_map[i]);
                rgb[0] = static_cast<uint8_t>((id * 67) & 255);
                rgb[1] = static_cast<uint8_t>((id * 149) & 255);
                rgb[2] = static_cast<uint8_t>((id * 211) & 255);
            }
            colors[i * 3] = rgb[0]; colors[i * 3 + 1] = rgb[1]; colors[i * 3 + 2] = rgb[2];
        }
        if (!forge::media::save_png(argv[3], result.width, result.height, 3, colors.data(), error)) {
            std::cerr << "failed to save mask: " << error << '\n';
        } else {
            std::cout << "wrote " << argv[3] << " (" << result.width << 'x' << result.height << ")\n";
        }
    } else {
        std::cerr << "segmentation failed\n";
    }
    segmentation_free_result(&result);
    segmentation_free_session(session);
    segmentation_free_model(model);
    segmentation_runtime_free(runtime);
    return ok && error.empty() ? 0 : 1;
}
