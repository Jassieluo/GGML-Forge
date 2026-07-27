#include "categories/visual_perception/depth_estimation.h"
#include "image/image_io.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4 || argc > 6) {
        std::cerr << "usage: estimate-depth <model.gguf> <left> <depth.png> "
                     "[CPU|CUDA0|SYCL0] [right]\n";
        return 2;
    }
    forge::media::Image left, right;
    std::string error;
    if (!forge::media::load_image(argv[2], left, error) ||
        (argc == 6 && !forge::media::load_image(argv[5], right, error))) {
        std::cerr << "failed to load image: " << error << '\n';
        return 1;
    }
    depth_runtime_params runtime_params = depth_runtime_default_params();
    if (argc >= 5) runtime_params.device = argv[4];
    depth_runtime_ptr runtime = depth_runtime_create(runtime_params);
    depth_model_ptr model = depth_load_model(runtime, argv[1]);
    depth_session_ptr session = depth_create_session(model);
    if (!runtime || !model || !session) {
        std::cerr << "failed to initialize depth model\n";
        depth_free_session(session); depth_free_model(model); depth_runtime_free(runtime);
        return 1;
    }
    const depth_image left_input{left.width, left.height, 3, left.pixels.data()};
    const depth_image right_input{right.width, right.height, 3, right.pixels.data()};
    depth_request_params request = depth_request_default_params();
    if (argc == 6) request.task = DEPTH_TASK_STEREO;
    depth_map map{};
    const bool ok = depth_estimate(session, &left_input, argc == 6 ? &right_input : nullptr,
                                   request, &map);
    if (ok) {
        const size_t count = static_cast<size_t>(map.width) * map.height;
        float minimum = std::numeric_limits<float>::infinity();
        float maximum = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < count; ++i) if (std::isfinite(map.data[i]) && map.data[i] > 0) {
            minimum = std::min(minimum, map.data[i]); maximum = std::max(maximum, map.data[i]);
        }
        std::vector<uint8_t> pixels(count);
        const float range = maximum > minimum ? maximum - minimum : 1.0f;
        for (size_t i = 0; i < count; ++i) {
            const float value = std::isfinite(map.data[i]) ? map.data[i] : minimum;
            pixels[i] = static_cast<uint8_t>(std::clamp((value - minimum) / range, 0.0f, 1.0f) * 255.0f);
        }
        if (std::filesystem::path(argv[3]).extension() == ".f32") {
            std::ofstream output(argv[3], std::ios::binary);
            output.write(reinterpret_cast<const char*>(map.data),
                         static_cast<std::streamsize>(count * sizeof(float)));
            if (!output) error = "failed to save raw depth map";
        } else if (!forge::media::save_png(argv[3], map.width, map.height, 1, pixels.data(), error)) {
            std::cerr << "failed to save depth map: " << error << '\n';
        }
        if (error.empty()) {
            std::cout << std::fixed << std::setprecision(3) << "wrote " << argv[3]
                      << " range=[" << minimum << ", " << maximum << "] kind=" << map.kind << '\n';
        }
    } else std::cerr << "depth estimation failed\n";
    depth_free_map(&map); depth_free_session(session); depth_free_model(model); depth_runtime_free(runtime);
    return ok && error.empty() ? 0 : 1;
}
