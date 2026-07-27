#include "categories/object_detection/object_detection.h"
#include "image/image_io.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "usage: detect-image <model.gguf> <image> [device]\n";
        return 2;
    }

    forge::media::Image decoded;
    std::string error;
    if (!forge::media::load_image(argv[2], decoded, error)) {
        std::cerr << "failed to load image: " << error << '\n';
        return 1;
    }

    detection_runtime_params runtime_params = detection_runtime_default_params();
    if (argc == 4) runtime_params.device = argv[3];
    detection_runtime_ptr runtime = detection_runtime_create(runtime_params);
    detection_model_ptr model = detection_load_model(runtime, argv[1]);
    detection_session_ptr session = detection_create_session(model);
    if (!runtime || !model || !session) {
        std::cerr << "failed to initialize detection model\n";
        detection_free_session(session);
        detection_free_model(model);
        detection_runtime_free(runtime);
        return 1;
    }

    const detection_image image{
        decoded.width, decoded.height, 3, decoded.pixels.data()};
    detection_result result{};
    detection_request_params request = detection_request_default_params();
    const detection_capabilities capabilities = detection_model_get_capabilities(model);
    if (capabilities.oriented_boxes) request.task = DETECTION_TASK_ORIENTED_BOXES;
    else if (capabilities.instance_masks) request.task = DETECTION_TASK_INSTANCE_MASKS;
    else if (capabilities.keypoints) request.task = DETECTION_TASK_KEYPOINTS;
    const bool ok = detection_detect(session, &image, request, &result);
    if (!ok) {
        std::cerr << "detection failed\n";
    } else {
        std::cout << std::fixed << std::setprecision(2);
        for (size_t index = 0; index < result.instance_count; ++index) {
            const detection_instance& item = result.instances[index];
            const char* label = detection_model_get_label(model, item.class_id);
            const float x1 = item.x - item.width * 0.5f;
            const float y1 = item.y - item.height * 0.5f;
            const float x2 = item.x + item.width * 0.5f;
            const float y2 = item.y + item.height * 0.5f;
            std::cout << index << "  " << (label ? label : "class")
                      << '[' << item.class_id << "]  score=" << item.score;
            if (request.task == DETECTION_TASK_ORIENTED_BOXES) {
                std::cout << "  rbox=(" << item.x << ", " << item.y << ", "
                          << item.width << ", " << item.height
                          << ", angle=" << item.angle << ')';
            } else {
                std::cout << "  box=(" << x1 << ", " << y1 << ", "
                          << x2 << ", " << y2 << ')';
            }
            if (item.mask) {
                size_t foreground = 0;
                const size_t mask_size =
                    static_cast<size_t>(item.mask_width) * item.mask_height;
                for (size_t pixel = 0; pixel < mask_size; ++pixel) {
                    if (item.mask[pixel]) ++foreground;
                }
                std::cout << "  mask=" << item.mask_width << 'x' << item.mask_height
                          << " foreground=" << foreground;
            }
            if (item.keypoints) {
                std::cout << "  keypoints=" << item.keypoint_count;
                for (size_t point = 0; point < item.keypoint_count; ++point) {
                    const detection_keypoint& keypoint = item.keypoints[point];
                    std::cout << " (" << keypoint.x << ',' << keypoint.y
                              << ',' << keypoint.score << ')';
                }
            }
            std::cout << '\n';
        }
    }

    detection_free_result(&result);
    detection_free_session(session);
    detection_free_model(model);
    detection_runtime_free(runtime);
    return ok ? 0 : 1;
}
