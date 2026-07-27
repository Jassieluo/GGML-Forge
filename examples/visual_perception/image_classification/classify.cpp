#include "categories/visual_perception/image_classification.h"
#include "image/image_io.h"

#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "usage: classify-image <model.gguf> <image> [CPU|CUDA0|SYCL0]\n";
        return 2;
    }
    forge::media::Image decoded;
    std::string error;
    if (!forge::media::load_image(argv[2], decoded, error)) {
        std::cerr << "failed to load image: " << error << '\n';
        return 1;
    }
    classification_runtime_params params = classification_runtime_default_params();
    if (argc == 4) params.device = argv[3];
    classification_runtime_ptr runtime = classification_runtime_create(params);
    classification_model_ptr model = classification_load_model(runtime, argv[1]);
    classification_session_ptr session = classification_create_session(model);
    if (!runtime || !model || !session) {
        std::cerr << "failed to initialize classification model\n";
        classification_free_session(session);
        classification_free_model(model);
        classification_runtime_free(runtime);
        return 1;
    }
    const classification_image image{
        decoded.width, decoded.height, 3, decoded.pixels.data()};
    classification_result result{};
    const bool ok = classification_classify(
        session, &image, classification_request_default_params(), &result);
    if (ok) {
        std::cout << std::fixed << std::setprecision(4);
        for (size_t index = 0; index < result.score_count; ++index) {
            const classification_score& score = result.scores[index];
            const char* label = classification_model_get_label(model, score.class_id);
            std::cout << index << "  " << (label ? label : "class")
                      << '[' << score.class_id << "]  score=" << score.score << '\n';
        }
    } else {
        std::cerr << "classification failed\n";
    }
    classification_free_result(&result);
    classification_free_session(session);
    classification_free_model(model);
    classification_runtime_free(runtime);
    return ok ? 0 : 1;
}
