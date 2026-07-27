#include "categories/visual_perception/image_classification.h"

#include <cstring>
#include <iostream>
#include <vector>

namespace {
int failures = 0;
void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}
}

int main() {
    const classification_runtime_params defaults = classification_runtime_default_params();
    expect(defaults.device && std::strcmp(defaults.device, "auto") == 0 &&
               defaults.n_threads > 0, "classification runtime defaults");
    classification_runtime_ptr runtime = classification_runtime_create({"cpu", 2});
    expect(runtime && std::strcmp(classification_runtime_get_device(runtime), "cpu") == 0 &&
               classification_runtime_get_thread_count(runtime) == 2,
           "classification runtime retains configuration");
    const classification_request_params request = classification_request_default_params();
    expect(request.top_k == 5, "classification request defaults to Top-5");
    expect(classification_load_model(nullptr, "missing.gguf") == nullptr,
           "classification rejects null runtime");
    expect(classification_create_session(nullptr) == nullptr,
           "classification rejects null model session");
    const classification_capabilities empty =
        classification_model_get_capabilities(nullptr);
    expect(empty.class_count == 0, "null classification capabilities are empty");
    std::vector<uint8_t> pixels(8 * 8 * 3, 127);
    const classification_image image{8, 8, 3, pixels.data()};
    classification_result result{};
    expect(!classification_classify(nullptr, &image, request, &result),
           "classification rejects null session");
    expect(result.scores == nullptr && result.score_count == 0,
           "failed classification leaves an empty result");
    classification_free_result(&result);
    classification_free_session(nullptr);
    classification_free_model(nullptr);
    classification_runtime_free(runtime);
    classification_runtime_free(nullptr);
    if (failures == 0) std::cout << "image classification API contract PASSED\n";
    return failures == 0 ? 0 : 1;
}
