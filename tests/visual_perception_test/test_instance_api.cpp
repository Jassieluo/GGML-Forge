// API-contract test for instance perception. No model files are
// required: it verifies defaults, null-handle behaviour, argument validation,
// and that non-GGUF / unknown-architecture inputs fail cleanly.
#include "categories/visual_perception/instance_perception.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* what) {
    if (!condition) {
        std::cerr << "FAILED: " << what << "\n";
        ++failures;
    }
}

} // namespace

int main() {
    const instance_runtime_params defaults = instance_runtime_default_params();
    expect(defaults.device && std::strcmp(defaults.device, "auto") == 0 && defaults.n_threads > 0,
           "runtime defaults");

    instance_runtime_ptr runtime = instance_runtime_create({"cpu", 2});
    expect(runtime != nullptr, "runtime creation");
    expect(runtime && std::strcmp(instance_runtime_get_device(runtime), "cpu") == 0 &&
               instance_runtime_get_thread_count(runtime) == 2,
           "runtime retains configuration");
    expect(instance_runtime_get_device(nullptr) == nullptr &&
               instance_runtime_get_thread_count(nullptr) == 0,
           "null runtime accessors");

    const instance_request_params request = instance_request_default_params();
    expect(request.task == INSTANCE_TASK_BOXES && request.score_threshold == 0.25f &&
               request.iou_threshold == 0.45f && request.max_instances == 300,
           "request defaults");

    // Model loading contract: null arguments, missing files, and files
    // without GGUF architecture metadata must all fail without crashing.
    expect(instance_load_model(nullptr, "missing.gguf") == nullptr, "load with null runtime");
    expect(instance_load_model(runtime, nullptr) == nullptr, "load with null path");
    expect(instance_load_model(runtime, "") == nullptr, "load with empty path");
    expect(instance_load_model(runtime, "definitely-missing.gguf") == nullptr,
           "load with missing file");
    {
        const char* bogus_path = "test_instance_bogus.bin";
        FILE* bogus = std::fopen(bogus_path, "wb");
        if (bogus) {
            std::fputs("not a gguf file", bogus);
            std::fclose(bogus);
            expect(instance_load_model(runtime, bogus_path) == nullptr,
                   "load with non-GGUF file");
            std::remove(bogus_path);
        }
    }

    expect(instance_model_get_provider(nullptr) == nullptr, "null model provider name");
    expect(instance_model_get_label(nullptr, 0) == nullptr, "null model label");
    const instance_capabilities empty = instance_model_get_capabilities(nullptr);
    expect(!empty.boxes && !empty.oriented_boxes && !empty.instance_masks && !empty.keypoints &&
               empty.class_count == 0 && empty.keypoint_count == 0,
           "null model capabilities are empty");

    expect(instance_create_session(nullptr) == nullptr, "session from null model");

    const std::vector<uint8_t> pixels(16 * 16 * 3, 127);
    const instance_image image{16, 16, 3, pixels.data()};
    instance_result result{};
    expect(!instance_perceive(nullptr, &image, request, &result), "perceive on null session");
    expect(result.instances == nullptr && result.instance_count == 0,
           "failed detect leaves result empty");
    instance_free_result(&result); // must be a safe no-op
    instance_free_result(nullptr); // likewise
    instance_free_model(nullptr);
    instance_free_session(nullptr);
    instance_runtime_free(runtime);
    instance_runtime_free(nullptr);

    if (failures == 0) std::cout << "instance perception API contract PASSED\n";
    return failures == 0 ? 0 : 1;
}
