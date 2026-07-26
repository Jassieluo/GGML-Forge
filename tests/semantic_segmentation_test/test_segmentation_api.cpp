// API-contract test for the semantic_segmentation category. No model files
// are required: it verifies defaults, null-handle behaviour, argument
// validation, and that non-GGUF / unknown-architecture inputs fail cleanly.
#include "categories/semantic_segmentation/semantic_segmentation.h"

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
    const segmentation_runtime_params defaults = segmentation_runtime_default_params();
    expect(defaults.device && std::strcmp(defaults.device, "auto") == 0 && defaults.n_threads > 0,
           "runtime defaults");

    segmentation_runtime_ptr runtime = segmentation_runtime_create({"cpu", 2});
    expect(runtime != nullptr, "runtime creation");
    expect(runtime && std::strcmp(segmentation_runtime_get_device(runtime), "cpu") == 0 &&
               segmentation_runtime_get_thread_count(runtime) == 2,
           "runtime retains configuration");
    expect(segmentation_runtime_get_device(nullptr) == nullptr &&
               segmentation_runtime_get_thread_count(nullptr) == 0,
           "null runtime accessors");

    const segmentation_request_params request = segmentation_request_default_params();
    expect(!request.want_confidence, "request defaults");

    expect(segmentation_load_model(nullptr, "missing.gguf") == nullptr, "load with null runtime");
    expect(segmentation_load_model(runtime, nullptr) == nullptr, "load with null path");
    expect(segmentation_load_model(runtime, "") == nullptr, "load with empty path");
    expect(segmentation_load_model(runtime, "definitely-missing.gguf") == nullptr,
           "load with missing file");
    {
        const char* bogus_path = "test_segmentation_bogus.bin";
        FILE* bogus = std::fopen(bogus_path, "wb");
        if (bogus) {
            std::fputs("not a gguf file", bogus);
            std::fclose(bogus);
            expect(segmentation_load_model(runtime, bogus_path) == nullptr,
                   "load with non-GGUF file");
            std::remove(bogus_path);
        }
    }

    expect(segmentation_model_get_provider(nullptr) == nullptr, "null model provider name");
    expect(segmentation_model_get_label(nullptr, 0) == nullptr, "null model label");
    uint8_t rgb[3] = {1, 2, 3};
    expect(!segmentation_model_get_color(nullptr, 0, rgb), "null model color");
    const segmentation_capabilities empty = segmentation_model_get_capabilities(nullptr);
    expect(empty.class_count == 0 && !empty.confidence, "null model capabilities are empty");

    expect(segmentation_create_session(nullptr) == nullptr, "session from null model");

    const std::vector<uint8_t> pixels(16 * 16 * 3, 127);
    const segmentation_image image{16, 16, 3, pixels.data()};
    segmentation_result result{};
    expect(!segmentation_segment(nullptr, &image, request, &result),
           "segment on null session");
    expect(result.class_map == nullptr && result.confidence == nullptr &&
               result.width == 0 && result.height == 0,
           "failed segment leaves result empty");
    segmentation_free_result(&result); // must be a safe no-op
    segmentation_free_result(nullptr); // likewise
    segmentation_free_model(nullptr);
    segmentation_free_session(nullptr);
    segmentation_runtime_free(runtime);
    segmentation_runtime_free(nullptr);

    if (failures == 0) std::cout << "semantic_segmentation API contract PASSED\n";
    return failures == 0 ? 0 : 1;
}
