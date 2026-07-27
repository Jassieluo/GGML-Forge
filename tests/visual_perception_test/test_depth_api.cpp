// API-contract test for the depth_estimation category. No model files are
// required: it verifies defaults, null-handle behaviour, argument validation,
// and that non-GGUF / unknown-architecture inputs fail cleanly.
#include "categories/visual_perception/depth_estimation.h"

#include <cstdio>
#include <cmath>
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
    const depth_runtime_params defaults = depth_runtime_default_params();
    expect(defaults.device && std::strcmp(defaults.device, "auto") == 0 && defaults.n_threads > 0,
           "runtime defaults");

    depth_runtime_ptr runtime = depth_runtime_create({"cpu", 2});
    expect(runtime != nullptr, "runtime creation");
    expect(runtime && std::strcmp(depth_runtime_get_device(runtime), "cpu") == 0 &&
               depth_runtime_get_thread_count(runtime) == 2,
           "runtime retains configuration");
    expect(depth_runtime_get_device(nullptr) == nullptr &&
               depth_runtime_get_thread_count(nullptr) == 0,
           "null runtime accessors");

    const depth_request_params request = depth_request_default_params();
    expect(request.task == DEPTH_TASK_MONOCULAR, "request defaults");

    expect(depth_load_model(nullptr, "missing.gguf") == nullptr, "load with null runtime");
    expect(depth_load_model(runtime, nullptr) == nullptr, "load with null path");
    expect(depth_load_model(runtime, "") == nullptr, "load with empty path");
    expect(depth_load_model(runtime, "definitely-missing.gguf") == nullptr,
           "load with missing file");
    {
        const char* bogus_path = "test_depth_bogus.bin";
        FILE* bogus = std::fopen(bogus_path, "wb");
        if (bogus) {
            std::fputs("not a gguf file", bogus);
            std::fclose(bogus);
            expect(depth_load_model(runtime, bogus_path) == nullptr, "load with non-GGUF file");
            std::remove(bogus_path);
        }
    }

    expect(depth_model_get_provider(nullptr) == nullptr, "null model provider name");
    const depth_capabilities empty = depth_model_get_capabilities(nullptr);
    expect(!empty.monocular && !empty.stereo && !empty.metric,
           "null model capabilities are empty");

    expect(depth_create_session(nullptr) == nullptr, "session from null model");

    const std::vector<uint8_t> pixels(16 * 16 * 3, 127);
    const depth_image image{16, 16, 3, pixels.data()};
    depth_map map{};
    expect(!depth_estimate(nullptr, &image, nullptr, request, &map),
           "estimate on null session");
    expect(map.data == nullptr && map.width == 0 && map.height == 0,
           "failed estimate leaves map empty");
    depth_free_map(&map); // must be a safe no-op
    float disparities[] = {10.0f, 20.0f, 0.0f, NAN};
    const depth_map disparity{2, 2, DEPTH_MAP_DISPARITY, disparities};
    depth_map metric{};
    expect(depth_disparity_to_metric(&disparity, {500.0f, 0.1f}, &metric),
           "disparity to metric conversion");
    expect(metric.kind == DEPTH_MAP_METRIC && metric.width == 2 && metric.height == 2 &&
               std::fabs(metric.data[0] - 5.0f) < 1e-6f &&
               std::fabs(metric.data[1] - 2.5f) < 1e-6f &&
               metric.data[2] == 0.0f && metric.data[3] == 0.0f,
           "stereo calibration formula");
    depth_free_map(&metric);
    expect(!depth_disparity_to_metric(&disparity, {0.0f, 0.1f}, &metric),
           "invalid stereo calibration is rejected");
    depth_free_map(nullptr); // likewise
    depth_free_model(nullptr);
    depth_free_session(nullptr);
    depth_runtime_free(runtime);
    depth_runtime_free(nullptr);

    if (failures == 0) std::cout << "depth_estimation API contract PASSED\n";
    return failures == 0 ? 0 : 1;
}
