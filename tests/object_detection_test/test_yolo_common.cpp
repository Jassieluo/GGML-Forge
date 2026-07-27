#include "providers/yolo/yolo_common.h"

#include <algorithm>
#include <cmath>
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

void set_channel(std::vector<float>& output, size_t anchors, size_t anchor,
                 size_t channel, float value) {
    output[anchor + anchors * channel] = value;
}

void set_distance(std::vector<float>& output, size_t anchors, size_t anchor,
                  uint32_t reg_max, size_t side, uint32_t bin) {
    for (uint32_t index = 0; index < reg_max; ++index) {
        set_channel(output, anchors, anchor, side * reg_max + index,
                    index == bin ? 12.0f : -12.0f);
    }
}

void set_candidate(std::vector<float>& output, size_t anchors, size_t anchor,
                   uint32_t reg_max, uint32_t class_id, float score_logit,
                   uint32_t left, uint32_t top, uint32_t right, uint32_t bottom) {
    set_distance(output, anchors, anchor, reg_max, 0, left);
    set_distance(output, anchors, anchor, reg_max, 1, top);
    set_distance(output, anchors, anchor, reg_max, 2, right);
    set_distance(output, anchors, anchor, reg_max, 3, bottom);
    set_channel(output, anchors, anchor, reg_max * 4 + class_id, score_logit);
}

} // namespace

int main() {
    std::vector<uint8_t> pixels(4 * 2 * 3, 255);
    const detection_image wide_image{4, 2, 3, pixels.data()};
    detection::yolo::LetterboxImage letterbox;
    expect(detection::yolo::make_letterbox(wide_image, 8, 8, letterbox),
           "letterbox accepts tightly packed RGB8");
    expect(std::abs(letterbox.scale - 2.0f) < 1e-6f &&
               letterbox.pad_x == 0 && letterbox.pad_y == 2,
           "letterbox records centered scale and padding");
    expect(letterbox.pixels.size() == 8 * 8 * 3,
           "letterbox produces planar RGB float input");
    expect(std::abs(letterbox.pixels[0] - 114.0f / 255.0f) < 1e-6f &&
               std::abs(letterbox.pixels[2 * 8] - 1.0f) < 1e-6f,
           "letterbox uses YOLO padding and normalized pixels");

    detection::yolo::ModelConfig config;
    config.version = "v8";
    config.input_width = 32;
    config.input_height = 32;
    config.class_count = 2;
    config.reg_max = 4;
    config.strides = {8, 16, 32};
    const size_t anchors = 16 + 4 + 1;
    const size_t channels = config.reg_max * 4 + config.class_count;
    std::vector<float> output(anchors * channels, -20.0f);

    // Two same-class predictions describe the same box and must collapse;
    // a different-class prediction for that box remains after class-aware NMS.
    set_candidate(output, anchors, 5, config.reg_max, 0, 10.0f, 1, 1, 2, 2);
    set_candidate(output, anchors, 6, config.reg_max, 0, 9.0f, 2, 1, 1, 2);
    set_candidate(output, anchors, 9, config.reg_max, 1, 8.0f, 1, 2, 2, 1);

    std::vector<uint8_t> square_pixels(32 * 32 * 3, 0);
    const detection_image square_image{32, 32, 3, square_pixels.data()};
    detection::yolo::LetterboxImage identity;
    expect(detection::yolo::make_letterbox(square_image, 32, 32, identity),
           "identity letterbox succeeds");
    detection::Request request;
    request.score_threshold = 0.25f;
    request.iou_threshold = 0.45f;
    request.max_instances = 10;
    detection::Result result;
    expect(detection::yolo::decode_detections(
               output.data(), output.size(), config, identity, square_image, request, result),
           "YOLO output decodes");
    expect(result.instances.size() == 2,
           "class-aware NMS suppresses duplicate boxes only within a class");
    if (result.instances.size() == 2) {
        expect(result.instances[0].class_id == 0 && result.instances[1].class_id == 1,
               "decoded classes retain score order");
        expect(std::abs(result.instances[0].x - 16.0f) < 0.01f &&
                   std::abs(result.instances[0].width - 24.0f) < 0.01f,
               "DFL distances decode into source-image coordinates");
    }

    detection::yolo::ModelConfig segment_config = config;
    segment_config.instance_masks = true;
    segment_config.mask_count = 2;
    const size_t segment_channels = channels + segment_config.mask_count;
    std::vector<float> segment_output(anchors * segment_channels, -20.0f);
    set_candidate(segment_output, anchors, 5, config.reg_max, 0, 10.0f, 1, 1, 2, 2);
    set_candidate(segment_output, anchors, 6, config.reg_max, 0, 9.0f, 2, 1, 1, 2);
    set_candidate(segment_output, anchors, 9, config.reg_max, 1, 8.0f, 1, 2, 2, 1);
    const size_t coefficient_channel = config.reg_max * 4 + config.class_count;
    for (size_t anchor : {size_t(5), size_t(6), size_t(9)}) {
        set_channel(segment_output, anchors, anchor, coefficient_channel, 10.0f);
        set_channel(segment_output, anchors, anchor, coefficient_channel + 1, 0.0f);
    }
    std::vector<float> prototypes(8 * 8 * segment_config.mask_count, 0.0f);
    std::fill(prototypes.begin(), prototypes.begin() + 8 * 8, 1.0f);
    detection::Result mask_result;
    request.task = DETECTION_TASK_INSTANCE_MASKS;
    expect(detection::yolo::decode_instance_masks(
               segment_output.data(), segment_output.size(), prototypes.data(),
               prototypes.size(), 8, 8, segment_config, identity, square_image,
               request, mask_result),
           "YOLO instance masks decode");
    expect(mask_result.instances.size() == 2 && mask_result.masks.size() == 2,
           "NMS-selected instances own one mask each");
    if (!mask_result.instances.empty()) {
        const detection_instance& masked = mask_result.instances.front();
        expect(masked.mask != nullptr && masked.mask_width > 0 && masked.mask_height > 0,
               "instance exposes a box-local mask");
        const size_t mask_pixels = static_cast<size_t>(masked.mask_width) * masked.mask_height;
        expect(std::all_of(masked.mask, masked.mask + mask_pixels,
                           [](uint8_t value) { return value == 255; }),
               "prototype coefficients synthesize the expected binary mask");
    }

    if (failures == 0) std::cout << "YOLO common pipeline PASSED\n";
    return failures == 0 ? 0 : 1;
}
