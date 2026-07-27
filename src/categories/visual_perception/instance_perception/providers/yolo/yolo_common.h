#pragma once

#include "providers/instance_provider.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace visual_perception::instance::yolo {

struct ModelConfig {
    std::string version;
    uint32_t input_width = 0;
    uint32_t input_height = 0;
    uint32_t class_count = 0;
    uint32_t reg_max = 0;
    bool oriented_boxes = false;
    uint32_t angle_count = 0;
    bool instance_masks = false;
    uint32_t mask_count = 0;
    bool keypoints = false;
    uint32_t keypoint_count = 0;
    uint32_t keypoint_dimensions = 0;
    std::array<uint32_t, 3> strides{8, 16, 32};
    std::vector<std::string> labels;
};

struct LetterboxImage {
    std::vector<float> pixels;
    float scale = 1.0f;
    int32_t pad_x = 0;
    int32_t pad_y = 0;
};

bool make_letterbox(const instance_image& image, uint32_t target_width,
                    uint32_t target_height, LetterboxImage& output);

bool decode_detections(const float* values, size_t value_count,
                       const ModelConfig& config, const LetterboxImage& letterbox,
                       const instance_image& source, const Request& request,
                       Result& result);

bool decode_instance_masks(const float* values, size_t value_count,
                           const float* prototypes, size_t prototype_count,
                           uint32_t prototype_width, uint32_t prototype_height,
                           const ModelConfig& config,
                           const LetterboxImage& letterbox,
                           const instance_image& source, const Request& request,
                           Result& result);

bool decode_keypoints(const float* values, size_t value_count,
                      const ModelConfig& config,
                      const LetterboxImage& letterbox,
                      const instance_image& source, const Request& request,
                      Result& result);

} // namespace visual_perception::instance::yolo
