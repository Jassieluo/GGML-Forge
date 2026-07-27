#include "providers/yolo/yolo_common.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace detection::yolo {
namespace {

float source_channel(const detection_image& image, int x, int y, int channel) {
    x = std::clamp(x, 0, static_cast<int>(image.width) - 1);
    y = std::clamp(y, 0, static_cast<int>(image.height) - 1);
    const int source_channel_index = image.channels == 1 ? 0 : channel;
    const size_t index = (static_cast<size_t>(y) * image.width + x) * image.channels +
                         static_cast<size_t>(source_channel_index);
    return image.data[index] / 255.0f;
}

float bilinear_channel(const detection_image& image, float x, float y, int channel) {
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const float wx = x - x0;
    const float wy = y - y0;
    const float top = source_channel(image, x0, y0, channel) * (1.0f - wx) +
                      source_channel(image, x0 + 1, y0, channel) * wx;
    const float bottom = source_channel(image, x0, y0 + 1, channel) * (1.0f - wx) +
                         source_channel(image, x0 + 1, y0 + 1, channel) * wx;
    return top * (1.0f - wy) + bottom * wy;
}

float sigmoid(float value) {
    if (value >= 0.0f) {
        const float exponent = std::exp(-value);
        return 1.0f / (1.0f + exponent);
    }
    const float exponent = std::exp(value);
    return exponent / (1.0f + exponent);
}

float dfl_expectation(const float* values, size_t anchors, size_t channel,
                      uint32_t reg_max, size_t anchor) {
    float maximum = -std::numeric_limits<float>::infinity();
    for (uint32_t index = 0; index < reg_max; ++index) {
        maximum = std::max(maximum, values[anchor + anchors * (channel + index)]);
    }
    float denominator = 0.0f;
    float numerator = 0.0f;
    for (uint32_t index = 0; index < reg_max; ++index) {
        const float probability = std::exp(
            values[anchor + anchors * (channel + index)] - maximum);
        denominator += probability;
        numerator += probability * index;
    }
    return denominator > 0.0f ? numerator / denominator : 0.0f;
}

float intersection_over_union(const detection_instance& left,
                              const detection_instance& right) {
    const float left_x1 = left.x - left.width * 0.5f;
    const float left_y1 = left.y - left.height * 0.5f;
    const float left_x2 = left.x + left.width * 0.5f;
    const float left_y2 = left.y + left.height * 0.5f;
    const float right_x1 = right.x - right.width * 0.5f;
    const float right_y1 = right.y - right.height * 0.5f;
    const float right_x2 = right.x + right.width * 0.5f;
    const float right_y2 = right.y + right.height * 0.5f;
    const float intersection =
        std::max(0.0f, std::min(left_x2, right_x2) - std::max(left_x1, right_x1)) *
        std::max(0.0f, std::min(left_y2, right_y2) - std::max(left_y1, right_y1));
    const float union_area = left.width * left.height + right.width * right.height - intersection;
    return union_area > 0.0f ? intersection / union_area : 0.0f;
}

} // namespace

bool make_letterbox(const detection_image& image, uint32_t target_width,
                    uint32_t target_height, LetterboxImage& output) {
    output = {};
    if (!image.data || image.width == 0 || image.height == 0 ||
        (image.channels != 1 && image.channels != 3 && image.channels != 4) ||
        target_width == 0 || target_height == 0) return false;
    if (static_cast<size_t>(target_width) >
        std::numeric_limits<size_t>::max() / target_height / 3) return false;

    output.scale = std::min(
        static_cast<float>(target_width) / image.width,
        static_cast<float>(target_height) / image.height);
    const int resized_width = std::max(
        1, static_cast<int>(std::round(image.width * output.scale)));
    const int resized_height = std::max(
        1, static_cast<int>(std::round(image.height * output.scale)));
    output.pad_x = (static_cast<int>(target_width) - resized_width) / 2;
    output.pad_y = (static_cast<int>(target_height) - resized_height) / 2;
    output.pixels.assign(static_cast<size_t>(target_width) * target_height * 3,
                         114.0f / 255.0f);
    for (int y = 0; y < resized_height; ++y) {
        const float source_y = (y + 0.5f) / output.scale - 0.5f;
        for (int x = 0; x < resized_width; ++x) {
            const float source_x = (x + 0.5f) / output.scale - 0.5f;
            for (int channel = 0; channel < 3; ++channel) {
                const size_t destination = static_cast<size_t>(x + output.pad_x) +
                    static_cast<size_t>(target_width) *
                        (static_cast<size_t>(y + output.pad_y) +
                         static_cast<size_t>(target_height) * channel);
                output.pixels[destination] =
                    bilinear_channel(image, source_x, source_y, channel);
            }
        }
    }
    return true;
}

namespace {

struct Candidate {
    detection_instance instance{};
    size_t anchor = 0;
};

bool decode_candidates(const float* values, size_t value_count,
                       const ModelConfig& config, const LetterboxImage& letterbox,
                       const detection_image& source, const Request& request,
                       std::vector<Candidate>& selected, size_t& anchors) {
    selected.clear();
    anchors = 0;
    if (!values || config.class_count == 0 || config.reg_max == 0 ||
        config.input_width == 0 || config.input_height == 0 || letterbox.scale <= 0.0f) return false;

    std::array<size_t, 4> offsets{};
    for (size_t scale = 0; scale < config.strides.size(); ++scale) {
        const uint32_t stride = config.strides[scale];
        if (stride == 0 || config.input_width % stride != 0 ||
            config.input_height % stride != 0) return false;
        offsets[scale + 1] = offsets[scale] +
            static_cast<size_t>(config.input_width / stride) *
            static_cast<size_t>(config.input_height / stride);
    }
    anchors = offsets.back();
    const size_t channels = static_cast<size_t>(config.reg_max) * 4 +
                            config.class_count + config.mask_count;
    if (anchors == 0 || channels > std::numeric_limits<size_t>::max() / anchors ||
        value_count != anchors * channels) return false;

    std::vector<Candidate> candidates;
    for (size_t scale = 0; scale < config.strides.size(); ++scale) {
        const uint32_t stride = config.strides[scale];
        const uint32_t grid_width = config.input_width / stride;
        for (size_t anchor = offsets[scale]; anchor < offsets[scale + 1]; ++anchor) {
            const size_t local = anchor - offsets[scale];
            const uint32_t grid_x = static_cast<uint32_t>(local % grid_width);
            const uint32_t grid_y = static_cast<uint32_t>(local / grid_width);
            float score = 0.0f;
            int32_t class_id = 0;
            for (uint32_t class_index = 0; class_index < config.class_count; ++class_index) {
                const float candidate = sigmoid(values[
                    anchor + anchors * (static_cast<size_t>(config.reg_max) * 4 + class_index)]);
                if (candidate > score) {
                    score = candidate;
                    class_id = static_cast<int32_t>(class_index);
                }
            }
            if (score < request.score_threshold) continue;

            const float left = dfl_expectation(values, anchors, 0, config.reg_max, anchor);
            const float top = dfl_expectation(values, anchors, config.reg_max, config.reg_max, anchor);
            const float right = dfl_expectation(values, anchors, config.reg_max * 2, config.reg_max, anchor);
            const float bottom = dfl_expectation(values, anchors, config.reg_max * 3, config.reg_max, anchor);
            const float center_x = (grid_x + 0.5f) * stride;
            const float center_y = (grid_y + 0.5f) * stride;
            float x1 = (center_x - left * stride - letterbox.pad_x) / letterbox.scale;
            float y1 = (center_y - top * stride - letterbox.pad_y) / letterbox.scale;
            float x2 = (center_x + right * stride - letterbox.pad_x) / letterbox.scale;
            float y2 = (center_y + bottom * stride - letterbox.pad_y) / letterbox.scale;
            x1 = std::clamp(x1, 0.0f, static_cast<float>(source.width));
            y1 = std::clamp(y1, 0.0f, static_cast<float>(source.height));
            x2 = std::clamp(x2, 0.0f, static_cast<float>(source.width));
            y2 = std::clamp(y2, 0.0f, static_cast<float>(source.height));
            if (x2 <= x1 || y2 <= y1) continue;
            detection_instance instance{};
            instance.x = (x1 + x2) * 0.5f;
            instance.y = (y1 + y2) * 0.5f;
            instance.width = x2 - x1;
            instance.height = y2 - y1;
            instance.class_id = class_id;
            instance.score = score;
            candidates.push_back({instance, anchor});
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
        return left.instance.score > right.instance.score;
    });
    for (const Candidate& candidate : candidates) {
        bool suppressed = false;
        for (const Candidate& kept : selected) {
            if (candidate.instance.class_id == kept.instance.class_id &&
                intersection_over_union(candidate.instance, kept.instance) > request.iou_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            selected.push_back(candidate);
            if (selected.size() >= static_cast<size_t>(request.max_instances)) break;
        }
    }
    return true;
}

float sample_prototype(const float* prototypes, size_t plane_size,
                       uint32_t width, uint32_t height, uint32_t channel,
                       float x, float y) {
    x = std::clamp(x, 0.0f, static_cast<float>(width - 1));
    y = std::clamp(y, 0.0f, static_cast<float>(height - 1));
    const uint32_t x0 = static_cast<uint32_t>(std::floor(x));
    const uint32_t y0 = static_cast<uint32_t>(std::floor(y));
    const uint32_t x1 = std::min(x0 + 1, width - 1);
    const uint32_t y1 = std::min(y0 + 1, height - 1);
    const float wx = x - x0;
    const float wy = y - y0;
    const size_t base = static_cast<size_t>(channel) * plane_size;
    const float top = prototypes[base + x0 + static_cast<size_t>(width) * y0] * (1.0f - wx) +
                      prototypes[base + x1 + static_cast<size_t>(width) * y0] * wx;
    const float bottom = prototypes[base + x0 + static_cast<size_t>(width) * y1] * (1.0f - wx) +
                         prototypes[base + x1 + static_cast<size_t>(width) * y1] * wx;
    return top * (1.0f - wy) + bottom * wy;
}

} // namespace

bool decode_detections(const float* values, size_t value_count,
                       const ModelConfig& config, const LetterboxImage& letterbox,
                       const detection_image& source, const Request& request,
                       Result& result) {
    result.instances.clear();
    result.masks.clear();
    std::vector<Candidate> selected;
    size_t anchors = 0;
    if (!decode_candidates(values, value_count, config, letterbox, source,
                           request, selected, anchors)) return false;
    result.instances.reserve(selected.size());
    for (const Candidate& candidate : selected) result.instances.push_back(candidate.instance);
    return true;
}

bool decode_instance_masks(const float* values, size_t value_count,
                           const float* prototypes, size_t prototype_count,
                           uint32_t prototype_width, uint32_t prototype_height,
                           const ModelConfig& config,
                           const LetterboxImage& letterbox,
                           const detection_image& source, const Request& request,
                           Result& result) {
    result.instances.clear();
    result.masks.clear();
    if (!config.instance_masks || config.mask_count == 0 || !prototypes ||
        prototype_width == 0 || prototype_height == 0) return false;
    const size_t plane_size = static_cast<size_t>(prototype_width) * prototype_height;
    if (plane_size > std::numeric_limits<size_t>::max() / config.mask_count ||
        prototype_count != plane_size * config.mask_count) return false;

    std::vector<Candidate> selected;
    size_t anchors = 0;
    if (!decode_candidates(values, value_count, config, letterbox, source,
                           request, selected, anchors)) return false;
    result.masks.reserve(selected.size());
    result.instances.reserve(selected.size());
    const size_t coefficient_base = static_cast<size_t>(config.reg_max) * 4 +
                                    config.class_count;
    for (const Candidate& candidate : selected) {
        detection_instance instance = candidate.instance;
        const uint32_t mask_width = std::max(
            1u, std::min(source.width, static_cast<uint32_t>(std::ceil(instance.width))));
        const uint32_t mask_height = std::max(
            1u, std::min(source.height, static_cast<uint32_t>(std::ceil(instance.height))));
        if (static_cast<size_t>(mask_width) >
            std::numeric_limits<size_t>::max() / mask_height) return false;
        std::vector<uint8_t> mask(static_cast<size_t>(mask_width) * mask_height);
        const float x1 = instance.x - instance.width * 0.5f;
        const float y1 = instance.y - instance.height * 0.5f;
        for (uint32_t y = 0; y < mask_height; ++y) {
            const float source_y = y1 + (y + 0.5f) * instance.height / mask_height;
            const float model_y = source_y * letterbox.scale + letterbox.pad_y;
            const float prototype_y = (model_y + 0.5f) * prototype_height /
                                      config.input_height - 0.5f;
            for (uint32_t x = 0; x < mask_width; ++x) {
                const float source_x = x1 + (x + 0.5f) * instance.width / mask_width;
                const float model_x = source_x * letterbox.scale + letterbox.pad_x;
                const float prototype_x = (model_x + 0.5f) * prototype_width /
                                          config.input_width - 0.5f;
                float logit = 0.0f;
                for (uint32_t channel = 0; channel < config.mask_count; ++channel) {
                    const float coefficient = values[
                        candidate.anchor + anchors * (coefficient_base + channel)];
                    logit += coefficient * sample_prototype(
                        prototypes, plane_size, prototype_width, prototype_height,
                        channel, prototype_x, prototype_y);
                }
                mask[static_cast<size_t>(y) * mask_width + x] =
                    sigmoid(logit) >= 0.5f ? 255 : 0;
            }
        }
        result.masks.push_back(std::move(mask));
        instance.mask = result.masks.back().data();
        instance.mask_width = mask_width;
        instance.mask_height = mask_height;
        result.instances.push_back(instance);
    }
    return true;
}

} // namespace detection::yolo
