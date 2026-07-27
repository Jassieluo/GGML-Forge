#include "image/preprocess.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace forge::media {
namespace {

struct Contributors {
    std::vector<uint32_t> indices;
    std::vector<float> weights;
};

Contributors contributors(uint32_t source_size, uint32_t resized_size,
                          uint32_t resized_index) {
    const float scale = static_cast<float>(source_size) / resized_size;
    const float support = std::max(1.0f, scale);
    const float center = (resized_index + 0.5f) * scale;
    const int first = std::max(
        0, static_cast<int>(std::floor(center - support - 0.5f)) + 1);
    const int last = std::min(
        static_cast<int>(source_size),
        static_cast<int>(std::ceil(center + support - 0.5f)));
    Contributors result;
    float total = 0.0f;
    for (int index = first; index < last; ++index) {
        const float weight = std::max(
            0.0f, 1.0f - std::abs((index + 0.5f - center) / support));
        if (weight == 0.0f) continue;
        result.indices.push_back(static_cast<uint32_t>(index));
        result.weights.push_back(weight);
        total += weight;
    }
    if (total > 0.0f) {
        for (float& weight : result.weights) weight /= total;
    }
    return result;
}

} // namespace

bool resize_rgb8_planar(
    const uint8_t* source, uint32_t width, uint32_t height, uint32_t channels,
    uint32_t output_width, uint32_t output_height, std::vector<float>& output) {
    output.clear();
    if (!source || width == 0 || height == 0 || output_width == 0 || output_height == 0 ||
        (channels != 1 && channels != 3 && channels != 4) ||
        static_cast<size_t>(output_width) >
            std::numeric_limits<size_t>::max() / output_height / 3 ||
        static_cast<size_t>(height) >
            std::numeric_limits<size_t>::max() / output_width / 3) return false;
    std::vector<Contributors> horizontal(output_width);
    std::vector<Contributors> vertical(output_height);
    for (uint32_t x = 0; x < output_width; ++x) horizontal[x] = contributors(width, output_width, x);
    for (uint32_t y = 0; y < output_height; ++y) vertical[y] = contributors(height, output_height, y);
    std::vector<float> rows(static_cast<size_t>(height) * output_width * 3);
    for (uint32_t source_y = 0; source_y < height; ++source_y) {
        for (uint32_t x = 0; x < output_width; ++x) {
            for (uint32_t channel = 0; channel < 3; ++channel) {
                float value = 0.0f;
                for (size_t item = 0; item < horizontal[x].indices.size(); ++item) {
                    const uint32_t source_channel = channels == 1 ? 0 : channel;
                    const size_t offset =
                        (static_cast<size_t>(source_y) * width + horizontal[x].indices[item]) *
                        channels + source_channel;
                    value += source[offset] * horizontal[x].weights[item];
                }
                rows[(static_cast<size_t>(source_y) * output_width + x) * 3 + channel] = value;
            }
        }
    }
    output.resize(static_cast<size_t>(output_width) * output_height * 3);
    for (uint32_t y = 0; y < output_height; ++y) {
        for (uint32_t x = 0; x < output_width; ++x) {
            for (uint32_t channel = 0; channel < 3; ++channel) {
                float value = 0.0f;
                for (size_t item = 0; item < vertical[y].indices.size(); ++item) {
                    value += rows[(static_cast<size_t>(vertical[y].indices[item]) * output_width + x) *
                                  3 + channel] * vertical[y].weights[item];
                }
                output[x + static_cast<size_t>(output_width) *
                    (y + static_cast<size_t>(output_height) * channel)] = value / 255.0f;
            }
        }
    }
    return true;
}

bool resize_shortest_normalized_rgb8(
    const uint8_t* source, uint32_t width, uint32_t height, uint32_t channels,
    uint32_t shortest_edge, const float mean[3], const float stddev[3],
    uint32_t& output_width, uint32_t& output_height, std::vector<float>& output) {
    output.clear();
    output_width = 0;
    output_height = 0;
    if (!source || !mean || !stddev || width == 0 || height == 0 || shortest_edge == 0 ||
        (channels != 1 && channels != 3 && channels != 4) ||
        !(stddev[0] > 0.0f) || !(stddev[1] > 0.0f) || !(stddev[2] > 0.0f)) return false;
    uint64_t resized_width = shortest_edge;
    uint64_t resized_height = shortest_edge;
    if (width < height) resized_height = static_cast<uint64_t>(shortest_edge) * height / width;
    if (height < width) resized_width = static_cast<uint64_t>(shortest_edge) * width / height;
    if (resized_width > std::numeric_limits<uint32_t>::max() ||
        resized_height > std::numeric_limits<uint32_t>::max() ||
        resized_width > std::numeric_limits<size_t>::max() / resized_height / 3) return false;
    output_width = static_cast<uint32_t>(resized_width);
    output_height = static_cast<uint32_t>(resized_height);

    std::vector<Contributors> horizontal(output_width);
    std::vector<Contributors> vertical(output_height);
    for (uint32_t x = 0; x < output_width; ++x) horizontal[x] = contributors(width, output_width, x);
    for (uint32_t y = 0; y < output_height; ++y) vertical[y] = contributors(height, output_height, y);
    if (static_cast<size_t>(height) >
        std::numeric_limits<size_t>::max() / output_width / 3) return false;
    std::vector<float> rows(static_cast<size_t>(height) * output_width * 3);
    for (uint32_t source_y = 0; source_y < height; ++source_y) {
        for (uint32_t x = 0; x < output_width; ++x) {
            for (uint32_t channel = 0; channel < 3; ++channel) {
                float value = 0.0f;
                for (size_t item = 0; item < horizontal[x].indices.size(); ++item) {
                    const uint32_t source_channel = channels == 1 ? 0 : channel;
                    const size_t offset =
                        (static_cast<size_t>(source_y) * width + horizontal[x].indices[item]) *
                        channels + source_channel;
                    value += source[offset] * horizontal[x].weights[item];
                }
                rows[(static_cast<size_t>(source_y) * output_width + x) * 3 + channel] = value;
            }
        }
    }
    output.resize(static_cast<size_t>(output_width) * output_height * 3);
    for (uint32_t y = 0; y < output_height; ++y) {
        for (uint32_t x = 0; x < output_width; ++x) {
            for (uint32_t channel = 0; channel < 3; ++channel) {
                float value = 0.0f;
                for (size_t item = 0; item < vertical[y].indices.size(); ++item) {
                    value += rows[(static_cast<size_t>(vertical[y].indices[item]) * output_width + x) *
                                  3 + channel] * vertical[y].weights[item];
                }
                output[x + static_cast<size_t>(output_width) *
                    (y + static_cast<size_t>(output_height) * channel)] =
                    (value / 255.0f - mean[channel]) / stddev[channel];
            }
        }
    }
    return true;
}

bool resize_shortest_center_crop_rgb8(
    const uint8_t* source, uint32_t width, uint32_t height, uint32_t channels,
    uint32_t target_size, std::vector<float>& output) {
    output.clear();
    if (!source || width == 0 || height == 0 || target_size == 0 ||
        (channels != 1 && channels != 3 && channels != 4)) return false;
    if (static_cast<size_t>(target_size) >
        std::numeric_limits<size_t>::max() / target_size / 3) return false;

    uint64_t resized_width_64 = target_size;
    uint64_t resized_height_64 = target_size;
    if (width < height) {
        resized_height_64 = static_cast<uint64_t>(target_size) * height / width;
    } else if (height < width) {
        resized_width_64 = static_cast<uint64_t>(target_size) * width / height;
    }
    if (resized_width_64 > std::numeric_limits<uint32_t>::max() ||
        resized_height_64 > std::numeric_limits<uint32_t>::max()) return false;
    if (static_cast<size_t>(height) >
        std::numeric_limits<size_t>::max() / target_size / 3) return false;
    const uint32_t resized_width = static_cast<uint32_t>(resized_width_64);
    const uint32_t resized_height = static_cast<uint32_t>(resized_height_64);
    const uint32_t crop_x = (resized_width - target_size) / 2;
    const uint32_t crop_y = (resized_height - target_size) / 2;

    std::vector<Contributors> horizontal;
    std::vector<Contributors> vertical;
    horizontal.reserve(target_size);
    vertical.reserve(target_size);
    for (uint32_t x = 0; x < target_size; ++x) {
        horizontal.push_back(contributors(width, resized_width, crop_x + x));
    }
    for (uint32_t y = 0; y < target_size; ++y) {
        vertical.push_back(contributors(height, resized_height, crop_y + y));
    }

    std::vector<float> rows(static_cast<size_t>(height) * target_size * 3);
    for (uint32_t source_y = 0; source_y < height; ++source_y) {
        for (uint32_t x = 0; x < target_size; ++x) {
            const Contributors& filter = horizontal[x];
            for (uint32_t channel = 0; channel < 3; ++channel) {
                float value = 0.0f;
                for (size_t item = 0; item < filter.indices.size(); ++item) {
                    const uint32_t source_channel = channels == 1 ? 0 : channel;
                    const size_t offset =
                        (static_cast<size_t>(source_y) * width + filter.indices[item]) *
                        channels + source_channel;
                    value += source[offset] * filter.weights[item];
                }
                rows[(static_cast<size_t>(source_y) * target_size + x) * 3 + channel] =
                    value / 255.0f;
            }
        }
    }

    output.resize(static_cast<size_t>(target_size) * target_size * 3);
    for (uint32_t y = 0; y < target_size; ++y) {
        const Contributors& filter = vertical[y];
        for (uint32_t x = 0; x < target_size; ++x) {
            for (uint32_t channel = 0; channel < 3; ++channel) {
                float value = 0.0f;
                for (size_t item = 0; item < filter.indices.size(); ++item) {
                    value += rows[(static_cast<size_t>(filter.indices[item]) * target_size + x) *
                                  3 + channel] * filter.weights[item];
                }
                output[x + static_cast<size_t>(target_size) *
                    (y + static_cast<size_t>(target_size) * channel)] = value;
            }
        }
    }
    return true;
}

} // namespace forge::media
