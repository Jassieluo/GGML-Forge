#pragma once

#include <cstdint>
#include <vector>

namespace forge::media {

// Resize the shortest edge to target_size with an antialiased triangle filter,
// then take the centered target_size square. Output is normalized planar RGB.
bool resize_shortest_center_crop_rgb8(
    const uint8_t* source, uint32_t width, uint32_t height, uint32_t channels,
    uint32_t target_size, std::vector<float>& output);

// Resize the shortest edge while retaining aspect ratio. Output is planar RGB
// normalized with (value / 255 - mean[channel]) / std[channel].
bool resize_shortest_normalized_rgb8(
    const uint8_t* source, uint32_t width, uint32_t height, uint32_t channels,
    uint32_t shortest_edge, const float mean[3], const float stddev[3],
    uint32_t& output_width, uint32_t& output_height, std::vector<float>& output);

} // namespace forge::media
