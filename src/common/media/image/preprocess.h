#pragma once

#include <cstdint>
#include <vector>

namespace forge::media {

// Resize the shortest edge to target_size with an antialiased triangle filter,
// then take the centered target_size square. Output is normalized planar RGB.
bool resize_shortest_center_crop_rgb8(
    const uint8_t* source, uint32_t width, uint32_t height, uint32_t channels,
    uint32_t target_size, std::vector<float>& output);

} // namespace forge::media
