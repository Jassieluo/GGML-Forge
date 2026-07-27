#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace forge::media {

// Decoded, tightly packed RGB8 pixels. File formats and source channel counts
// are deliberately hidden from inference categories.
struct Image {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> pixels;
};

bool decode_image(const uint8_t* encoded, size_t encoded_size, Image& output,
                  std::string& error);
bool load_image(const std::filesystem::path& path, Image& output, std::string& error);
bool save_png(const std::filesystem::path& path, uint32_t width, uint32_t height,
              uint32_t channels, const uint8_t* pixels, std::string& error);

} // namespace forge::media
