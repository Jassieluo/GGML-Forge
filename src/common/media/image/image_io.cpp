#include "image/image_io.h"

#include <fstream>
#include <limits>

#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STBI_NO_STDIO
#define STBI_MAX_DIMENSIONS 16384
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

namespace forge::media {

bool decode_image(const uint8_t* encoded, size_t encoded_size, Image& output,
                  std::string& error) {
    output = {};
    error.clear();
    if (!encoded || encoded_size == 0 ||
        encoded_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
        error = "encoded image buffer is empty or too large";
        return false;
    }

    int width = 0;
    int height = 0;
    int source_channels = 0;
    stbi_uc* decoded = stbi_load_from_memory(
        encoded, static_cast<int>(encoded_size), &width, &height, &source_channels, 3);
    if (!decoded || width <= 0 || height <= 0) {
        error = stbi_failure_reason() ? stbi_failure_reason() : "image decode failed";
        stbi_image_free(decoded);
        return false;
    }

    const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (pixel_count > std::numeric_limits<size_t>::max() / 3) {
        error = "decoded image dimensions overflow";
        stbi_image_free(decoded);
        return false;
    }
    output.width = static_cast<uint32_t>(width);
    output.height = static_cast<uint32_t>(height);
    output.pixels.assign(decoded, decoded + pixel_count * 3);
    stbi_image_free(decoded);
    return true;
}

bool load_image(const std::filesystem::path& path, Image& output, std::string& error) {
    output = {};
    error.clear();
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        error = "failed to open image file";
        return false;
    }
    const std::streamoff length = input.tellg();
    if (length <= 0 || static_cast<uintmax_t>(length) >
            static_cast<uintmax_t>(std::numeric_limits<size_t>::max())) {
        error = "image file is empty or too large";
        return false;
    }
    std::vector<uint8_t> encoded(static_cast<size_t>(length));
    input.seekg(0, std::ios::beg);
    if (!input.read(reinterpret_cast<char*>(encoded.data()), length)) {
        error = "failed to read image file";
        return false;
    }
    return decode_image(encoded.data(), encoded.size(), output, error);
}

} // namespace forge::media
