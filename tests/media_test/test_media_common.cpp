#include "audio/audio_io.h"
#include "audio/resample.h"
#include "image/image_io.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

} // namespace

int main() {
    // A self-contained 1x1 RGBA PNG; the decoder normalizes it to RGB8.
    static constexpr uint8_t png[] = {
        0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a,0x00,0x00,0x00,0x0d,
        0x49,0x48,0x44,0x52,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x01,
        0x08,0x06,0x00,0x00,0x00,0x1f,0x15,0xc4,0x89,0x00,0x00,0x00,
        0x0d,0x49,0x44,0x41,0x54,0x08,0x1d,0x63,0xf8,0xcf,0xc0,0xf0,
        0x1f,0x00,0x05,0x80,0x02,0x3f,0x49,0xc2,0xfc,0xe1,0x00,0x00,
        0x00,0x00,0x49,0x45,0x4e,0x44,0xae,0x42,0x60,0x82,
    };
    forge::media::Image image;
    std::string error;
    expect(forge::media::decode_image(png, sizeof(png), image, error), "decode embedded PNG");
    expect(image.width == 1 && image.height == 1 && image.pixels.size() == 3,
           "PNG is normalized to tightly packed RGB8");
    const uint8_t invalid_image[] = {1, 2, 3, 4};
    expect(!forge::media::decode_image(invalid_image, sizeof(invalid_image), image, error) &&
               !error.empty(), "invalid image reports an error");

    forge::media::Audio source;
    source.sample_rate = 16000;
    source.channels = 2;
    source.samples = {-1.0f, 1.0f, -0.5f, 0.5f, 0.25f, 0.75f};
    std::vector<uint8_t> wav;
    expect(forge::media::encode_wav_pcm16(source, wav, error), "encode stereo PCM16 WAV");
    forge::media::Audio decoded;
    expect(forge::media::decode_wav(wav.data(), wav.size(), decoded, error), "decode PCM16 WAV");
    expect(decoded.sample_rate == source.sample_rate && decoded.channels == source.channels &&
               decoded.samples.size() == source.samples.size(), "WAV format round trip");
    const std::vector<float> mono = forge::media::mix_to_mono(decoded);
    expect(mono.size() == 3 && std::abs(mono[0]) < 1e-4f &&
               std::abs(mono[1]) < 1e-4f && std::abs(mono[2] - 0.5f) < 1e-4f,
           "stereo downmix");
    expect(!forge::media::decode_wav(invalid_image, sizeof(invalid_image), decoded, error) &&
               !error.empty(), "invalid WAV reports an error");

    std::vector<float> tone(80, 0.25f);
    const std::vector<float> resampled = forge::media::resample_mono(
        tone.data(), tone.size(), 8000, 16000);
    expect(resampled.size() == 160, "resampler output length");
    bool finite = !resampled.empty();
    for (float value : resampled) finite = finite && std::isfinite(value);
    expect(finite, "resampler output is finite");
    tone[0] = NAN;
    expect(!forge::media::valid_mono_audio(tone.data(), tone.size(), 8000),
           "non-finite audio is rejected");

    if (failures == 0) std::cout << "common media tests PASSED\n";
    return failures == 0 ? 0 : 1;
}
