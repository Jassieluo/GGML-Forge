#include "providers/gpt_sovits/dsp.h"
#include "providers/gpt_sovits/models/speaker_encoder/eres2net_v2.h"
#include "common/wav.h"

#include "ggml-backend.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4 && argc != 5) {
        std::cerr << "usage: test_speaker_encoder <model.gguf> <reference.wav> <expected.sv.bin> [expected.fbank.bin]\n";
        return 2;
    }
    const example::Audio source = example::load_wav(argv[2]);
    if (source.samples.empty()) return 1;
    std::vector<float> audio = source.sample_rate == 16000
        ? source.samples
        : gpt_sovits::dsp::resample_audio(
              source.samples.data(), source.samples.size(), source.sample_rate, 16000);
    int frames = 0;
    std::vector<float> fbank = gpt_sovits::dsp::compute_kaldi_fbank_80(
        audio.data(), audio.size(), frames);
    if (fbank.empty() || frames <= 0) return 1;
    if (argc == 5) {
        std::ifstream feature_input(argv[4], std::ios::binary | std::ios::ate);
        const std::streamsize bytes = feature_input.tellg();
        feature_input.seekg(0);
        const int expected_frames = static_cast<int>(bytes / (80 * sizeof(float)));
        std::vector<float> frame_major(static_cast<size_t>(expected_frames) * 80);
        if (bytes <= 0 || bytes % (80 * sizeof(float)) != 0 ||
            !feature_input.read(reinterpret_cast<char*>(frame_major.data()), bytes)) return 1;
        double feature_error = 0.0;
        const int compared_frames = std::min(frames, expected_frames);
        for (int frame = 0; frame < compared_frames; ++frame) {
            for (int mel = 0; mel < 80; ++mel) {
                const double error = fbank[static_cast<size_t>(mel) * frames + frame] -
                    frame_major[static_cast<size_t>(frame) * 80 + mel];
                feature_error += error * error;
            }
        }
        std::cout << "fbank_frames=" << frames << '/' << expected_frames
                  << ", fbank_rmse=" << std::sqrt(feature_error / (compared_frames * 80)) << '\n';
        frames = expected_frames;
        fbank.resize(frame_major.size());
        for (int frame = 0; frame < frames; ++frame) {
            for (int mel = 0; mel < 80; ++mel) {
                fbank[static_cast<size_t>(mel) * frames + frame] =
                    frame_major[static_cast<size_t>(frame) * 80 + mel];
            }
        }
    }

    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!backend) backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) return 1;
    gpt_sovits::ERes2NetV2 model;
    if (!model.load(argv[1], backend)) {
        ggml_backend_free(backend);
        return 1;
    }
    std::vector<float> actual;
    gpt_sovits::ERes2NetV2Runner runner(model, backend);
    const bool encoded = runner.encode(fbank, frames, actual);

    std::ifstream input(argv[3], std::ios::binary);
    std::vector<float> expected(20480);
    const bool expected_ok = static_cast<bool>(
        input.read(reinterpret_cast<char*>(expected.data()), expected.size() * sizeof(float)));
    double dot = 0.0;
    double actual_norm = 0.0;
    double expected_norm = 0.0;
    double squared_error = 0.0;
    double max_error = 0.0;
    if (encoded && expected_ok && actual.size() == expected.size()) {
        for (size_t i = 0; i < actual.size(); ++i) {
            dot += static_cast<double>(actual[i]) * expected[i];
            actual_norm += static_cast<double>(actual[i]) * actual[i];
            expected_norm += static_cast<double>(expected[i]) * expected[i];
            const double error = actual[i] - expected[i];
            squared_error += error * error;
            max_error = std::max(max_error, std::abs(error));
        }
    }
    const double cosine = dot / std::sqrt(actual_norm * expected_norm);
    const double rmse = std::sqrt(squared_error / expected.size());
    std::cout << "frames=" << frames << ", cosine=" << cosine
              << ", rmse=" << rmse << ", max_error=" << max_error << '\n';
    ggml_backend_free(backend);
    return encoded && expected_ok && actual.size() == expected.size() && cosine > 0.999 ? 0 : 1;
}
