#ifdef FORGE_SCHEMA_HAS_TTS
#include "providers/gpt_sovits/models/bert/bert.h"
#include "providers/gpt_sovits/models/hubert/hubert.h"
#include "providers/gpt_sovits/models/speaker_encoder/eres2net_v2.h"
#include "providers/gpt_sovits/models/t2s/gpt_t2s.h"
#include "providers/gpt_sovits/models/vits/vits.h"
#endif
#ifdef FORGE_SCHEMA_HAS_YOLO
#include "providers/yolo/v8/yolo_v8.h"
#endif

#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

int positive_int(const char* value, const char* name) {
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (!value[0] || !end || *end != '\0' || parsed <= 0 || parsed > 1000000) {
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }
    return static_cast<int>(parsed);
}

template <size_t Size>
std::array<int, Size> positive_csv(const char* value, const char* name, bool allow_zero) {
    std::array<int, Size> output{};
    std::istringstream stream(value);
    std::string item;
    size_t index = 0;
    while (std::getline(stream, item, ',')) {
        if (index >= Size) throw std::invalid_argument(std::string(name) + " has too many values");
        char* end = nullptr;
        const long parsed = std::strtol(item.c_str(), &end, 10);
        if (item.empty() || !end || *end != '\0' || parsed < (allow_zero ? 0 : 1) ||
            parsed > 1000000) {
            throw std::invalid_argument(std::string(name) + " contains an invalid value");
        }
        output[index++] = static_cast<int>(parsed);
    }
    if (index != Size) throw std::invalid_argument(std::string(name) + " has the wrong value count");
    return output;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::invalid_argument("usage: nn-model-schema <architecture> [topology]");
        const std::string architecture = argv[1];
#ifdef FORGE_SCHEMA_HAS_TTS
        if (architecture == "bert") {
            if (argc != 2) throw std::invalid_argument("bert schema takes no topology arguments");
            gpt_sovits::BertModel model;
            std::cout << model.schema().to_json() << '\n'; return 0;
        }
        if (architecture == "hubert") {
            if (argc != 2) throw std::invalid_argument("hubert schema takes no topology arguments");
            gpt_sovits::HubertModel model;
            std::cout << model.schema().to_json() << '\n'; return 0;
        }
        if (architecture == "speaker_encoder") {
            if (argc != 2) throw std::invalid_argument("speaker_encoder schema takes no topology arguments");
            gpt_sovits::ERes2NetV2 model;
            std::cout << model.schema().to_json() << '\n'; return 0;
        }
        if (architecture == "t2s") {
            if (argc != 5) throw std::invalid_argument("t2s schema requires: layers heads hidden_dim");
            const int layers = positive_int(argv[2], "layers");
            const int heads = positive_int(argv[3], "heads");
            const int hidden = positive_int(argv[4], "hidden_dim");
            if (hidden % heads != 0) throw std::invalid_argument("hidden_dim must be divisible by heads");
            gpt_sovits::T2SModel model;
            model.decoder.reset(layers, heads, hidden / heads, nn::ActivationType::RELU, 1e-5f);
            std::cout << model.schema().to_json() << '\n'; return 0;
        }
        if (architecture == "vits") {
            if (argc != 3) throw std::invalid_argument("vits schema requires an exact version");
            std::unique_ptr<gpt_sovits::VITSModel> model =
                gpt_sovits::VITSModel::create_for_version(argv[2]);
            if (!model) throw std::invalid_argument("unsupported VITS version");
            std::cout << model->schema().to_json() << '\n'; return 0;
        }
#endif
#ifdef FORGE_SCHEMA_HAS_YOLO
        if (architecture == "yolo_v8" || architecture == "yolo_v8_seg") {
            const bool segmentation = architecture == "yolo_v8_seg";
            if (argc != (segmentation ? 12 : 9)) {
                throw std::invalid_argument(
                    "yolo_v8 schema requires: classes reg_max box_channels class_channels "
                    "out_channels_csv hidden_channels_csv repeats_csv"
                    " [mask_count mask_channels prototype_channels]");
            }
            detection::yolo::v8::Config config;
            config.class_count = positive_int(argv[2], "classes");
            config.reg_max = positive_int(argv[3], "reg_max");
            config.detect_box_channels = positive_int(argv[4], "box_channels");
            config.detect_class_channels = positive_int(argv[5], "class_channels");
            config.out_channels = positive_csv<23>(argv[6], "out_channels", true);
            config.hidden_channels = positive_csv<23>(argv[7], "hidden_channels", true);
            config.repeats = positive_csv<23>(argv[8], "repeats", true);
            if (segmentation) {
                config.task = detection::yolo::v8::Task::instance_segmentation;
                config.mask_count = positive_int(argv[9], "mask_count");
                config.mask_channels = positive_int(argv[10], "mask_channels");
                config.prototype_channels = positive_int(argv[11], "prototype_channels");
            }
            if (!config.valid()) throw std::invalid_argument("invalid yolo_v8 topology");
            detection::yolo::v8::Model model(config);
            std::cout << model.schema().to_json() << '\n'; return 0;
        }
#endif
        throw std::invalid_argument("unsupported architecture: " + architecture);
    } catch (const std::exception& error) {
        std::cerr << "nn-model-schema: " << error.what() << '\n';
        return 2;
    }
}
