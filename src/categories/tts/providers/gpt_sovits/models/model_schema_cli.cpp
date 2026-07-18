#include "providers/gpt_sovits/models/t2s/gpt_t2s.h"
#include "providers/gpt_sovits/models/hubert/hubert.h"
#include "providers/gpt_sovits/models/vits/vits.h"
#include "providers/gpt_sovits/models/bert/bert.h"

#include <cstdlib>
#include <iostream>
#include <memory>
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

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            throw std::invalid_argument(
                "usage: nn-model-schema <bert|hubert|t2s|vits> [topology or version]");
        }
        const std::string architecture = argv[1];
        if (architecture == "bert") {
            if (argc != 2) throw std::invalid_argument("bert schema takes no topology arguments");
            gpt_sovits::BertModel model;
            std::cout << model.schema().to_json() << '\n';
            return 0;
        }
        if (architecture == "hubert") {
            if (argc != 2) throw std::invalid_argument("hubert schema takes no topology arguments");
            gpt_sovits::HubertModel model;
            std::cout << model.schema().to_json() << '\n';
            return 0;
        }
        if (architecture == "t2s") {
            if (argc != 5) {
                throw std::invalid_argument("t2s schema requires: layers heads hidden_dim");
            }
            const int layers = positive_int(argv[2], "layers");
            const int heads = positive_int(argv[3], "heads");
            const int hidden_dim = positive_int(argv[4], "hidden_dim");
            if (hidden_dim % heads != 0) {
                throw std::invalid_argument("hidden_dim must be divisible by heads");
            }
            gpt_sovits::T2SModel model;
            model.decoder.reset(
                layers, heads, hidden_dim / heads,
                nn::ActivationType::RELU, 1e-5f);
            std::cout << model.schema().to_json() << '\n';
            return 0;
        }
        if (architecture == "vits") {
            if (argc != 3) throw std::invalid_argument("vits schema requires an exact version");
            std::unique_ptr<gpt_sovits::VITSModel> model =
                gpt_sovits::VITSModel::create_for_version(argv[2]);
            if (!model) throw std::invalid_argument("unsupported VITS version");
            std::cout << model->schema().to_json() << '\n';
            return 0;
        }
        throw std::invalid_argument("unsupported architecture: " + architecture);
    } catch (const std::exception& error) {
        std::cerr << "nn-model-schema: " << error.what() << '\n';
        return 2;
    }
}
