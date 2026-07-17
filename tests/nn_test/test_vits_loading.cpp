#include "models/synthesis/vits/vits.h"
#include "nn/state_dict.h"

#include "ggml-backend.h"

#include <iostream>
#include <memory>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: test_vits_loading <model.gguf>\n";
        return 2;
    }
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        std::cerr << "failed to initialize CPU backend\n";
        return 1;
    }
    std::unique_ptr<gpt_sovits::VITSModel> model = gpt_sovits::VITSModel::create(argv[1]);
    if (!model || !model->load(argv[1], backend)) {
        std::cerr << "VITS loading test failed\n";
        ggml_backend_free(backend);
        return 1;
    }
    const nn::Parameter* conv_pre = model->artifact_parameters.find("dec.conv_pre.weight");
    if (!model->state_dict() || model->state_dict()->size() != model->artifact_parameters.size() ||
        !conv_pre || !conv_pre->is_bound()) {
        std::cerr << "VITS parameter ownership validation failed\n";
        model.reset();
        ggml_backend_free(backend);
        return 1;
    }
    std::cout << "VITS loading test passed: " << model->profile.exact_version
              << ", parameters=" << model->artifact_parameters.size() << "\n";
    model.reset();
    ggml_backend_free(backend);
    return 0;
}
