#include "models/synthesis/vits/vits.h"
#include "nn/io/gguf.h"
#include "nn/state_dict.h"

#include "ggml-backend.h"

#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>

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
    const nn::Parameter* conv_pre = model->find_parameter("generator.pre.weight");
    nn::io::GGUFSource source(argv[1]);
    if (!model->state_dict() || model->state_dict()->size() != source.size() ||
        !conv_pre || !conv_pre->is_bound()) {
        std::cerr << "VITS parameter ownership validation failed\n";
        model.reset();
        ggml_backend_free(backend);
        return 1;
    }
    std::unordered_set<const ggml_tensor*> module_tensors;
    size_t unbound_module_parameters = 0;
    model->for_each_parameter([&](std::string_view, const nn::Parameter& parameter) {
        if (parameter.is_bound()) module_tensors.insert(parameter.tensor());
        else if (parameter.is_required()) ++unbound_module_parameters;
    });
    std::vector<std::string> unconsumed;
    for (size_t index = 0; index < source.size(); ++index) {
        const std::string& name = source.info(index).name;
        const nn::Parameter* parameter = model->find_parameter(name);
        if (!parameter || !parameter->is_bound() || module_tensors.find(parameter->tensor()) == module_tensors.end()) {
            unconsumed.push_back(name);
        }
    }
    if (unbound_module_parameters != 0 || !unconsumed.empty()) {
        std::cerr << "VITS canonical parameter validation failed\n";
        model.reset();
        ggml_backend_free(backend);
        return 1;
    }
    std::cout << "VITS loading test passed: " << model->profile.exact_version
              << ", artifact_tensors=" << source.size()
              << ", module_parameters=" << model->parameter_count()
              << ", required_unbound=" << unbound_module_parameters
              << ", artifact_unconsumed=" << unconsumed.size() << "\n";
    for (const std::string& name : unconsumed) std::cout << "  unconsumed: " << name << "\n";
    model.reset();
    ggml_backend_free(backend);
    return 0;
}
