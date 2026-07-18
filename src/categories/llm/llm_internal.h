#pragma once

#include "providers/llm_provider.h"

#include <memory>
#include <string>

struct llm_runtime {
    llm::RuntimeConfig config;
};

struct llm_model {
    std::string provider_name;
    std::shared_ptr<llm::ILLMModel> implementation;
};

struct llm_session {
    std::shared_ptr<llm::ILLMModel> model;
    std::unique_ptr<llm::ILLMSession> implementation;
};
