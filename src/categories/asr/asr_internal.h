#pragma once

#include "providers/asr_provider.h"

#include <memory>
#include <string>

struct asr_runtime {
    asr::RuntimeConfig config;
};

struct asr_model {
    std::string provider_name;
    std::shared_ptr<asr::IASRModel> implementation;
};

struct asr_session {
    std::shared_ptr<asr::IASRModel> model;
    std::unique_ptr<asr::IASRSession> implementation;
};
