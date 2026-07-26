#pragma once

#include "providers/detection_provider.h"

#include <memory>
#include <string>

struct detection_runtime {
    detection::RuntimeConfig config;
};

struct detection_model {
    std::string provider_name;
    std::shared_ptr<detection::IDetectionModel> implementation;
};

struct detection_session {
    std::shared_ptr<detection::IDetectionModel> model;
    std::unique_ptr<detection::IDetectionSession> implementation;
};
