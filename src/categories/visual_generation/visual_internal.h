#pragma once

#include "providers/visual_provider.h"

#include <memory>
#include <string>

struct visual_runtime {
    visual::RuntimeConfig config;
};

struct visual_model {
    std::string provider_name;
    std::shared_ptr<visual::IVisualModel> implementation;
};

struct visual_session {
    std::shared_ptr<visual::IVisualModel> model;
    std::unique_ptr<visual::IVisualSession> implementation;
};
