#pragma once

#include "providers/instance_provider.h"

#include <memory>
#include <string>

struct instance_runtime {
    visual_perception::instance::RuntimeConfig config;
};

struct instance_model {
    std::string provider_name;
    std::shared_ptr<visual_perception::instance::IInstanceModel> implementation;
};

struct instance_session {
    std::shared_ptr<visual_perception::instance::IInstanceModel> model;
    std::unique_ptr<visual_perception::instance::IInstanceSession> implementation;
};
