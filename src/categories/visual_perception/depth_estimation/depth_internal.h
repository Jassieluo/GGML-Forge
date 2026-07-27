#pragma once

#include "providers/depth_provider.h"

#include <memory>
#include <string>

struct depth_runtime {
    depth::RuntimeConfig config;
};

struct depth_model {
    std::string provider_name;
    std::shared_ptr<depth::IDepthModel> implementation;
};

struct depth_session {
    std::shared_ptr<depth::IDepthModel> model;
    std::unique_ptr<depth::IDepthSession> implementation;
};
