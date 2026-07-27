#pragma once

#include "providers/classification_provider.h"

#include <memory>
#include <string>

struct classification_runtime {
    visual_perception::classification::RuntimeConfig config;
};

struct classification_model {
    std::string provider_name;
    std::shared_ptr<visual_perception::classification::IClassificationModel> implementation;
};

struct classification_session {
    std::shared_ptr<visual_perception::classification::IClassificationModel> model;
    std::unique_ptr<visual_perception::classification::IClassificationSession> implementation;
};
