#pragma once

#include "providers/segmentation_provider.h"

#include <memory>
#include <string>

struct segmentation_runtime {
    segmentation::RuntimeConfig config;
};

struct segmentation_model {
    std::string provider_name;
    std::shared_ptr<segmentation::ISegmentationModel> implementation;
};

struct segmentation_session {
    std::shared_ptr<segmentation::ISegmentationModel> model;
    std::unique_ptr<segmentation::ISegmentationSession> implementation;
};
