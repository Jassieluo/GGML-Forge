#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

namespace tts {

enum class ComponentResidency {
    resident,
    on_demand,
};

struct ComponentRuntimePolicy {
    std::string device;
    ComponentResidency residency = ComponentResidency::resident;
};

struct RuntimeContext {
    uint32_t n_threads = 1;
    uint32_t max_concurrency = 1;
    std::string device_name = "cpu";
    std::unordered_map<std::string, ComponentRuntimePolicy> components;
};

} // namespace tts
