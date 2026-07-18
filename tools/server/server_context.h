#pragma once

#include "httplib.h"
#include "model_registry.h"
#include "server_config.h"

#include <string>

namespace forge::server {

class ServerContext {
public:
    bool initialize(const ServerConfig& config, std::string& error);
    bool listen();
    void stop();
    ModelRegistry& registry() { return registry_; }

private:
    ServerConfig config_;
    ModelRegistry registry_;
    httplib::Server http_;
};

} // namespace forge::server
