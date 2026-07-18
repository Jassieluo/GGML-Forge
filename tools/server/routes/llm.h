#pragma once

#include "httplib.h"
#include "model_registry.h"

namespace forge::server {

void register_llm_routes(httplib::Server& server, ModelRegistry& registry);

} // namespace forge::server
