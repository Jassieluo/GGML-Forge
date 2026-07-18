#pragma once

#include "httplib.h"
#include "model_registry.h"
#include "protocols/chat.h"

#include <string>

namespace forge::server {

bool parse_json_body(const httplib::Request& request, Json& body, httplib::Response& response);
void json_response(httplib::Response& response, const Json& body, int status = 200);
void error_response(
    httplib::Response& response,
    int status,
    const std::string& message,
    const std::string& type = "invalid_request_error");
void register_common_routes(httplib::Server& server, ModelRegistry& registry);

} // namespace forge::server
