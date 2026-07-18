#include "server_context.h"

#include "routes/common.h"
#include "routes/llm.h"
#include "routes/media.h"

#include <exception>

namespace forge::server {

bool ServerContext::initialize(const ServerConfig& config, std::string& error) {
    config_ = config;
    if (!registry_.initialize(config_, error)) return false;

    http_.set_payload_max_length(256ull * 1024 * 1024);
    http_.set_pre_routing_handler([this](const httplib::Request& request, httplib::Response& response) {
        if (config_.api_key.empty() || request.path == "/health") {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        const std::string authorization = request.get_header_value("Authorization");
        const std::string api_key = request.get_header_value("x-api-key");
        if (authorization == "Bearer " + config_.api_key || api_key == config_.api_key) {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        error_response(response, 401, "invalid or missing API key", "authentication_error");
        return httplib::Server::HandlerResponse::Handled;
    });
    http_.set_exception_handler([](const httplib::Request&, httplib::Response& response, std::exception_ptr error) {
        try {
            if (error) std::rethrow_exception(error);
        } catch (const std::exception& exception) {
            error_response(response, 500, exception.what(), "server_error");
            return;
        } catch (...) {
        }
        error_response(response, 500, "unknown server exception", "server_error");
    });
    register_common_routes(http_, registry_);
    register_llm_routes(http_, registry_);
    register_media_routes(http_, registry_);
    return true;
}

bool ServerContext::listen() { return http_.listen(config_.host, config_.port); }
void ServerContext::stop() { http_.stop(); }

} // namespace forge::server
