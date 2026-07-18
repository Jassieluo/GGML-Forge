#include "server_context.h"

#include <iostream>

int main(int argc, char** argv) {
    forge::server::ServerConfig config;
    std::string error;
    if (!forge::server::parse_server_config(argc, argv, config, error)) {
        if (error.empty()) return 0;
        std::cerr << "forge-server: " << error << "\n\n";
        forge::server::print_usage(std::cerr, argv[0]);
        return 2;
    }
    forge::server::ServerContext server;
    if (!server.initialize(config, error)) {
        std::cerr << "forge-server: " << error << '\n';
        return 1;
    }
    std::cout << "forge-server listening on http://" << config.host << ':' << config.port << '\n';
    for (const forge::server::ServedModel& model : server.registry().models()) {
        std::cout << "  " << model.id << " [" << model.category << "] via " << model.provider << '\n';
    }
    return server.listen() ? 0 : 1;
}
