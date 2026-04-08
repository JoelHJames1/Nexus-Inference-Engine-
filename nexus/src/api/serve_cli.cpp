/// NEXUS CLI — `serve` subcommand: launch OpenAI-compatible HTTP API server.
///
/// Usage:
///   nexus serve <model.nxf> [options]
///
/// Options:
///   --port <n>          Port to listen on (default: 8080)
///   --ram-limit <gb>    RAM limit in GB (default: 48)
///   --no-metal          Disable Metal GPU acceleration
///   --threads <n>       Number of CPU threads (default: auto)

#include "core/engine.h"
#include "api/http_server.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

/// Global server pointer for signal handling.
static nexus::HttpServer* g_server = nullptr;

static void signal_handler(int sig) {
    (void)sig;
    fprintf(stderr, "\n[nexus-serve] Caught signal, shutting down...\n");
    if (g_server) {
        g_server->stop();
    }
}

int cmd_serve(int argc, char** argv) {
    if (argc < 1) {
        fprintf(stderr,
            "Usage: nexus serve <model.nxf> [options]\n"
            "\n"
            "Options:\n"
            "  --port <n>          Port to listen on (default: 8080)\n"
            "  --ram-limit <gb>    RAM limit in GB (default: 48)\n"
            "  --no-metal          Disable Metal GPU acceleration\n"
            "  --threads <n>       Number of CPU threads (default: auto)\n"
            "\n"
        );
        return 1;
    }

    std::string model_path = argv[0];
    int port = 8080;
    nexus::EngineConfig config;

    // Parse options.
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
            if (port <= 0 || port > 65535) {
                fprintf(stderr, "Error: invalid port number\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--ram-limit") == 0 && i + 1 < argc) {
            config.memory.ram_limit = static_cast<size_t>(
                atof(argv[++i]) * 1024 * 1024 * 1024);
        } else if (strcmp(argv[i], "--no-metal") == 0) {
            config.use_metal = false;
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            config.num_threads = atoi(argv[++i]);
        } else {
            fprintf(stderr, "Warning: unknown option '%s'\n", argv[i]);
        }
    }

    // Load the model.
    fprintf(stderr, "[nexus-serve] Loading model: %s\n", model_path.c_str());
    auto engine = nexus::Engine::create(model_path, config);
    if (!engine) {
        fprintf(stderr, "[nexus-serve] ERROR: Failed to load model: %s\n",
                model_path.c_str());
        return 1;
    }

    const auto& info = engine->model_info();
    fprintf(stderr, "[nexus-serve] Model loaded: %s (%s)\n",
            info.name.c_str(), info.architecture.c_str());
    fprintf(stderr, "[nexus-serve] Starting HTTP server on port %d...\n", port);

    // Create and start the server.
    nexus::HttpServer server(*engine, port);
    g_server = &server;

    // Install signal handlers for graceful shutdown.
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // start() blocks until stop() is called.
    server.start();

    g_server = nullptr;
    fprintf(stderr, "[nexus-serve] Bye.\n");
    return 0;
}
