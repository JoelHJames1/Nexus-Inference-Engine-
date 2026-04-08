#pragma once
/// NEXUS HTTP API Server — OpenAI-compatible REST API.
///
/// Provides endpoints:
///   POST /v1/chat/completions   Chat completion (streaming + non-streaming)
///   POST /v1/completions        Text completion (streaming + non-streaming)
///   GET  /v1/models             List loaded model
///   GET  /health                Health check
///
/// Thread-per-connection model using POSIX sockets. No external dependencies.

#include "core/engine.h"
#include <atomic>
#include <string>
#include <thread>
#include <vector>
#include <mutex>

namespace nexus {

class HttpServer {
public:
    /// Construct a server bound to the given engine and port.
    explicit HttpServer(Engine& engine, int port = 8080);
    ~HttpServer();

    /// Start listening. Blocks until stop() is called from another thread.
    void start();

    /// Signal the server to stop and close the listening socket.
    void stop();

    /// Returns true if the server is currently running.
    bool is_running() const { return running_.load(); }

    /// Returns the port the server is listening on.
    int port() const { return port_; }

private:
    // ── HTTP parsing ────────────────────────────────────────────────────────
    struct HttpRequest {
        std::string method;         // "GET", "POST", etc.
        std::string path;           // "/v1/chat/completions"
        std::string body;           // Request body (for POST)
        int content_length = 0;
        bool keep_alive = false;
    };

    // ── Minimal JSON helpers ────────────────────────────────────────────────
    struct ChatMessage {
        std::string role;
        std::string content;
    };

    struct CompletionRequest {
        std::string model;
        std::vector<ChatMessage> messages;   // For chat completions
        std::string prompt;                  // For text completions
        float temperature = 0.7f;
        float top_p = 0.9f;
        int   top_k = 40;
        int   max_tokens = 2048;
        bool  stream = false;
    };

    // ── Request handling ────────────────────────────────────────────────────
    void accept_loop();
    void handle_connection(int client_fd);
    bool parse_request(int client_fd, HttpRequest& req);

    // ── Route handlers ──────────────────────────────────────────────────────
    void handle_chat_completions(int client_fd, const HttpRequest& req);
    void handle_completions(int client_fd, const HttpRequest& req);
    void handle_list_models(int client_fd);
    void handle_health(int client_fd);
    void handle_options(int client_fd);
    void send_error(int client_fd, int status, const char* message);

    // ── JSON parsing (hand-rolled) ──────────────────────────────────────────
    bool parse_completion_request(const std::string& json, CompletionRequest& req);
    static std::string json_extract_string(const std::string& json, const std::string& key);
    static double      json_extract_number(const std::string& json, const std::string& key, double def);
    static bool        json_extract_bool(const std::string& json, const std::string& key, bool def);
    static bool        parse_messages(const std::string& json, std::vector<ChatMessage>& msgs);

    // ── JSON output ─────────────────────────────────────────────────────────
    std::string build_chat_response(const std::string& id,
                                    const std::string& content,
                                    const std::string& finish_reason,
                                    int prompt_tokens, int completion_tokens);
    std::string build_completion_response(const std::string& id,
                                          const std::string& text,
                                          const std::string& finish_reason,
                                          int prompt_tokens, int completion_tokens);
    std::string build_models_response();

    // ── SSE streaming helpers ───────────────────────────────────────────────
    void send_sse_headers(int client_fd);
    void send_sse_chat_delta(int client_fd, const std::string& id, const std::string& content);
    void send_sse_chat_done(int client_fd, const std::string& id, const std::string& finish_reason);
    void send_sse_completion_delta(int client_fd, const std::string& id, const std::string& text);
    void send_sse_completion_done(int client_fd, const std::string& id, const std::string& finish_reason);

    // ── HTTP response helpers ───────────────────────────────────────────────
    void send_response(int client_fd, int status, const std::string& content_type,
                       const std::string& body);
    void send_raw(int client_fd, const std::string& data);
    static std::string generate_id();
    static std::string json_escape(const std::string& s);

    // ── State ───────────────────────────────────────────────────────────────
    Engine& engine_;
    int port_;
    int listen_fd_ = -1;
    std::atomic<bool> running_{false};
    std::mutex connections_mutex_;
    std::vector<std::thread> connection_threads_;
};

}  // namespace nexus
