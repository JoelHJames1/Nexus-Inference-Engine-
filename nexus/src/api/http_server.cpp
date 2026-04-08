/// NEXUS HTTP API Server — OpenAI-compatible REST API implementation.
///
/// Phase 5 MVP: Thread-per-connection, POSIX sockets, hand-rolled JSON.

#include "api/http_server.h"

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <signal.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <random>
#include <sstream>

namespace nexus {

// ─── Construction / destruction ─────────────────────────────────────────────

HttpServer::HttpServer(Engine& engine, int port)
    : engine_(engine), port_(port) {}

HttpServer::~HttpServer() {
    stop();
}

// ─── Lifecycle ──────────────────────────────────────────────────────────────

void HttpServer::start() {
    // Ignore SIGPIPE so writes to closed sockets don't crash us.
    signal(SIGPIPE, SIG_IGN);

    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        fprintf(stderr, "[nexus-http] ERROR: socket(): %s\n", strerror(errno));
        return;
    }

    int opt = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(static_cast<uint16_t>(port_));

    if (bind(listen_fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        fprintf(stderr, "[nexus-http] ERROR: bind(port %d): %s\n", port_, strerror(errno));
        close(listen_fd_);
        listen_fd_ = -1;
        return;
    }

    if (listen(listen_fd_, 16) < 0) {
        fprintf(stderr, "[nexus-http] ERROR: listen(): %s\n", strerror(errno));
        close(listen_fd_);
        listen_fd_ = -1;
        return;
    }

    running_.store(true);
    fprintf(stderr, "[nexus-http] Listening on http://0.0.0.0:%d\n", port_);
    fprintf(stderr, "[nexus-http] Endpoints:\n");
    fprintf(stderr, "[nexus-http]   POST /v1/chat/completions\n");
    fprintf(stderr, "[nexus-http]   POST /v1/completions\n");
    fprintf(stderr, "[nexus-http]   GET  /v1/models\n");
    fprintf(stderr, "[nexus-http]   GET  /health\n");

    accept_loop();
}

void HttpServer::stop() {
    running_.store(false);

    if (listen_fd_ >= 0) {
        shutdown(listen_fd_, SHUT_RDWR);
        close(listen_fd_);
        listen_fd_ = -1;
    }

    // Join all connection threads.
    std::lock_guard<std::mutex> lock(connections_mutex_);
    for (auto& t : connection_threads_) {
        if (t.joinable()) t.join();
    }
    connection_threads_.clear();

    fprintf(stderr, "[nexus-http] Server stopped.\n");
}

// ─── Accept loop ────────────────────────────────────────────────────────────

void HttpServer::accept_loop() {
    while (running_.load()) {
        struct sockaddr_in client_addr{};
        socklen_t addr_len = sizeof(client_addr);

        int client_fd = accept(listen_fd_,
                               reinterpret_cast<struct sockaddr*>(&client_addr),
                               &addr_len);
        if (client_fd < 0) {
            if (running_.load()) {
                fprintf(stderr, "[nexus-http] accept(): %s\n", strerror(errno));
            }
            continue;
        }

        // Spawn a thread for this connection.
        std::lock_guard<std::mutex> lock(connections_mutex_);

        // Clean up finished threads periodically.
        connection_threads_.erase(
            std::remove_if(connection_threads_.begin(), connection_threads_.end(),
                           [](std::thread& t) {
                               if (t.joinable()) {
                                   // We can't easily check if a thread is done without
                                   // a flag, so we just let them accumulate and join
                                   // in stop(). For MVP this is fine.
                                   return false;
                               }
                               return true;
                           }),
            connection_threads_.end()
        );

        connection_threads_.emplace_back([this, client_fd]() {
            handle_connection(client_fd);
        });
    }
}

// ─── Connection handler ─────────────────────────────────────────────────────

void HttpServer::handle_connection(int client_fd) {
    HttpRequest req;
    if (!parse_request(client_fd, req)) {
        send_error(client_fd, 400, "Bad Request");
        close(client_fd);
        return;
    }

    // CORS preflight.
    if (req.method == "OPTIONS") {
        handle_options(client_fd);
    }
    // Route dispatch.
    else if (req.method == "POST" && req.path == "/v1/chat/completions") {
        handle_chat_completions(client_fd, req);
    }
    else if (req.method == "POST" && req.path == "/v1/completions") {
        handle_completions(client_fd, req);
    }
    else if (req.method == "GET" && req.path == "/v1/models") {
        handle_list_models(client_fd);
    }
    else if (req.method == "GET" && req.path == "/health") {
        handle_health(client_fd);
    }
    else {
        send_error(client_fd, 404, "Not Found");
    }

    close(client_fd);
}

// ─── HTTP request parsing ───────────────────────────────────────────────────

bool HttpServer::parse_request(int client_fd, HttpRequest& req) {
    // Read the full header block (up to 8 KB).
    char buf[8192];
    int total_read = 0;
    int header_end = -1;

    while (total_read < static_cast<int>(sizeof(buf) - 1)) {
        int n = static_cast<int>(recv(client_fd, buf + total_read,
                                       sizeof(buf) - 1 - total_read, 0));
        if (n <= 0) return false;
        total_read += n;
        buf[total_read] = '\0';

        // Look for end of headers.
        const char* found = strstr(buf, "\r\n\r\n");
        if (found) {
            header_end = static_cast<int>(found - buf) + 4;
            break;
        }
    }

    if (header_end < 0) return false;

    // Parse request line.
    const char* p = buf;
    const char* method_end = strchr(p, ' ');
    if (!method_end) return false;
    req.method.assign(p, method_end);

    p = method_end + 1;
    const char* path_end = strchr(p, ' ');
    if (!path_end) return false;
    req.path.assign(p, path_end);

    // Strip query string from path.
    auto qpos = req.path.find('?');
    if (qpos != std::string::npos) {
        req.path.resize(qpos);
    }

    // Parse headers for Content-Length.
    req.content_length = 0;
    const char* cl = strcasestr(buf, "Content-Length:");
    if (cl) {
        cl += 15; // strlen("Content-Length:")
        while (*cl == ' ') cl++;
        req.content_length = atoi(cl);
    }

    // Read body if present.
    if (req.content_length > 0) {
        int body_already_read = total_read - header_end;
        req.body.assign(buf + header_end, body_already_read);

        // Read remaining body bytes.
        while (static_cast<int>(req.body.size()) < req.content_length) {
            char chunk[4096];
            int remaining = req.content_length - static_cast<int>(req.body.size());
            int to_read = std::min(remaining, static_cast<int>(sizeof(chunk)));
            int n = static_cast<int>(recv(client_fd, chunk, to_read, 0));
            if (n <= 0) break;
            req.body.append(chunk, n);
        }
    }

    return true;
}

// ─── Chat completions ───────────────────────────────────────────────────────

void HttpServer::handle_chat_completions(int client_fd, const HttpRequest& req) {
    CompletionRequest creq;
    if (!parse_completion_request(req.body, creq)) {
        send_error(client_fd, 400, "Invalid JSON in request body");
        return;
    }

    // Build prompt from messages (simple concatenation).
    std::string prompt;
    for (const auto& msg : creq.messages) {
        if (msg.role == "system") {
            prompt += msg.content + "\n\n";
        } else if (msg.role == "user") {
            prompt += "User: " + msg.content + "\n";
        } else if (msg.role == "assistant") {
            prompt += "Assistant: " + msg.content + "\n";
        }
    }
    prompt += "Assistant:";

    SamplingParams params;
    params.temperature = creq.temperature;
    params.top_p       = creq.top_p;
    params.top_k       = creq.top_k;
    params.max_tokens  = creq.max_tokens;

    int prompt_tokens = static_cast<int>(engine_.tokenize(prompt).size());
    std::string request_id = generate_id();

    if (creq.stream) {
        // ── SSE streaming mode ──────────────────────────────────────────
        send_sse_headers(client_fd);

        int completion_tokens = 0;
        engine_.generate(prompt, params, [&](const std::string& token) {
            send_sse_chat_delta(client_fd, request_id, token);
            completion_tokens++;
        });

        send_sse_chat_done(client_fd, request_id, "stop");

        // Final [DONE] event.
        send_raw(client_fd, "data: [DONE]\n\n");
    } else {
        // ── Non-streaming mode ──────────────────────────────────────────
        std::string output = engine_.generate(prompt, params, nullptr);
        int completion_tokens = static_cast<int>(engine_.tokenize(output).size());

        std::string json = build_chat_response(
            request_id, output, "stop", prompt_tokens, completion_tokens);
        send_response(client_fd, 200, "application/json", json);
    }
}

// ─── Text completions ───────────────────────────────────────────────────────

void HttpServer::handle_completions(int client_fd, const HttpRequest& req) {
    CompletionRequest creq;
    if (!parse_completion_request(req.body, creq)) {
        send_error(client_fd, 400, "Invalid JSON in request body");
        return;
    }

    SamplingParams params;
    params.temperature = creq.temperature;
    params.top_p       = creq.top_p;
    params.top_k       = creq.top_k;
    params.max_tokens  = creq.max_tokens;

    int prompt_tokens = static_cast<int>(engine_.tokenize(creq.prompt).size());
    std::string request_id = generate_id();

    if (creq.stream) {
        send_sse_headers(client_fd);

        int completion_tokens = 0;
        engine_.generate(creq.prompt, params, [&](const std::string& token) {
            send_sse_completion_delta(client_fd, request_id, token);
            completion_tokens++;
        });

        send_sse_completion_done(client_fd, request_id, "stop");
        send_raw(client_fd, "data: [DONE]\n\n");
    } else {
        std::string output = engine_.generate(creq.prompt, params, nullptr);
        int completion_tokens = static_cast<int>(engine_.tokenize(output).size());

        std::string json = build_completion_response(
            request_id, output, "stop", prompt_tokens, completion_tokens);
        send_response(client_fd, 200, "application/json", json);
    }
}

// ─── List models ────────────────────────────────────────────────────────────

void HttpServer::handle_list_models(int client_fd) {
    std::string json = build_models_response();
    send_response(client_fd, 200, "application/json", json);
}

// ─── Health check ───────────────────────────────────────────────────────────

void HttpServer::handle_health(int client_fd) {
    char buf[256];
    snprintf(buf, sizeof(buf),
             "{\"status\":\"ok\",\"model\":\"%s\",\"memory_mb\":%.1f}",
             json_escape(engine_.model_info().name).c_str(),
             engine_.memory_usage() / (1024.0 * 1024.0));
    send_response(client_fd, 200, "application/json", buf);
}

// ─── CORS preflight ─────────────────────────────────────────────────────────

void HttpServer::handle_options(int client_fd) {
    std::string resp =
        "HTTP/1.1 204 No Content\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        "Access-Control-Max-Age: 86400\r\n"
        "Content-Length: 0\r\n"
        "\r\n";
    send_raw(client_fd, resp);
}

// ─── Error response ─────────────────────────────────────────────────────────

void HttpServer::send_error(int client_fd, int status, const char* message) {
    char body[512];
    snprintf(body, sizeof(body),
             "{\"error\":{\"message\":\"%s\",\"type\":\"invalid_request_error\",\"code\":%d}}",
             message, status);
    send_response(client_fd, status, "application/json", body);
}

// ─── JSON parsing (hand-rolled state machine) ───────────────────────────────

/// Skip whitespace, return index of next non-whitespace char.
static size_t skip_ws(const std::string& s, size_t pos) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' ||
                               s[pos] == '\r' || s[pos] == '\n')) {
        pos++;
    }
    return pos;
}

/// Extract a JSON string value starting at pos (must point to opening '"').
/// Returns the string content and advances pos past the closing '"'.
static std::string extract_json_string(const std::string& s, size_t& pos) {
    if (pos >= s.size() || s[pos] != '"') return "";
    pos++; // skip opening quote

    std::string result;
    while (pos < s.size() && s[pos] != '"') {
        if (s[pos] == '\\' && pos + 1 < s.size()) {
            pos++;
            switch (s[pos]) {
                case '"':  result += '"';  break;
                case '\\': result += '\\'; break;
                case '/':  result += '/';  break;
                case 'n':  result += '\n'; break;
                case 'r':  result += '\r'; break;
                case 't':  result += '\t'; break;
                case 'b':  result += '\b'; break;
                case 'f':  result += '\f'; break;
                default:   result += s[pos]; break;
            }
        } else {
            result += s[pos];
        }
        pos++;
    }
    if (pos < s.size()) pos++; // skip closing quote
    return result;
}

/// Find the value corresponding to a key in a JSON object.
/// Returns the start position of the value, or std::string::npos.
static size_t find_json_key(const std::string& json, const std::string& key) {
    // Search for "key" : <value>
    std::string needle = "\"" + key + "\"";
    size_t pos = 0;
    while (pos < json.size()) {
        pos = json.find(needle, pos);
        if (pos == std::string::npos) return std::string::npos;
        pos += needle.size();
        pos = skip_ws(json, pos);
        if (pos < json.size() && json[pos] == ':') {
            pos++;
            pos = skip_ws(json, pos);
            return pos;
        }
    }
    return std::string::npos;
}

/// Find the end of a JSON value starting at pos (string, number, bool, null,
/// object, or array). Returns position after the value.
static size_t find_json_value_end(const std::string& s, size_t pos) {
    pos = skip_ws(s, pos);
    if (pos >= s.size()) return pos;

    if (s[pos] == '"') {
        // String: skip to closing unescaped quote.
        pos++;
        while (pos < s.size()) {
            if (s[pos] == '\\') { pos += 2; continue; }
            if (s[pos] == '"') { pos++; return pos; }
            pos++;
        }
        return pos;
    }
    if (s[pos] == '{' || s[pos] == '[') {
        // Object or array: balance braces/brackets.
        char open = s[pos], close_ch = (open == '{') ? '}' : ']';
        int depth = 1;
        pos++;
        bool in_str = false;
        while (pos < s.size() && depth > 0) {
            if (s[pos] == '\\' && in_str) { pos += 2; continue; }
            if (s[pos] == '"') in_str = !in_str;
            if (!in_str) {
                if (s[pos] == open)     depth++;
                if (s[pos] == close_ch) depth--;
            }
            pos++;
        }
        return pos;
    }
    // Number, bool, or null: read until delimiter.
    while (pos < s.size() && s[pos] != ',' && s[pos] != '}' && s[pos] != ']' &&
           s[pos] != ' ' && s[pos] != '\r' && s[pos] != '\n' && s[pos] != '\t') {
        pos++;
    }
    return pos;
}

std::string HttpServer::json_extract_string(const std::string& json,
                                             const std::string& key) {
    size_t pos = find_json_key(json, key);
    if (pos == std::string::npos) return "";
    return extract_json_string(json, pos);
}

double HttpServer::json_extract_number(const std::string& json,
                                        const std::string& key, double def) {
    size_t pos = find_json_key(json, key);
    if (pos == std::string::npos) return def;
    // Read numeric value.
    size_t end = find_json_value_end(json, pos);
    std::string val = json.substr(pos, end - pos);
    try { return std::stod(val); } catch (...) { return def; }
}

bool HttpServer::json_extract_bool(const std::string& json,
                                    const std::string& key, bool def) {
    size_t pos = find_json_key(json, key);
    if (pos == std::string::npos) return def;
    if (json.compare(pos, 4, "true") == 0) return true;
    if (json.compare(pos, 5, "false") == 0) return false;
    return def;
}

bool HttpServer::parse_messages(const std::string& json,
                                 std::vector<ChatMessage>& msgs) {
    size_t pos = find_json_key(json, "messages");
    if (pos == std::string::npos) return false;

    pos = skip_ws(json, pos);
    if (pos >= json.size() || json[pos] != '[') return false;
    pos++; // skip '['

    // Parse array of message objects.
    while (pos < json.size()) {
        pos = skip_ws(json, pos);
        if (pos >= json.size()) break;
        if (json[pos] == ']') break;
        if (json[pos] == ',') { pos++; continue; }

        if (json[pos] == '{') {
            size_t obj_start = pos;
            size_t obj_end = find_json_value_end(json, pos);
            std::string obj = json.substr(obj_start, obj_end - obj_start);

            ChatMessage msg;
            msg.role    = json_extract_string(obj, "role");
            msg.content = json_extract_string(obj, "content");
            if (!msg.role.empty()) {
                msgs.push_back(std::move(msg));
            }
            pos = obj_end;
        } else {
            pos++;
        }
    }

    return !msgs.empty();
}

bool HttpServer::parse_completion_request(const std::string& json,
                                           CompletionRequest& req) {
    if (json.empty()) return false;

    req.model       = json_extract_string(json, "model");
    req.prompt      = json_extract_string(json, "prompt");
    req.temperature = static_cast<float>(json_extract_number(json, "temperature", 0.7));
    req.top_p       = static_cast<float>(json_extract_number(json, "top_p", 0.9));
    req.top_k       = static_cast<int>(json_extract_number(json, "top_k", 40));
    req.max_tokens  = static_cast<int>(json_extract_number(json, "max_tokens", 2048));
    req.stream      = json_extract_bool(json, "stream", false);

    parse_messages(json, req.messages);

    // At minimum we need either a prompt or messages.
    return !req.prompt.empty() || !req.messages.empty();
}

// ─── JSON output builders ───────────────────────────────────────────────────

std::string HttpServer::json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char hex[8];
                    snprintf(hex, sizeof(hex), "\\u%04x", static_cast<unsigned char>(c));
                    out += hex;
                } else {
                    out += c;
                }
                break;
        }
    }
    return out;
}

std::string HttpServer::build_chat_response(const std::string& id,
                                             const std::string& content,
                                             const std::string& finish_reason,
                                             int prompt_tokens,
                                             int completion_tokens) {
    long long ts = static_cast<long long>(std::time(nullptr));
    int total = prompt_tokens + completion_tokens;

    char buf[8192];
    snprintf(buf, sizeof(buf),
        "{"
          "\"id\":\"%s\","
          "\"object\":\"chat.completion\","
          "\"created\":%lld,"
          "\"model\":\"%s\","
          "\"choices\":["
            "{"
              "\"index\":0,"
              "\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},"
              "\"finish_reason\":\"%s\""
            "}"
          "],"
          "\"usage\":{"
            "\"prompt_tokens\":%d,"
            "\"completion_tokens\":%d,"
            "\"total_tokens\":%d"
          "}"
        "}",
        id.c_str(),
        ts,
        json_escape(engine_.model_info().name).c_str(),
        json_escape(content).c_str(),
        finish_reason.c_str(),
        prompt_tokens, completion_tokens, total);

    return buf;
}

std::string HttpServer::build_completion_response(const std::string& id,
                                                   const std::string& text,
                                                   const std::string& finish_reason,
                                                   int prompt_tokens,
                                                   int completion_tokens) {
    long long ts = static_cast<long long>(std::time(nullptr));
    int total = prompt_tokens + completion_tokens;

    char buf[8192];
    snprintf(buf, sizeof(buf),
        "{"
          "\"id\":\"%s\","
          "\"object\":\"text_completion\","
          "\"created\":%lld,"
          "\"model\":\"%s\","
          "\"choices\":["
            "{"
              "\"index\":0,"
              "\"text\":\"%s\","
              "\"finish_reason\":\"%s\""
            "}"
          "],"
          "\"usage\":{"
            "\"prompt_tokens\":%d,"
            "\"completion_tokens\":%d,"
            "\"total_tokens\":%d"
          "}"
        "}",
        id.c_str(),
        ts,
        json_escape(engine_.model_info().name).c_str(),
        json_escape(text).c_str(),
        finish_reason.c_str(),
        prompt_tokens, completion_tokens, total);

    return buf;
}

std::string HttpServer::build_models_response() {
    const auto& info = engine_.model_info();
    long long ts = static_cast<long long>(std::time(nullptr));

    char buf[1024];
    snprintf(buf, sizeof(buf),
        "{"
          "\"object\":\"list\","
          "\"data\":["
            "{"
              "\"id\":\"%s\","
              "\"object\":\"model\","
              "\"created\":%lld,"
              "\"owned_by\":\"nexus\","
              "\"permission\":[]"
            "}"
          "]"
        "}",
        json_escape(info.name).c_str(),
        ts);

    return buf;
}

// ─── SSE streaming ──────────────────────────────────────────────────────────

void HttpServer::send_sse_headers(int client_fd) {
    std::string headers =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: keep-alive\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n";
    send_raw(client_fd, headers);
}

void HttpServer::send_sse_chat_delta(int client_fd, const std::string& id,
                                      const std::string& content) {
    long long ts = static_cast<long long>(std::time(nullptr));

    char buf[4096];
    snprintf(buf, sizeof(buf),
        "data: {"
          "\"id\":\"%s\","
          "\"object\":\"chat.completion.chunk\","
          "\"created\":%lld,"
          "\"model\":\"%s\","
          "\"choices\":["
            "{"
              "\"index\":0,"
              "\"delta\":{\"content\":\"%s\"},"
              "\"finish_reason\":null"
            "}"
          "]"
        "}\n\n",
        id.c_str(),
        ts,
        json_escape(engine_.model_info().name).c_str(),
        json_escape(content).c_str());

    send_raw(client_fd, buf);
}

void HttpServer::send_sse_chat_done(int client_fd, const std::string& id,
                                     const std::string& finish_reason) {
    long long ts = static_cast<long long>(std::time(nullptr));

    char buf[2048];
    snprintf(buf, sizeof(buf),
        "data: {"
          "\"id\":\"%s\","
          "\"object\":\"chat.completion.chunk\","
          "\"created\":%lld,"
          "\"model\":\"%s\","
          "\"choices\":["
            "{"
              "\"index\":0,"
              "\"delta\":{},"
              "\"finish_reason\":\"%s\""
            "}"
          "]"
        "}\n\n",
        id.c_str(),
        ts,
        json_escape(engine_.model_info().name).c_str(),
        finish_reason.c_str());

    send_raw(client_fd, buf);
}

void HttpServer::send_sse_completion_delta(int client_fd, const std::string& id,
                                            const std::string& text) {
    long long ts = static_cast<long long>(std::time(nullptr));

    char buf[4096];
    snprintf(buf, sizeof(buf),
        "data: {"
          "\"id\":\"%s\","
          "\"object\":\"text_completion.chunk\","
          "\"created\":%lld,"
          "\"model\":\"%s\","
          "\"choices\":["
            "{"
              "\"index\":0,"
              "\"text\":\"%s\","
              "\"finish_reason\":null"
            "}"
          "]"
        "}\n\n",
        id.c_str(),
        ts,
        json_escape(engine_.model_info().name).c_str(),
        json_escape(text).c_str());

    send_raw(client_fd, buf);
}

void HttpServer::send_sse_completion_done(int client_fd, const std::string& id,
                                           const std::string& finish_reason) {
    long long ts = static_cast<long long>(std::time(nullptr));

    char buf[2048];
    snprintf(buf, sizeof(buf),
        "data: {"
          "\"id\":\"%s\","
          "\"object\":\"text_completion.chunk\","
          "\"created\":%lld,"
          "\"model\":\"%s\","
          "\"choices\":["
            "{"
              "\"index\":0,"
              "\"text\":\"\","
              "\"finish_reason\":\"%s\""
            "}"
          "]"
        "}\n\n",
        id.c_str(),
        ts,
        json_escape(engine_.model_info().name).c_str(),
        finish_reason.c_str());

    send_raw(client_fd, buf);
}

// ─── HTTP helpers ───────────────────────────────────────────────────────────

void HttpServer::send_response(int client_fd, int status,
                                const std::string& content_type,
                                const std::string& body) {
    const char* status_text = "OK";
    switch (status) {
        case 200: status_text = "OK"; break;
        case 400: status_text = "Bad Request"; break;
        case 404: status_text = "Not Found"; break;
        case 500: status_text = "Internal Server Error"; break;
    }

    char header[512];
    snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "\r\n",
        status, status_text,
        content_type.c_str(),
        body.size());

    std::string resp = std::string(header) + body;
    send_raw(client_fd, resp);
}

void HttpServer::send_raw(int client_fd, const std::string& data) {
    size_t sent = 0;
    while (sent < data.size()) {
        ssize_t n = send(client_fd, data.data() + sent, data.size() - sent, 0);
        if (n <= 0) break;
        sent += static_cast<size_t>(n);
    }
}

std::string HttpServer::generate_id() {
    static std::mt19937 rng(static_cast<unsigned>(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    static std::mutex rng_mutex;

    char buf[32];
    {
        std::lock_guard<std::mutex> lock(rng_mutex);
        uint64_t val = (static_cast<uint64_t>(rng()) << 32) | rng();
        snprintf(buf, sizeof(buf), "nexus-%016llx",
                 static_cast<unsigned long long>(val));
    }
    return buf;
}

}  // namespace nexus
