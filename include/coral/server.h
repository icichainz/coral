// HTTP server (POSIX sockets + kqueue) exposing an OpenAI-compatible API:
//   POST /v1/chat/completions   (stream: SSE)
//   POST /v1/completions        (stream: SSE)
//   GET  /v1/models
//   GET  /health
//
// Threading (src/server/http.cpp, src/server/openai_api.cpp):
//   * One IO thread (the caller of run()) owns every socket: it accepts,
//     parses requests, calls handlers and is the only thread that send()s or
//     close()s a file descriptor.
//   * Handlers run on the IO thread. A handler that must not block it calls
//     HttpResponse::defer() and finishes the response later from any thread.
//     Response bytes are queued on the connection (per-connection mutex) and
//     the IO thread is woken through a kqueue EVFILT_USER event to flush them
//     with non-blocking send(). A writer blocks (backpressure) while more than
//     ServerOptions::max_pending_write_bytes are waiting for a slow client.
//   * The OpenAI layer submits every generation to a BatchEngine driven by
//     ONE worker thread (continuous batching: up to max_batch sequences decode
//     together, the rest wait in FIFO order); everything else is answered
//     inline.
//   * The IO thread keeps reading while a response is in flight, so a peer
//     close is seen at once: client_gone() turns true and that sequence alone
//     is cancelled at the next step.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>

#include "coral/engine.h"

namespace coral {

struct HttpRequest {
    std::string method, path, body;                 // path excludes the query string
    std::map<std::string, std::string> headers;     // lower-case keys
    std::map<std::string, std::string> query;       // URL-decoded
    std::string target;                             // raw request target (path?query)
    std::string version;                            // "HTTP/1.1"
    std::string remote;                             // peer address

    // Header value by (any-case) name, or `def`.
    std::string header(std::string_view name, std::string_view def = {}) const;
};

// Streaming-capable response writer. Thread-safe: after defer(), any thread
// may write/end it. Headers go out on the first write()/end().
class HttpResponse {
public:
    virtual ~HttpResponse() = default;
    virtual void status(int code) = 0;
    virtual void header(std::string_view name, std::string_view value) = 0;
    virtual void write(std::string_view chunk) = 0;      // chunked transfer
    virtual void end(std::string_view body = {}) = 0;    // Content-Length if nothing was written
    virtual bool client_gone() const = 0;                // peer closed; stop generating

    // Keep the response open after the handler returns; the returned handle
    // may be moved to another thread, which must eventually call end() (a
    // handle dropped without end() ends the response).
    virtual std::shared_ptr<HttpResponse> defer() = 0;
    virtual bool headers_sent() const = 0;
    // Free-form text appended to this request's access-log line.
    virtual void annotate(std::string_view note) = 0;
};

using HttpHandler = std::function<void(const HttpRequest&, HttpResponse&)>;

struct ServerOptions {
    std::string host = "127.0.0.1";                  // IPv4 or IPv6 literal or name; "::" = all (dual stack)
    uint16_t port = 8080;                            // 0 = ephemeral (see HttpServer::port())
    uint32_t max_connections = 256;
    std::string model_name = "gpt-oss-20b";

    size_t max_body_bytes = 16u << 20;               // larger bodies get 413
    size_t max_header_bytes = 64u << 10;             // larger header blocks get 431
    size_t max_pending_write_bytes = 8u << 20;       // per-connection backpressure threshold
    uint32_t keep_alive_seconds = 75;                // idle keep-alive connections are closed after this
    bool handle_signals = false;                     // SIGINT/SIGTERM -> stop()
    bool log_requests = false;                       // one line per request on stderr

    // OpenAI layer.
    std::string device_name;                         // reported by /health
    std::string default_reasoning = "medium";        // low | medium | high
    uint32_t kv_capacity = 0;                        // context size; 0 = engine default (grows per request)
    uint32_t max_batch = 8;                          // sequences decoding together (continuous batching), 1..8
};

class HttpServer {
public:
    // Binds and listens immediately (throws std::runtime_error on failure).
    static std::unique_ptr<HttpServer> create(const ServerOptions& opts);
    virtual ~HttpServer() = default;
    virtual void route(std::string_view method, std::string_view path, HttpHandler handler) = 0;
    virtual void run() = 0;    // blocks until stop()
    virtual void stop() = 0;   // any thread, also from a signal-free context; idempotent

    virtual uint16_t port() const = 0;                         // bound port (useful with port 0)
    virtual const ServerOptions& options() const = 0;
    // Keep `obj` alive until the server is destroyed (destroyed before the
    // server's own state, after run() has returned).
    virtual void attach(std::shared_ptr<void> obj) = 0;
};

// OpenAI-style error body: {"error":{"message","type","code"}}.
void send_json_error(HttpResponse& res, int status, std::string_view message,
                     std::string_view type = {}, std::string_view code = {});

// Installs the OpenAI-compatible routes on `server`, backed by a BatchEngine
// made from `engine` (Engine::make_batch_engine, opts.max_batch). `engine`
// must outlive `server` and must not be used while the server runs.
// Generation runs on one worker thread owned by the server (see attach()).
void install_openai_api(HttpServer& server, Engine& engine, const ServerOptions& opts);

} // namespace coral
