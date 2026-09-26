// HTTP/1.1 server: POSIX sockets, one kqueue IO thread, keep-alive, chunked
// streaming responses writable from other threads. See include/coral/server.h
// for the threading model.
#include "coral/server.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/event.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "coral/json.h"

namespace coral {

std::string HttpRequest::header(std::string_view name, std::string_view def) const {
    std::string k(name);
    for (char& c : k) c = char(std::tolower(uint8_t(c)));
    auto it = headers.find(k);
    return it == headers.end() ? std::string(def) : it->second;
}

void send_json_error(HttpResponse& res, int status, std::string_view message, std::string_view type,
                     std::string_view code) {
    if (type.empty()) type = status >= 500 ? "server_error" : "invalid_request_error";
    Json::Object e{{"message", std::string(message)}, {"type", std::string(type)},
                   {"code", code.empty() ? Json() : Json(std::string(code))}, {"param", Json()}};
    res.status(status);
    res.header("Content-Type", "application/json");
    res.end(Json(Json::Object{{"error", Json(std::move(e))}}).dump() + "\n");
}

namespace {

using clk = std::chrono::steady_clock;

constexpr uintptr_t kWakeIdent = 1, kTimerIdent = 2;

const char* reason_phrase(int code) {
    switch (code) {
        case 100: return "Continue";
        case 200: return "OK";
        case 201: return "Created";
        case 204: return "No Content";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 408: return "Request Timeout";
        case 411: return "Length Required";
        case 413: return "Payload Too Large";
        case 429: return "Too Many Requests";
        case 431: return "Request Header Fields Too Large";
        case 500: return "Internal Server Error";
        case 501: return "Not Implemented";
        case 503: return "Service Unavailable";
        case 505: return "HTTP Version Not Supported";
        default:  return code < 400 ? "OK" : "Error";
    }
}

std::string lower(std::string_view s) {
    std::string o(s);
    for (char& c : o) c = char(std::tolower(uint8_t(c)));
    return o;
}

std::string_view trim(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) s.remove_suffix(1);
    return s;
}

std::string url_decode(std::string_view s, bool plus_is_space) {
    std::string o;
    o.reserve(s.size());
    auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size() && hex(s[i + 1]) >= 0 && hex(s[i + 2]) >= 0) {
            o += char(hex(s[i + 1]) * 16 + hex(s[i + 2]));
            i += 2;
        } else if (plus_is_space && s[i] == '+') {
            o += ' ';
        } else {
            o += s[i];
        }
    }
    return o;
}

// Token list header check, e.g. Connection: keep-alive, Upgrade.
bool has_token(std::string_view value, std::string_view token) {
    std::string v = lower(value);
    size_t p = 0;
    while (p <= v.size()) {
        size_t q = v.find(',', p);
        if (q == std::string::npos) q = v.size();
        if (trim(std::string_view(v).substr(p, q - p)) == token) return true;
        p = q + 1;
    }
    return false;
}

struct Core;

struct Conn {
    uint64_t id = 0;
    int fd = -1;
    std::string peer;
    // IO thread only.
    std::string in;
    bool busy = false;              // a request is being handled
    bool sent_continue = false;
    bool want_write = false;        // EVFILT_WRITE registered
    clk::time_point last_active = clk::now();
    // Shared (mu).
    std::mutex mu;
    std::condition_variable cv;     // backpressure wait
    std::string out;                // queued bytes
    size_t out_off = 0;
    bool response_done = false;     // current response fully queued
    bool close_after = false;       // close once `out` drains
    std::atomic<bool> gone{false};  // peer closed / connection dropped
};

struct Route { std::map<std::string, HttpHandler> by_method; };

struct Core : std::enable_shared_from_this<Core> {
    ServerOptions opts;
    int kq = -1;
    std::vector<int> listeners;
    uint16_t bound_port = 0;
    std::map<std::string, Route> routes;

    std::atomic<bool> stopping{false};
    std::atomic<bool> running{false};
    std::thread::id io_thread;

    std::unordered_map<uint64_t, std::shared_ptr<Conn>> conns;   // IO thread only
    uint64_t next_id = 16;                                       // ids below are reserved idents

    std::mutex dirty_mu;
    std::vector<std::shared_ptr<Conn>> dirty;                    // connections with queued output

    ~Core() {
        for (int fd : listeners) ::close(fd);
        if (kq >= 0) ::close(kq);
    }

    void wake() {
        struct kevent ev;
        EV_SET(&ev, kWakeIdent, EVFILT_USER, 0, NOTE_TRIGGER, 0, nullptr);
        kevent(kq, &ev, 1, nullptr, 0, nullptr);
    }

    bool on_io_thread() const { return running.load() && std::this_thread::get_id() == io_thread; }

    // Queue bytes on a connection (any thread).
    void enqueue(const std::shared_ptr<Conn>& c, std::string_view bytes, bool done, bool close) {
        {
            std::unique_lock<std::mutex> lk(c->mu);
            if (c->gone) return;
            if (!on_io_thread() && !bytes.empty())
                c->cv.wait(lk, [&] { return c->gone.load() || c->out.size() - c->out_off < opts.max_pending_write_bytes; });
            if (c->gone) return;
            c->out.append(bytes);
            if (done) { c->response_done = true; if (close) c->close_after = true; }
        }
        {
            std::lock_guard<std::mutex> lk(dirty_mu);
            dirty.push_back(c);
        }
        if (!on_io_thread()) wake();
    }
};

class Response final : public HttpResponse, public std::enable_shared_from_this<Response> {
public:
    Response(std::shared_ptr<Core> core, std::shared_ptr<Conn> conn, const HttpRequest& req, bool keep_alive)
        : core_(std::move(core)), conn_(std::move(conn)), method_(req.method), path_(req.path),
          keep_alive_(keep_alive), t0_(clk::now()) {}

    ~Response() override { end(); }

    void status(int code) override { std::lock_guard<std::mutex> lk(mu_); if (!headers_sent_) status_ = code; }
    void header(std::string_view name, std::string_view value) override {
        std::lock_guard<std::mutex> lk(mu_);
        if (headers_sent_) return;
        std::string n = lower(name);
        if (n == "content-length" || n == "transfer-encoding" || n == "connection") return;  // managed here
        headers_.emplace_back(std::string(name), std::string(value));
    }
    void write(std::string_view chunk) override {
        std::string buf;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (ended_) return;
            if (!headers_sent_) buf = head(/*chunked=*/true, 0);
            if (!chunk.empty()) {
                char n[20];
                std::snprintf(n, sizeof n, "%zx\r\n", chunk.size());
                buf += n;
                buf.append(chunk);
                buf += "\r\n";
                bytes_ += chunk.size();
            }
        }
        if (!buf.empty()) core_->enqueue(conn_, buf, false, false);
    }
    void end(std::string_view body = {}) override {
        std::string buf;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (ended_) return;
            ended_ = true;
            if (!headers_sent_) {
                buf = head(/*chunked=*/false, body.size());
                buf.append(body);
            } else {
                if (!body.empty()) {
                    char n[20];
                    std::snprintf(n, sizeof n, "%zx\r\n", body.size());
                    buf += n;
                    buf.append(body);
                    buf += "\r\n";
                }
                buf += "0\r\n\r\n";
            }
            bytes_ += body.size();
        }
        core_->enqueue(conn_, buf, true, !keep_alive_);
        if (core_->opts.log_requests) {
            const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0_).count();
            std::fprintf(stderr, "[coral] %s %s %s %d %.1f ms%s%s%s\n", conn_->peer.c_str(), method_.c_str(),
                         path_.c_str(), status_, ms, note_.empty() ? "" : " | ", note_.c_str(),
                         client_gone() ? " (client gone)" : "");
        }
    }
    bool client_gone() const override { return conn_->gone.load() || core_->stopping.load(); }
    std::shared_ptr<HttpResponse> defer() override { deferred_ = true; return shared_from_this(); }
    bool headers_sent() const override { std::lock_guard<std::mutex> lk(mu_); return headers_sent_; }
    void annotate(std::string_view note) override {
        std::lock_guard<std::mutex> lk(mu_);
        if (!note_.empty()) note_ += ' ';
        note_.append(note);
    }

    bool deferred() const { return deferred_; }
    bool ended() const { std::lock_guard<std::mutex> lk(mu_); return ended_; }

private:
    std::string head(bool chunked, size_t length) {
        headers_sent_ = true;
        std::string h = "HTTP/1.1 " + std::to_string(status_) + " " + reason_phrase(status_) + "\r\n";
        bool has_ct = false;
        for (auto& [k, v] : headers_) {
            if (lower(k) == "content-type") has_ct = true;
            h += k + ": " + v + "\r\n";
        }
        if (!has_ct && (chunked || length)) h += "Content-Type: text/plain; charset=utf-8\r\n";
        h += "Server: coral\r\nAccess-Control-Allow-Origin: *\r\n";
        if (chunked) h += "Transfer-Encoding: chunked\r\n";
        else if (status_ != 204 && status_ != 304) h += "Content-Length: " + std::to_string(length) + "\r\n";
        h += keep_alive_ ? "Connection: keep-alive\r\n" : "Connection: close\r\n";
        h += "\r\n";
        return h;
    }

    std::shared_ptr<Core> core_;
    std::shared_ptr<Conn> conn_;
    std::string method_, path_;
    bool keep_alive_;
    clk::time_point t0_;
    mutable std::mutex mu_;
    int status_ = 200;
    std::vector<std::pair<std::string, std::string>> headers_;
    bool headers_sent_ = false, ended_ = false;
    std::atomic<bool> deferred_{false};
    std::string note_;
    size_t bytes_ = 0;
};

// ---------------------------------------------------------------------------

class Server final : public HttpServer {
public:
    explicit Server(const ServerOptions& o) : core_(std::make_shared<Core>()) {
        core_->opts = o;
        core_->kq = kqueue();
        if (core_->kq < 0) throw std::runtime_error(std::string("kqueue: ") + std::strerror(errno));
        listen_all();
        std::vector<struct kevent> ch;
        struct kevent ev;
        EV_SET(&ev, kWakeIdent, EVFILT_USER, EV_ADD | EV_CLEAR, 0, 0, nullptr); ch.push_back(ev);
        EV_SET(&ev, kTimerIdent, EVFILT_TIMER, EV_ADD, NOTE_SECONDS, 1, nullptr); ch.push_back(ev);
        for (int fd : core_->listeners) { EV_SET(&ev, fd, EVFILT_READ, EV_ADD, 0, 0, (void*)uintptr_t(0)); ch.push_back(ev); }
        if (kevent(core_->kq, ch.data(), int(ch.size()), nullptr, 0, nullptr) < 0)
            throw std::runtime_error(std::string("kevent: ") + std::strerror(errno));
    }

    ~Server() override {
        stop();
        attachments_.clear();   // joins worker threads while the core is still alive
    }

    void route(std::string_view method, std::string_view path, HttpHandler h) override {
        core_->routes[std::string(path)].by_method[std::string(method)] = std::move(h);
    }
    uint16_t port() const override { return core_->bound_port; }
    const ServerOptions& options() const override { return core_->opts; }
    void attach(std::shared_ptr<void> obj) override { attachments_.push_back(std::move(obj)); }

    void stop() override {
        core_->stopping = true;
        core_->wake();
    }

    void run() override;

private:
    void listen_all();
    void accept_ready(int lfd);
    void on_readable(const std::shared_ptr<Conn>& c, bool eof);
    void process_input(const std::shared_ptr<Conn>& c);
    void dispatch(const std::shared_ptr<Conn>& c, HttpRequest& req, bool keep_alive);
    void flush(const std::shared_ptr<Conn>& c);
    void close_conn(const std::shared_ptr<Conn>& c);
    void set_write_interest(const std::shared_ptr<Conn>& c, bool on);
    void reject(const std::shared_ptr<Conn>& c, int status, const std::string& msg);

    std::shared_ptr<Core> core_;
    std::vector<std::shared_ptr<void>> attachments_;
};

void Server::listen_all() {
    const ServerOptions& o = core_->opts;
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE | AI_NUMERICSERV;
    addrinfo* res = nullptr;
    const std::string port = std::to_string(o.port);
    const char* host = o.host.empty() ? nullptr : o.host.c_str();
    if (int e = getaddrinfo(host, port.c_str(), &hints, &res); e != 0)
        throw std::runtime_error("getaddrinfo(" + o.host + "): " + gai_strerror(e));
    std::string last_err;
    uint16_t bound = o.port;
    for (addrinfo* ai = res; ai; ai = ai->ai_next) {
        int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) { last_err = std::strerror(errno); continue; }
        int one = 1, zero = 0;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
        if (ai->ai_family == AF_INET6) {
            // "::" serves IPv4 too; a specific v6 address stays v6-only.
            const bool any = IN6_IS_ADDR_UNSPECIFIED(&reinterpret_cast<sockaddr_in6*>(ai->ai_addr)->sin6_addr);
            setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, any ? &zero : &one, sizeof one);
        }
        // With port 0 and several addresses (e.g. "localhost"), bind all to the first ephemeral port.
        if (bound != o.port || o.port == 0) {
            if (ai->ai_family == AF_INET) reinterpret_cast<sockaddr_in*>(ai->ai_addr)->sin_port = htons(bound);
            else if (ai->ai_family == AF_INET6) reinterpret_cast<sockaddr_in6*>(ai->ai_addr)->sin6_port = htons(bound);
        }
        if (bind(fd, ai->ai_addr, ai->ai_addrlen) < 0 || listen(fd, 512) < 0) {
            last_err = std::strerror(errno);
            ::close(fd);
            continue;
        }
        fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);
        sockaddr_storage ss{};
        socklen_t sl = sizeof ss;
        getsockname(fd, reinterpret_cast<sockaddr*>(&ss), &sl);
        bound = ntohs(ss.ss_family == AF_INET6 ? reinterpret_cast<sockaddr_in6*>(&ss)->sin6_port
                                               : reinterpret_cast<sockaddr_in*>(&ss)->sin_port);
        core_->listeners.push_back(fd);
    }
    freeaddrinfo(res);
    if (core_->listeners.empty())
        throw std::runtime_error("cannot listen on " + o.host + ":" + port + ": " + last_err);
    core_->bound_port = bound;
}

void Server::set_write_interest(const std::shared_ptr<Conn>& c, bool on) {
    if (c->want_write == on || c->fd < 0) return;
    struct kevent ev;
    EV_SET(&ev, c->fd, EVFILT_WRITE, on ? EV_ADD : EV_DELETE, 0, 0, (void*)uintptr_t(c->id));
    kevent(core_->kq, &ev, 1, nullptr, 0, nullptr);
    c->want_write = on;
}

void Server::close_conn(const std::shared_ptr<Conn>& c) {
    if (c->fd < 0) return;
    {
        std::lock_guard<std::mutex> lk(c->mu);
        c->gone = true;
    }
    c->cv.notify_all();
    ::close(c->fd);   // also drops its kqueue filters
    c->fd = -1;
    core_->conns.erase(c->id);
}

void Server::accept_ready(int lfd) {
    for (;;) {
        sockaddr_storage ss{};
        socklen_t sl = sizeof ss;
        int fd = accept(lfd, reinterpret_cast<sockaddr*>(&ss), &sl);
        if (fd < 0) return;   // EAGAIN or transient error
        int one = 1;
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);
        if (core_->conns.size() >= core_->opts.max_connections) {
            static const char busy[] =
                "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nConnection: close\r\n"
                "Content-Length: 97\r\n\r\n"
                "{\"error\":{\"message\":\"too many connections\",\"type\":\"server_error\",\"code\":\"max_connections\"}}\n\n";
            (void)!::send(fd, busy, sizeof busy - 1, 0);
            ::close(fd);
            continue;
        }
        auto c = std::make_shared<Conn>();
        c->id = core_->next_id++;
        c->fd = fd;
        char host[INET6_ADDRSTRLEN] = "?";
        if (ss.ss_family == AF_INET) inet_ntop(AF_INET, &reinterpret_cast<sockaddr_in*>(&ss)->sin_addr, host, sizeof host);
        else if (ss.ss_family == AF_INET6) inet_ntop(AF_INET6, &reinterpret_cast<sockaddr_in6*>(&ss)->sin6_addr, host, sizeof host);
        c->peer = host;
        struct kevent ev;
        EV_SET(&ev, fd, EVFILT_READ, EV_ADD, 0, 0, (void*)uintptr_t(c->id));
        if (kevent(core_->kq, &ev, 1, nullptr, 0, nullptr) < 0) { ::close(fd); continue; }
        core_->conns[c->id] = c;
    }
}

void Server::on_readable(const std::shared_ptr<Conn>& c, bool eof) {
    char buf[65536];
    for (;;) {
        ssize_t n = recv(c->fd, buf, sizeof buf, 0);
        if (n > 0) {
            c->in.append(buf, size_t(n));
            c->last_active = clk::now();
            // Bytes pipelined behind an in-flight request: bounded.
            if (c->in.size() > core_->opts.max_body_bytes + core_->opts.max_header_bytes + 65536) { close_conn(c); return; }
            continue;
        }
        if (n == 0) { close_conn(c); return; }   // peer closed
        if (errno == EAGAIN || errno == EWOULDBLOCK) break;
        if (errno == EINTR) continue;
        close_conn(c);
        return;
    }
    if (eof) { close_conn(c); return; }
    process_input(c);
}

void Server::reject(const std::shared_ptr<Conn>& c, int status, const std::string& msg) {
    HttpRequest req;
    req.method = "?";
    req.path = "?";
    auto res = std::make_shared<Response>(core_, c, req, /*keep_alive=*/false);
    c->busy = true;
    c->in.clear();
    const char* code = status == 413 ? "request_too_large" : status == 431 ? "headers_too_large" : nullptr;
    send_json_error(*res, status, msg, {}, code ? code : "");
}

void Server::process_input(const std::shared_ptr<Conn>& c) {
    const ServerOptions& o = core_->opts;
    while (c->fd >= 0 && !c->busy) {
        const size_t he = c->in.find("\r\n\r\n");
        if (he == std::string::npos) {
            if (c->in.size() > o.max_header_bytes) reject(c, 431, "request header block too large");
            return;
        }
        if (he > o.max_header_bytes) { reject(c, 431, "request header block too large"); return; }
        std::string_view head(c->in.data(), he);

        HttpRequest req;
        req.remote = c->peer;
        // Request line.
        size_t le = head.find("\r\n");
        std::string_view line = head.substr(0, le);
        const size_t s1 = line.find(' '), s2 = line.rfind(' ');
        if (s1 == std::string_view::npos || s2 == s1) { reject(c, 400, "malformed request line"); return; }
        req.method = std::string(line.substr(0, s1));
        req.target = std::string(line.substr(s1 + 1, s2 - s1 - 1));
        req.version = std::string(line.substr(s2 + 1));
        if (req.version != "HTTP/1.1" && req.version != "HTTP/1.0") { reject(c, 505, "unsupported HTTP version"); return; }
        // Headers.
        size_t p = le == std::string_view::npos ? head.size() : le + 2;
        bool bad = false;
        while (p < head.size()) {
            size_t e = head.find("\r\n", p);
            if (e == std::string_view::npos) e = head.size();
            std::string_view h = head.substr(p, e - p);
            p = e + 2;
            const size_t colon = h.find(':');
            if (colon == std::string_view::npos || colon == 0) { bad = true; break; }
            std::string k = lower(trim(h.substr(0, colon)));
            std::string v(trim(h.substr(colon + 1)));
            auto [it, fresh] = req.headers.emplace(k, v);
            if (!fresh) it->second += ", " + v;
        }
        if (bad) { reject(c, 400, "malformed header line"); return; }
        // Target -> path + query.
        std::string_view tgt = req.target;
        if (tgt.rfind("http://", 0) == 0 || tgt.rfind("https://", 0) == 0) {   // absolute form
            size_t sl = tgt.find('/', tgt.find("//") + 2);
            tgt = sl == std::string_view::npos ? std::string_view("/") : tgt.substr(sl);
        }
        const size_t q = tgt.find('?');
        req.path = url_decode(tgt.substr(0, q), false);
        if (q != std::string_view::npos) {
            std::string_view qs = tgt.substr(q + 1);
            size_t a = 0;
            while (a <= qs.size()) {
                size_t b = qs.find('&', a);
                if (b == std::string_view::npos) b = qs.size();
                std::string_view kv = qs.substr(a, b - a);
                if (!kv.empty()) {
                    const size_t eq = kv.find('=');
                    req.query[url_decode(kv.substr(0, eq), true)] =
                        eq == std::string_view::npos ? std::string() : url_decode(kv.substr(eq + 1), true);
                }
                a = b + 1;
            }
        }
        // Body framing.
        if (req.headers.count("transfer-encoding")) {
            reject(c, 501, "chunked request bodies are not supported; send Content-Length");
            return;
        }
        size_t len = 0;
        if (auto it = req.headers.find("content-length"); it != req.headers.end()) {
            const std::string& v = it->second;
            if (v.empty() || v.size() > 19 || !std::all_of(v.begin(), v.end(), [](char ch) { return ch >= '0' && ch <= '9'; })) {
                reject(c, 400, "invalid Content-Length");
                return;
            }
            len = std::stoull(v);
        }
        if (len > o.max_body_bytes) {
            reject(c, 413, "request body exceeds " + std::to_string(o.max_body_bytes) + " bytes");
            return;
        }
        const size_t total = he + 4 + len;
        if (c->in.size() < total) {
            if (!c->sent_continue && has_token(req.header("expect"), "100-continue")) {
                c->sent_continue = true;
                core_->enqueue(c, "HTTP/1.1 100 Continue\r\n\r\n", false, false);
            }
            return;
        }
        req.body = c->in.substr(he + 4, len);
        c->in.erase(0, total);
        c->sent_continue = false;

        const std::string conn_hdr = req.header("connection");
        const bool keep_alive = !core_->stopping &&
            (req.version == "HTTP/1.1" ? !has_token(conn_hdr, "close") : has_token(conn_hdr, "keep-alive"));
        dispatch(c, req, keep_alive);
    }
}

void Server::dispatch(const std::shared_ptr<Conn>& c, HttpRequest& req, bool keep_alive) {
    c->busy = true;
    auto res = std::make_shared<Response>(core_, c, req, keep_alive);
    auto rit = core_->routes.find(req.path);
    if (req.method == "OPTIONS") {   // CORS preflight
        res->status(204);
        res->header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res->header("Access-Control-Allow-Headers", "Authorization, Content-Type");
        res->header("Access-Control-Max-Age", "86400");
        res->end();
        return;
    }
    if (rit == core_->routes.end()) {
        send_json_error(*res, 404, "no route for " + req.method + " " + req.path, "invalid_request_error", "not_found");
        return;
    }
    auto mit = rit->second.by_method.find(req.method);
    if (mit == rit->second.by_method.end()) {
        std::string allow;
        for (auto& [m, h] : rit->second.by_method) allow += (allow.empty() ? "" : ", ") + m;
        res->header("Allow", allow);
        send_json_error(*res, 405, "method " + req.method + " not allowed on " + req.path, "invalid_request_error",
                        "method_not_allowed");
        return;
    }
    try {
        mit->second(req, *res);
    } catch (const std::exception& e) {
        if (!res->headers_sent()) send_json_error(*res, 500, e.what(), "server_error", "internal_error");
        else res->end();
        return;
    }
    if (!res->deferred()) res->end();
}

void Server::flush(const std::shared_ptr<Conn>& c) {
    if (c->fd < 0) return;
    bool done = false, close_after = false, error = false;
    {
        std::lock_guard<std::mutex> lk(c->mu);
        while (c->out_off < c->out.size()) {
            ssize_t n = ::send(c->fd, c->out.data() + c->out_off, c->out.size() - c->out_off, 0);
            if (n > 0) { c->out_off += size_t(n); continue; }
            if (n < 0 && errno == EINTR) continue;
            if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
            error = true;
            break;
        }
        if (c->out_off == c->out.size()) { c->out.clear(); c->out_off = 0; }
        else if (c->out_off > (1u << 20)) { c->out.erase(0, c->out_off); c->out_off = 0; }
        const bool empty = c->out.empty();
        if (empty && c->response_done) {
            done = true;
            close_after = c->close_after;
            c->response_done = false;
        }
    }
    c->cv.notify_all();
    if (error) { close_conn(c); return; }
    {
        std::lock_guard<std::mutex> lk(c->mu);
        set_write_interest(c, !c->out.empty());
    }
    c->last_active = clk::now();
    if (done) {
        if (close_after) {
            ::shutdown(c->fd, SHUT_WR);
            close_conn(c);
            return;
        }
        c->busy = false;
        process_input(c);   // pipelined request already buffered?
    }
}

void Server::run() {
    Core& k = *core_;
    k.io_thread = std::this_thread::get_id();
    k.running = true;
    if (k.opts.handle_signals) {
        std::signal(SIGINT, SIG_IGN);
        std::signal(SIGTERM, SIG_IGN);
        struct kevent ev[2];
        EV_SET(&ev[0], SIGINT, EVFILT_SIGNAL, EV_ADD, 0, 0, nullptr);
        EV_SET(&ev[1], SIGTERM, EVFILT_SIGNAL, EV_ADD, 0, 0, nullptr);
        kevent(k.kq, ev, 2, nullptr, 0, nullptr);
    }
    std::signal(SIGPIPE, SIG_IGN);

    std::vector<struct kevent> evs(128);
    while (!k.stopping) {
        const int n = kevent(k.kq, nullptr, 0, evs.data(), int(evs.size()), nullptr);
        if (n < 0) {
            if (errno == EINTR) continue;
            std::fprintf(stderr, "[coral] kevent: %s\n", std::strerror(errno));
            break;
        }
        for (int i = 0; i < n && !k.stopping; ++i) {
            const struct kevent& e = evs[i];
            if (e.filter == EVFILT_USER) continue;   // wake-up: dirty list handled below
            if (e.filter == EVFILT_SIGNAL) {
                std::fprintf(stderr, "\n[coral] signal %d, shutting down\n", int(e.ident));
                k.stopping = true;
                break;
            }
            if (e.filter == EVFILT_TIMER) {
                const auto now = clk::now();
                std::vector<std::shared_ptr<Conn>> idle;
                for (auto& [id, c] : k.conns)
                    if (!c->busy && now - c->last_active > std::chrono::seconds(k.opts.keep_alive_seconds)) idle.push_back(c);
                for (auto& c : idle) close_conn(c);
                continue;
            }
            const uint64_t id = uint64_t(uintptr_t(e.udata));
            if (id == 0) { accept_ready(int(e.ident)); continue; }
            auto it = k.conns.find(id);
            if (it == k.conns.end()) continue;
            std::shared_ptr<Conn> c = it->second;
            if (e.filter == EVFILT_READ) on_readable(c, (e.flags & EV_EOF) != 0);
            else if (e.filter == EVFILT_WRITE) flush(c);
        }
        // Flush connections with queued output (from any thread).
        for (;;) {
            std::vector<std::shared_ptr<Conn>> d;
            {
                std::lock_guard<std::mutex> lk(k.dirty_mu);
                d.swap(k.dirty);
            }
            if (d.empty()) break;
            std::sort(d.begin(), d.end());
            d.erase(std::unique(d.begin(), d.end()), d.end());
            for (auto& c : d) flush(c);
        }
    }

    // Shutdown: stop accepting, drop every connection (in-flight generations
    // see client_gone() and stop at the next token).
    for (int fd : k.listeners) ::close(fd);
    k.listeners.clear();
    std::vector<std::shared_ptr<Conn>> all;
    for (auto& [id, c] : k.conns) all.push_back(c);
    for (auto& c : all) { flush(c); close_conn(c); }
    {
        std::lock_guard<std::mutex> lk(k.dirty_mu);
        k.dirty.clear();
    }
    k.running = false;
    if (k.opts.handle_signals) {
        std::signal(SIGINT, SIG_DFL);
        std::signal(SIGTERM, SIG_DFL);
    }
}

} // namespace

std::unique_ptr<HttpServer> HttpServer::create(const ServerOptions& opts) {
    return std::make_unique<Server>(opts);
}

} // namespace coral
