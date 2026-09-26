// HTTP server (no model): keep-alive, routing errors, 413, chunked streaming
// from a worker thread, client_gone() after a disconnect. With the model:
// the OpenAI-compatible routes end to end.
#include "test.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "coral/engine.h"
#include "coral/gpu.h"
#include "coral/harmony.h"
#include "coral/json.h"
#include "coral/model.h"
#include "coral/server.h"
#include "coral/tokenizer.h"

using namespace coral;

gpu::Device& attn_test_device();
Model& attn_test_model();
std::shared_ptr<Tokenizer> forward_test_tokenizer();

namespace {

using clk = std::chrono::steady_clock;

// ---- tiny blocking HTTP/1.1 client -----------------------------------------
struct Reply {
    int status = 0;
    std::map<std::string, std::string> headers;   // lower-case
    std::string body;                              // de-chunked
    std::vector<std::string> chunks;               // chunk payloads (chunked replies)
};

class Client {
public:
    explicit Client(uint16_t port) {
        fd_ = socket(AF_INET, SOCK_STREAM, 0);
        CHECK(fd_ >= 0);
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons(port);
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        CHECK(connect(fd_, reinterpret_cast<sockaddr*>(&a), sizeof a) == 0);
        timeval tv{120, 0};   // generous: model-backed requests
        setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        int one = 1;
        setsockopt(fd_, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
    }
    ~Client() { close(); }
    void close() { if (fd_ >= 0) { ::close(fd_); fd_ = -1; } }

    void send_raw(const std::string& s) {
        size_t off = 0;
        while (off < s.size()) {
            ssize_t n = ::send(fd_, s.data() + off, s.size() - off, 0);
            CHECK(n > 0);
            off += size_t(n);
        }
    }
    void request(const std::string& method, const std::string& path, const std::string& body = {},
                 const std::string& extra = {}) {
        std::string r = method + " " + path + " HTTP/1.1\r\nHost: localhost\r\n" + extra;
        if (!body.empty() || method == "POST")
            r += "Content-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) + "\r\n";
        r += "\r\n" + body;
        send_raw(r);
    }
    // Reads one full response. `on_chunk` sees each chunk as it arrives; returning false stops early.
    Reply read_reply(const std::function<bool(const std::string&)>& on_chunk = {}) {
        Reply rep;
        size_t he;
        while ((he = buf_.find("\r\n\r\n")) == std::string::npos) fill();
        std::string head = buf_.substr(0, he);
        buf_.erase(0, he + 4);
        CHECK(head.rfind("HTTP/1.1 ", 0) == 0);
        rep.status = std::stoi(head.substr(9, 3));
        size_t p = head.find("\r\n");
        while (p != std::string::npos && p + 2 < head.size()) {
            size_t e = head.find("\r\n", p + 2);
            std::string line = head.substr(p + 2, e == std::string::npos ? std::string::npos : e - p - 2);
            size_t c = line.find(':');
            std::string k = line.substr(0, c);
            for (auto& ch : k) ch = char(std::tolower(uint8_t(ch)));
            size_t vs = line.find_first_not_of(' ', c + 1);
            rep.headers[k] = vs == std::string::npos ? "" : line.substr(vs);
            p = e;
        }
        if (rep.headers["transfer-encoding"] == "chunked") {
            for (;;) {
                size_t le;
                while ((le = buf_.find("\r\n")) == std::string::npos) fill();
                const size_t n = std::stoul(buf_.substr(0, le), nullptr, 16);
                while (buf_.size() < le + 2 + n + 2) fill();
                std::string data = buf_.substr(le + 2, n);
                CHECK(buf_.compare(le + 2 + n, 2, "\r\n") == 0);
                buf_.erase(0, le + 2 + n + 2);
                if (n == 0) break;
                rep.body += data;
                rep.chunks.push_back(data);
                if (on_chunk && !on_chunk(data)) return rep;
            }
        } else {
            const size_t n = rep.headers.count("content-length") ? std::stoul(rep.headers["content-length"]) : 0;
            while (buf_.size() < n) fill();
            rep.body = buf_.substr(0, n);
            buf_.erase(0, n);
        }
        return rep;
    }
    // True once the server has closed the connection (reads EOF).
    bool at_eof() {
        char c;
        ssize_t n = recv(fd_, &c, 1, 0);
        return n == 0;
    }

private:
    void fill() {
        char tmp[65536];
        ssize_t n = recv(fd_, tmp, sizeof tmp, 0);
        if (n <= 0) throw test::Failure("connection closed / timed out while reading a reply");
        buf_.append(tmp, size_t(n));
    }
    int fd_ = -1;
    std::string buf_;
};

// A server on 127.0.0.1:<ephemeral> running on its own thread.
struct Running {
    std::unique_ptr<HttpServer> srv;
    std::thread th;
    explicit Running(std::unique_ptr<HttpServer> s) : srv(std::move(s)) {}
    void start() { th = std::thread([this] { srv->run(); }); }
    ~Running() {
        srv->stop();
        if (th.joinable()) th.join();
        srv.reset();
    }
    uint16_t port() const { return srv->port(); }
};

ServerOptions local_opts() {
    ServerOptions o;
    o.host = "127.0.0.1";
    o.port = 0;
    return o;
}

} // namespace

CORAL_TEST(server_keepalive_and_query) {
    Running r(HttpServer::create(local_opts()));
    CHECK(r.port() != 0);
    r.srv->route("GET", "/hello", [](const HttpRequest& q, HttpResponse& res) {
        res.header("Content-Type", "text/plain");
        auto it = q.query.find("name");
        res.end("hi " + (it == q.query.end() ? std::string("?") : it->second) + " " + q.header("X-Test", "-"));
    });
    r.srv->route("POST", "/echo", [](const HttpRequest& q, HttpResponse& res) { res.end(q.body); });
    r.start();

    Client c(r.port());
    c.request("GET", "/hello?name=a%20b+c&x", {}, "x-TEST: yes\r\n");
    Reply a = c.read_reply();
    CHECK_EQ(a.status, 200);
    CHECK_EQ(a.body, std::string("hi a b c yes"));
    CHECK_EQ(a.headers["connection"], std::string("keep-alive"));
    // Second and third requests on the same connection (the third pipelined behind a body).
    c.send_raw("POST /echo HTTP/1.1\r\nContent-Length: 5\r\n\r\nhelloGET /hello HTTP/1.1\r\nConnection: close\r\n\r\n");
    Reply b = c.read_reply();
    CHECK_EQ(b.status, 200);
    CHECK_EQ(b.body, std::string("hello"));
    Reply d = c.read_reply();
    CHECK_EQ(d.body, std::string("hi ? -"));
    CHECK_EQ(d.headers["connection"], std::string("close"));
    CHECK(c.at_eof());
}

CORAL_TEST(server_errors_json) {
    Running r(HttpServer::create(local_opts()));
    r.srv->route("GET", "/x", [](const HttpRequest&, HttpResponse& res) { res.end("x"); });
    r.srv->route("GET", "/boom", [](const HttpRequest&, HttpResponse&) { throw std::runtime_error("kaput"); });
    r.start();
    Client c(r.port());
    c.request("GET", "/nope");
    Reply a = c.read_reply();
    CHECK_EQ(a.status, 404);
    CHECK_EQ(a.headers["content-type"], std::string("application/json"));
    Json j = Json::parse(a.body);
    CHECK(j["error"]["message"].is_string());
    CHECK_EQ(j["error"]["code"].as_string(), std::string("not_found"));
    c.request("POST", "/x", "{}");                   // same connection: errors keep it alive
    Reply b = c.read_reply();
    CHECK_EQ(b.status, 405);
    CHECK_EQ(b.headers["allow"], std::string("GET"));
    c.request("GET", "/boom");
    Reply e = c.read_reply();
    CHECK_EQ(e.status, 500);
    CHECK_EQ(Json::parse(e.body)["error"]["message"].as_string(), std::string("kaput"));
}

CORAL_TEST(server_413_oversize_body) {
    Running r(HttpServer::create(local_opts()));
    r.srv->route("POST", "/x", [](const HttpRequest&, HttpResponse& res) { res.end("x"); });
    r.start();
    Client c(r.port());
    c.send_raw("POST /x HTTP/1.1\r\nContent-Length: " + std::to_string((16u << 20) + 1) + "\r\n\r\n");
    Reply a = c.read_reply();
    CHECK_EQ(a.status, 413);
    CHECK(Json::parse(a.body)["error"]["message"].is_string());
    CHECK_EQ(a.headers["connection"], std::string("close"));
}

CORAL_TEST(server_chunked_stream_from_thread) {
    Running r(HttpServer::create(local_opts()));
    std::vector<std::thread> threads;
    r.srv->route("GET", "/stream", [&](const HttpRequest&, HttpResponse& res) {
        auto h = res.defer();
        threads.emplace_back([h] {
            h->header("Content-Type", "text/event-stream");
            for (int i = 0; i < 5; ++i) {
                h->write("data: " + std::to_string(i) + "\n\n");
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            h->end();
        });
    });
    r.srv->route("GET", "/fast", [](const HttpRequest&, HttpResponse& res) { res.end("fast"); });
    r.start();
    Client c(r.port()), other(r.port());
    c.request("GET", "/stream");
    other.request("GET", "/fast");   // the IO thread keeps serving while a stream is open
    CHECK_EQ(other.read_reply().body, std::string("fast"));
    Reply a = c.read_reply();
    CHECK_EQ(a.status, 200);
    CHECK_EQ(a.headers["transfer-encoding"], std::string("chunked"));
    CHECK_EQ(a.chunks.size(), size_t(5));
    CHECK_EQ(a.body, std::string("data: 0\n\ndata: 1\n\ndata: 2\n\ndata: 3\n\ndata: 4\n\n"));
    c.request("GET", "/fast");       // keep-alive after a chunked response
    CHECK_EQ(c.read_reply().body, std::string("fast"));
    for (auto& t : threads) t.join();
}

CORAL_TEST(server_client_gone) {
    Running r(HttpServer::create(local_opts()));
    std::atomic<bool> saw_gone{false};
    std::atomic<int> writes{0};
    std::thread worker;
    r.srv->route("GET", "/forever", [&](const HttpRequest&, HttpResponse& res) {
        auto h = res.defer();
        worker = std::thread([h, &saw_gone, &writes] {
            const auto t0 = clk::now();
            while (clk::now() - t0 < std::chrono::seconds(5)) {
                if (h->client_gone()) { saw_gone = true; break; }
                h->write("tick\n");
                ++writes;
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            h->end();
        });
    });
    r.start();
    {
        Client c(r.port());
        c.request("GET", "/forever");
        int n = 0;
        c.read_reply([&](const std::string&) { return ++n < 3; });
        c.close();
    }
    const auto t0 = clk::now();
    worker.join();
    const double s = std::chrono::duration<double>(clk::now() - t0).count();
    std::printf("        client_gone after close: %s in %.1f ms (%d writes)\n", saw_gone ? "yes" : "NO", s * 1e3, writes.load());
    CHECK(saw_gone);
    CHECK(s < 1.0);
}

// ---------------------------------------------------------------------------
// With the model
// ---------------------------------------------------------------------------
namespace {

Engine& server_engine() {
    static std::unique_ptr<Engine> e = [] {
        std::shared_ptr<Model> m(&attn_test_model(), [](Model*) {});   // owned by the shared fixture
        auto x = Engine::create(attn_test_device(), m, forward_test_tokenizer());
        x->warmup();
        return x;
    }();
    return *e;
}

struct ApiServer : Running {
    ApiServer() : Running(HttpServer::create(local_opts())) {
        ServerOptions o = local_opts();
        o.device_name = "test";
        install_openai_api(*srv, server_engine(), o);
        start();
    }
};

Json post_json(uint16_t port, const std::string& path, const Json& body, int want_status = 200) {
    Client c(port);
    c.request("POST", path, body.dump());
    Reply r = c.read_reply();
    if (r.status != want_status) std::printf("        %s -> %d %s\n", path.c_str(), r.status, r.body.c_str());
    CHECK_EQ(r.status, want_status);
    return Json::parse(r.body);
}

// Parses an SSE body into the JSON payloads of its `data:` lines; checks the [DONE] terminator.
std::vector<Json> sse_events(const std::string& body) {
    std::vector<Json> out;
    bool done = false;
    size_t p = 0;
    while (p < body.size()) {
        size_t e = body.find("\n\n", p);
        CHECK(e != std::string::npos);
        std::string ev = body.substr(p, e - p);
        p = e + 2;
        if (ev.rfind(":", 0) == 0) continue;   // comment
        CHECK(ev.rfind("data: ", 0) == 0);
        CHECK(!done);
        std::string data = ev.substr(6);
        if (data == "[DONE]") { done = true; continue; }
        out.push_back(Json::parse(data));
    }
    CHECK(done);
    return out;
}

Json hello_request(bool stream) {
    Json::Object b{{"model", "gpt-oss-20b"},
                   {"messages", Json::Array{Json::Object{{"role", "user"}, {"content", "Say hello in one word."}}}},
                   {"reasoning_effort", "low"}, {"temperature", 0}, {"max_tokens", 200}};
    if (stream) {
        b["stream"] = true;
        b["stream_options"] = Json::Object{{"include_usage", true}};
    }
    return b;
}

} // namespace

CORAL_TEST(server_openai_chat) {
    test::model_dir_or_skip();
    ApiServer s;
    // Non-streaming.
    Json r = post_json(s.port(), "/v1/chat/completions", hello_request(false));
    const Json& ch = r["choices"][0];
    const std::string content = ch["message"]["content"].as_string();
    const std::string reasoning = ch["message"].get_string("reasoning_content", "");
    std::printf("        reasoning: \"%s\"\n        content: \"%s\"\n", reasoning.c_str(), content.c_str());
    CHECK(!content.empty());
    CHECK_EQ(ch["finish_reason"].as_string(), std::string("stop"));
    CHECK_EQ(r["object"].as_string(), std::string("chat.completion"));
    CHECK(r["id"].as_string().rfind("chatcmpl-", 0) == 0);
    const Json& u = r["usage"];
    CHECK_EQ(u["prompt_tokens"].as_int() + u["completion_tokens"].as_int(), u["total_tokens"].as_int());
    {
        harmony::RenderOptions ro;
        ro.reasoning = harmony::ReasoningEffort::Low;
        const auto ids = harmony::render(server_engine().tokenizer(), {{harmony::Role::User, "Say hello in one word."}}, ro);
        CHECK_EQ(u["prompt_tokens"].as_int(), int64_t(ids.size()));
    }
    CHECK(u["completion_tokens"].as_int() > 0);

    // Streaming, same greedy request: the deltas reassemble to the same message.
    Client c(s.port());
    c.request("POST", "/v1/chat/completions", Json(hello_request(true)).dump());
    Reply rep = c.read_reply();
    CHECK_EQ(rep.status, 200);
    CHECK_EQ(rep.headers["content-type"], std::string("text/event-stream"));
    auto evs = sse_events(rep.body);
    CHECK(evs.size() >= 3);
    std::string sc, sr, finish;
    bool saw_role = false, saw_usage = false;
    for (size_t i = 0; i < evs.size(); ++i) {
        const Json& e = evs[i];
        CHECK_EQ(e["object"].as_string(), std::string("chat.completion.chunk"));
        if (e["choices"].size() == 0) {   // usage chunk
            saw_usage = true;
            CHECK_EQ(e["usage"]["completion_tokens"].as_int(), u["completion_tokens"].as_int());
            CHECK_EQ(e["usage"]["prompt_tokens"].as_int(), u["prompt_tokens"].as_int());
            continue;
        }
        const Json& d = e["choices"][0]["delta"];
        if (i == 0) { CHECK_EQ(d["role"].as_string(), std::string("assistant")); saw_role = true; }
        sc += d.get_string("content", "");
        sr += d.get_string("reasoning_content", "");
        if (e["choices"][0]["finish_reason"].is_string()) finish = e["choices"][0]["finish_reason"].as_string();
    }
    CHECK(saw_role && saw_usage);
    CHECK_EQ(finish, std::string("stop"));
    CHECK_EQ(sc, content);
    CHECK_EQ(sr, reasoning);
}

CORAL_TEST(server_openai_completions_and_errors) {
    test::model_dir_or_skip();
    ApiServer s;
    Json r = post_json(s.port(), "/v1/completions",
                       Json::Object{{"prompt", "The capital of France is"}, {"max_tokens", 8}, {"temperature", 0}});
    const Json& ch = r["choices"][0];
    std::printf("        text: \"%s\"\n", ch["text"].as_string().c_str());
    CHECK(!ch["text"].as_string().empty());
    CHECK_EQ(r["object"].as_string(), std::string("text_completion"));
    CHECK(ch["finish_reason"].as_string() == "length" || ch["finish_reason"].as_string() == "stop");
    CHECK(r["usage"]["completion_tokens"].as_int() <= 8);
    CHECK(ch["text"].as_string().find("Paris") != std::string::npos);

    // Validation errors never reach the engine.
    Client c(s.port());
    c.request("POST", "/v1/chat/completions", "{not json");
    Reply a = c.read_reply();
    CHECK_EQ(a.status, 400);
    CHECK_EQ(Json::parse(a.body)["error"]["type"].as_string(), std::string("invalid_request_error"));
    Json bad = post_json(s.port(), "/v1/chat/completions", Json::Object{{"messages", Json::Array{}}}, 400);
    CHECK(bad["error"]["message"].is_string());
    post_json(s.port(), "/v1/chat/completions",
              Json::Object{{"n", 2}, {"messages", Json::Array{Json::Object{{"role", "user"}, {"content", "x"}}}}}, 400);
    Client m(s.port());
    m.request("GET", "/v1/models");
    Json models = Json::parse(m.read_reply().body);
    CHECK_EQ(models["data"][0]["id"].as_string(), std::string("gpt-oss-20b"));
    m.request("GET", "/health");
    CHECK_EQ(Json::parse(m.read_reply().body)["status"].as_string(), std::string("ok"));
}

CORAL_TEST(server_openai_tool_call) {
    test::model_dir_or_skip();
    ApiServer s;
    const std::string tools_json = R"([{"type":"function","function":{"name":"get_weather",
        "description":"Get the current weather for a city.",
        "parameters":{"type":"object","properties":{"location":{"type":"string","description":"City name"},
        "unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}}}])";
    Json body = Json::Object{
        {"messages", Json::Array{Json::Object{{"role", "user"}, {"content", "What's the weather in Paris?"}}}},
        {"tools", Json::parse(tools_json)}, {"reasoning_effort", "low"}, {"temperature", 0}, {"max_tokens", 400}};
    Json r = post_json(s.port(), "/v1/chat/completions", body);
    const Json& ch = r["choices"][0];
    const std::string finish = ch["finish_reason"].as_string();
    if (finish != "tool_calls") {
        std::printf("        model answered without a tool call (finish %s): %s\n", finish.c_str(), r.dump().c_str());
        return;
    }
    const Json& call = ch["message"]["tool_calls"][0];
    CHECK_EQ(call["type"].as_string(), std::string("function"));
    CHECK_EQ(call["function"]["name"].as_string(), std::string("get_weather"));
    CHECK(call["id"].as_string().rfind("call_", 0) == 0);
    const std::string args = call["function"]["arguments"].as_string();
    std::printf("        tool call: get_weather(%s)\n", args.c_str());
    Json a = Json::parse(args);   // must be valid JSON
    CHECK(a.dump().find("Paris") != std::string::npos);

    // Round trip: feed the tool result back; the model should answer in the final channel.
    Json::Array msgs = body["messages"].as_array();
    msgs.push_back(Json::Object{{"role", "assistant"}, {"content", Json()}, {"tool_calls", Json::Array{call}},
                                {"reasoning_content", ch["message"].get_string("reasoning_content", "")}});
    msgs.push_back(Json::Object{{"role", "tool"}, {"tool_call_id", call["id"]}, {"content", R"({"temperature_c": 18, "sky": "sunny"})"}});
    body["messages"] = msgs;
    Json r2 = post_json(s.port(), "/v1/chat/completions", body);
    const Json& ch2 = r2["choices"][0];
    std::printf("        after tool result (%s): \"%s\"\n", ch2["finish_reason"].as_string().c_str(),
                ch2["message"]["content"].is_string() ? ch2["message"]["content"].as_string().c_str() : "(null)");
    CHECK(ch2["finish_reason"].as_string() == "stop" || ch2["finish_reason"].as_string() == "tool_calls");
}
