// OpenAI-compatible API on top of HttpServer and Engine.
//
//   POST /v1/chat/completions  Harmony render -> Engine::generate -> harmony::Parser
//   POST /v1/completions       raw prompt (string or token ids), no Harmony
//   GET  /v1/models, /health
//
// Requests are parsed and validated on the IO thread (bad requests never
// wait behind a generation); generations run one at a time on a single
// worker thread that owns the Engine (FIFO). A queued request whose client
// disconnects is dropped before it starts; a running one is cancelled at the
// next token through the TokenCallback.
#include "coral/server.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <ctime>
#include <deque>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_set>

#include "coral/harmony.h"
#include "coral/json.h"
#include "coral/model.h"
#include "coral/tokenizer.h"

namespace coral {
namespace {

struct ApiError : std::runtime_error {
    int status;
    std::string type, code;
    ApiError(int s, const std::string& msg, std::string c = {}, std::string t = "invalid_request_error")
        : std::runtime_error(msg), status(s), type(std::move(t)), code(std::move(c)) {}
};

std::string random_hex(size_t n) {
    static thread_local std::mt19937_64 rng{(uint64_t(std::random_device{}()) << 32) ^ std::random_device{}()};
    static const char* d = "0123456789abcdef";
    std::string s(n, '0');
    for (auto& c : s) c = d[rng() & 15];
    return s;
}

int64_t unix_now() { return int64_t(std::time(nullptr)); }

// ---- raw JSON spans (keeps the client's key order for tool schemas) --------
size_t skip_ws(std::string_view s, size_t i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    return i;
}
size_t skip_string(std::string_view s, size_t i) {   // s[i] == '"'
    for (++i; i < s.size(); ++i) {
        if (s[i] == '\\') ++i;
        else if (s[i] == '"') return i + 1;
    }
    return std::string_view::npos;
}
size_t skip_value(std::string_view s, size_t i) {
    i = skip_ws(s, i);
    if (i >= s.size()) return std::string_view::npos;
    if (s[i] == '"') return skip_string(s, i);
    if (s[i] == '{' || s[i] == '[') {
        int depth = 0;
        for (; i < s.size(); ++i) {
            if (s[i] == '"') { i = skip_string(s, i); if (i == std::string_view::npos) return i; --i; continue; }
            if (s[i] == '{' || s[i] == '[') ++depth;
            else if ((s[i] == '}' || s[i] == ']') && --depth == 0) return i + 1;
        }
        return std::string_view::npos;
    }
    while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']' && s[i] != ' ' && s[i] != '\n' && s[i] != '\r' && s[i] != '\t') ++i;
    return i;
}
// Raw text of v[key] where `v` is an object; empty if absent.
std::string_view raw_member(std::string_view v, std::string_view key) {
    size_t i = skip_ws(v, 0);
    if (i >= v.size() || v[i] != '{') return {};
    ++i;
    for (;;) {
        i = skip_ws(v, i);
        if (i >= v.size() || v[i] != '"') return {};
        const size_t ks = i;
        i = skip_string(v, i);
        if (i == std::string_view::npos) return {};
        std::string_view k = v.substr(ks + 1, i - ks - 2);
        i = skip_ws(v, i);
        if (i >= v.size() || v[i] != ':') return {};
        const size_t vs = skip_ws(v, i + 1);
        const size_t ve = skip_value(v, vs);
        if (ve == std::string_view::npos) return {};
        if (k == key) return v.substr(vs, ve - vs);
        i = skip_ws(v, ve);
        if (i >= v.size() || v[i] != ',') return {};
        ++i;
    }
}
std::string_view raw_element(std::string_view v, size_t idx) {
    size_t i = skip_ws(v, 0);
    if (i >= v.size() || v[i] != '[') return {};
    ++i;
    for (size_t n = 0;; ++n) {
        const size_t vs = skip_ws(v, i);
        const size_t ve = skip_value(v, vs);
        if (ve == std::string_view::npos || vs >= v.size() || v[vs] == ']') return {};
        if (n == idx) return v.substr(vs, ve - vs);
        i = skip_ws(v, ve);
        if (i >= v.size() || v[i] != ',') return {};
        ++i;
    }
}

// ---- text stop sequences ----------------------------------------------------
// Holds back the longest tail that could still start a stop string; once one
// matches, everything from it on is dropped.
class StopFilter {
public:
    explicit StopFilter(std::vector<std::string> stops) : stops_(std::move(stops)) {}
    bool hit() const { return hit_; }
    std::string push(std::string_view s) {
        if (hit_ || stops_.empty()) return hit_ ? std::string() : std::string(s);
        held_.append(s);
        size_t best = std::string::npos;
        for (const auto& st : stops_) best = std::min(best, held_.find(st));
        if (best != std::string::npos) {
            hit_ = true;
            std::string out = held_.substr(0, best);
            held_.clear();
            return out;
        }
        size_t keep = 0;
        for (const auto& st : stops_)
            for (size_t k = std::min(st.size() - 1, held_.size()); k > keep; --k)
                if (held_.compare(held_.size() - k, k, st, 0, k) == 0) { keep = k; break; }
        std::string out = held_.substr(0, held_.size() - keep);
        held_.erase(0, held_.size() - keep);
        return out;
    }
    std::string flush() { return hit_ ? std::string() : std::exchange(held_, {}); }
private:
    std::vector<std::string> stops_;
    std::string held_;
    bool hit_ = false;
};

// ---- the generation worker --------------------------------------------------
class Worker {
public:
    Worker() : th_([this] { loop(); }) {}
    ~Worker() {
        std::deque<std::function<void()>> dropped;
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
            dropped.swap(q_);
        }
        cv_.notify_all();
        th_.join();
        // `dropped` goes last: pending responses end as their handles are released.
    }
    // `make(ahead)` builds the job, knowing how many jobs are ahead of it.
    // Returns that count.
    size_t submit(const std::function<std::function<void()>(size_t)>& make) {
        std::lock_guard<std::mutex> lk(mu_);
        const size_t ahead = q_.size() + (busy_ ? 1 : 0);
        q_.push_back(make(ahead));
        cv_.notify_one();
        return ahead;
    }
    size_t depth() const { std::lock_guard<std::mutex> lk(mu_); return q_.size() + (busy_ ? 1 : 0); }
private:
    void loop() {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return stop_ || !q_.empty(); });
                if (stop_) return;
                job = std::move(q_.front());
                q_.pop_front();
                busy_ = true;
            }
            try { job(); } catch (const std::exception& e) { std::fprintf(stderr, "[coral] worker: %s\n", e.what()); }
            job = nullptr;
            std::lock_guard<std::mutex> lk(mu_);
            busy_ = false;
        }
    }
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::function<void()>> q_;
    bool busy_ = false, stop_ = false;
    std::thread th_;   // last: started after the other members exist
};

// ---- request parameters -----------------------------------------------------
struct GenParams {
    std::vector<int32_t> prompt;
    std::optional<uint32_t> max_tokens;
    SamplingParams sampling;
    std::vector<std::string> stop;
    bool stream = false;
    bool include_usage = false;
};

const Json& require_object(const Json& j) {
    if (!j.is_object()) throw ApiError(400, "request body must be a JSON object");
    return j;
}

std::string content_text(const Json& c, const char* what) {
    if (c.is_null()) return {};
    if (c.is_string()) return c.as_string();
    if (c.is_array()) {   // content parts
        std::string s;
        for (const Json& p : c.as_array()) {
            if (p.is_string()) { s += p.as_string(); continue; }
            const std::string type = p.get_string("type", "text");
            if (type != "text" && type != "input_text" && type != "output_text")
                throw ApiError(400, std::string("unsupported content part type '") + type + "' in " + what, "unsupported_content");
            if (!p["text"].is_string()) throw ApiError(400, std::string("content part without text in ") + what);
            s += p["text"].as_string();
        }
        return s;
    }
    throw ApiError(400, std::string("invalid content in ") + what);
}

void parse_common(const Json& b, GenParams& g) {
    auto num = [&](const char* k) -> std::optional<double> {
        const Json& v = b[k];
        if (v.is_null()) return std::nullopt;
        if (!v.is_number()) throw ApiError(400, std::string("'") + k + "' must be a number", "invalid_type");
        return v.as_double();
    };
    if (auto n = num("n"); n && *n != 1)
        throw ApiError(400, "only n = 1 is supported", "unsupported_parameter");
    std::optional<double> mt = num("max_completion_tokens");
    if (!mt) mt = num("max_tokens");
    if (mt) {
        if (*mt < 1 || *mt > 1e9) throw ApiError(400, "max_tokens must be >= 1", "invalid_value");
        g.max_tokens = uint32_t(*mt);
    }
    SamplingParams& sp = g.sampling;
    sp.temperature = 1.0f;
    if (auto t = num("temperature")) {
        if (*t < 0 || *t > 2) throw ApiError(400, "temperature must be in [0, 2]", "invalid_value");
        sp.temperature = float(*t);
    }
    if (auto t = num("top_p")) {
        if (*t <= 0 || *t > 1) throw ApiError(400, "top_p must be in (0, 1]", "invalid_value");
        sp.top_p = float(*t);
    }
    if (auto t = num("top_k")) {
        if (*t < 0) throw ApiError(400, "top_k must be >= 0", "invalid_value");
        sp.top_k = int32_t(*t);
    }
    if (auto t = num("repetition_penalty")) {
        if (*t <= 0) throw ApiError(400, "repetition_penalty must be > 0", "invalid_value");
        sp.repetition_penalty = float(*t);
    }
    if (auto t = num("seed")) sp.seed = uint64_t(int64_t(*t));
    const Json& st = b["stop"];
    if (st.is_string()) { if (!st.as_string().empty()) g.stop.push_back(st.as_string()); }
    else if (st.is_array()) {
        for (const Json& s : st.as_array()) {
            if (!s.is_string()) throw ApiError(400, "'stop' must be a string or an array of strings", "invalid_type");
            if (!s.as_string().empty()) g.stop.push_back(s.as_string());
        }
    } else if (!st.is_null()) throw ApiError(400, "'stop' must be a string or an array of strings", "invalid_type");
    const Json& s = b["stream"];
    if (!s.is_null() && !s.is_bool()) throw ApiError(400, "'stream' must be a boolean", "invalid_type");
    g.stream = s.is_bool() && s.as_bool();
    g.include_usage = b["stream_options"].get_bool("include_usage", false);
}

// ---- the API ----------------------------------------------------------------
class Api {
public:
    Api(Engine& e, const ServerOptions& o) : engine_(e), tok_(e.tokenizer()), opts_(o) {
        ret_ = tok_.special("<|return|>");
        call_ = tok_.special("<|call|>");
        eot_ = tok_.special("<|endoftext|>");
        max_pos_ = e.model().config().max_position_embeddings;
    }

    void install(HttpServer& s) {
        Api* self = this;   // routes and jobs never outlive the Api (server attachment)
        s.route("GET", "/health", [self](const HttpRequest& q, HttpResponse& r) { self->health(q, r); });
        s.route("GET", "/v1/models", [self](const HttpRequest& q, HttpResponse& r) { self->models(q, r); });
        s.route("GET", "/v1/models/" + opts_.model_name, [self](const HttpRequest& q, HttpResponse& r) { self->model(q, r); });
        s.route("POST", "/v1/chat/completions", [self](const HttpRequest& q, HttpResponse& r) { self->guard(q, r, true); });
        s.route("POST", "/v1/completions", [self](const HttpRequest& q, HttpResponse& r) { self->guard(q, r, false); });
    }

private:
    static void json_reply(HttpResponse& r, const Json& j, int status = 200) {
        r.status(status);
        r.header("Content-Type", "application/json");
        r.end(j.dump() + "\n");
    }

    void health(const HttpRequest&, HttpResponse& r) {
        json_reply(r, Json::Object{{"status", "ok"}, {"model", opts_.model_name}, {"device", opts_.device_name},
                                   {"queue", int64_t(worker.depth())}});
    }
    Json model_obj() const {
        return Json::Object{{"id", opts_.model_name}, {"object", "model"}, {"created", int64_t(0)}, {"owned_by", "coral"}};
    }
    void models(const HttpRequest&, HttpResponse& r) {
        json_reply(r, Json::Object{{"object", "list"}, {"data", Json::Array{model_obj()}}});
    }
    void model(const HttpRequest&, HttpResponse& r) { json_reply(r, model_obj()); }

    void guard(const HttpRequest& q, HttpResponse& r, bool chat) {
        try {
            Json body;
            try { body = Json::parse(q.body); }
            catch (const std::exception& e) { throw ApiError(400, std::string("invalid JSON body: ") + e.what(), "invalid_json"); }
            require_object(body);
            if (chat) chat_completions(q, body, r);
            else completions(q, body, r);
        } catch (const ApiError& e) {
            send_json_error(r, e.status, e.what(), e.type, e.code);
        } catch (const std::exception& e) {
            send_json_error(r, 400, e.what(), "invalid_request_error", "invalid_request");
        }
    }

    // Fits the prompt and output into the context; fills req.kv_capacity/max_new_tokens.
    void size_request(const GenParams& g, GenerationRequest& req) const {
        const uint32_t limit = opts_.kv_capacity ? std::min(opts_.kv_capacity, max_pos_) : max_pos_;
        const size_t n = g.prompt.size();
        if (n == 0) throw ApiError(400, "empty prompt", "invalid_value");
        if (n >= limit)
            throw ApiError(400, "prompt is " + std::to_string(n) + " tokens; the context holds " + std::to_string(limit),
                           "context_length_exceeded");
        const uint32_t room = uint32_t(limit - n);
        uint32_t want = g.max_tokens ? *g.max_tokens
                        : opts_.kv_capacity ? room
                        : std::max<uint32_t>(uint32_t(std::max<int64_t>(int64_t(Engine::kDefaultKvCapacity) - int64_t(n), 0)), 4096);
        req.max_new_tokens = std::min(want, room);
        req.kv_capacity = opts_.kv_capacity ? limit : 0;
        req.prompt = g.prompt;
        req.sampling = g.sampling;
    }

    static void sse_headers(HttpResponse& r) {
        r.status(200);
        r.header("Content-Type", "text/event-stream");
        r.header("Cache-Control", "no-cache");
        r.header("X-Accel-Buffering", "no");
    }
    static void sse(HttpResponse& r, const Json& j) { r.write("data: " + j.dump() + "\n\n"); }

    static const char* finish_name(FinishReason w) {
        return w == FinishReason::Stop ? "stop" : w == FinishReason::Length ? "length" : "stop";
    }

    static std::string stats_note(const GenerationStats& st, size_t ahead) {
        char b[160];
        std::snprintf(b, sizeof b, "prompt %u tok, completion %u tok, %.1f tok/s, prefill %.0f ms%s", st.prompt_tokens,
                      st.generated_tokens, st.tokens_per_second(), st.prefill_seconds * 1e3,
                      ahead ? (", queued behind " + std::to_string(ahead)).c_str() : "");
        return b;
    }

    Json usage(const GenerationStats& st) const {
        return Json::Object{{"prompt_tokens", int64_t(st.prompt_tokens)},
                            {"completion_tokens", int64_t(st.generated_tokens)},
                            {"total_tokens", int64_t(st.prompt_tokens) + int64_t(st.generated_tokens)}};
    }

    // Queue `fn` on the worker; errors thrown there become 500s (or an SSE error event).
    template <class Fn>
    void enqueue(HttpResponse& r, bool stream, Fn fn) {
        std::shared_ptr<HttpResponse> h = r.defer();
        const size_t ahead = worker.submit([&](size_t n) -> std::function<void()> {
            return [h, stream, fn = std::move(fn), n]() mutable {
            if (h->client_gone()) { h->annotate("dropped from queue: client gone"); h->end(); return; }
            try {
                fn(*h, n);
            } catch (const std::exception& e) {
                if (!h->headers_sent()) send_json_error(*h, 500, e.what(), "server_error", "generation_failed");
                else {
                    if (stream)
                        sse(*h, Json::Object{{"error", Json::Object{{"message", std::string(e.what())},
                                                                    {"type", "server_error"}, {"code", Json()}}}});
                    h->end();
                }
            }
            };
        });
        // Streaming clients learn their queue position at once (SSE comment, ignored by parsers).
        if (stream && ahead > 0) {
            sse_headers(*h);
            h->write(": queued, position " + std::to_string(ahead) + "\n\n");
        }
    }

    // ---- /v1/chat/completions -------------------------------------------
    void chat_completions(const HttpRequest& q, const Json& b, HttpResponse& r) {
        GenParams g;
        parse_common(b, g);

        harmony::RenderOptions ro;
        std::string effort = opts_.default_reasoning;
        if (b["reasoning_effort"].is_string()) effort = b["reasoning_effort"].as_string();
        else if (b["reasoning"].is_object() && b["reasoning"]["effort"].is_string()) effort = b["reasoning"]["effort"].as_string();
        if (effort == "low" || effort == "minimal") ro.reasoning = harmony::ReasoningEffort::Low;
        else if (effort == "medium") ro.reasoning = harmony::ReasoningEffort::Medium;
        else if (effort == "high") ro.reasoning = harmony::ReasoningEffort::High;
        else throw ApiError(400, "reasoning_effort must be low, medium or high", "invalid_value");

        // Tools.
        const Json& tools = b["tools"];
        if (!tools.is_null()) {
            if (!tools.is_array()) throw ApiError(400, "'tools' must be an array", "invalid_type");
            const std::string_view raw_tools = raw_member(q.body, "tools");
            for (size_t i = 0; i < tools.size(); ++i) {
                const Json& t = tools[i];
                if (t.get_string("type", "function") != "function" || !t["function"].is_object())
                    throw ApiError(400, "only function tools are supported", "unsupported_tool");
                const Json& f = t["function"];
                if (!f["name"].is_string() || f["name"].as_string().empty())
                    throw ApiError(400, "tools[" + std::to_string(i) + "].function.name is required", "invalid_value");
                harmony::ToolSpec spec;
                spec.name = f["name"].as_string();
                spec.description = f.get_string("description", "");
                if (f["parameters"].is_object()) {
                    std::string_view raw = raw_member(raw_member(raw_element(raw_tools, i), "function"), "parameters");
                    spec.parameters_json_schema = raw.empty() ? f["parameters"].dump() : std::string(raw);
                }
                ro.tools.push_back(std::move(spec));
            }
        }
        if (b["tool_choice"].is_string() && b["tool_choice"].as_string() == "none") ro.tools.clear();

        // Messages.
        const Json& ms = b["messages"];
        if (!ms.is_array() || ms.size() == 0) throw ApiError(400, "'messages' must be a non-empty array", "invalid_value");
        std::vector<harmony::Message> msgs;
        std::map<std::string, std::string> call_names;   // tool_call_id -> function name
        for (size_t i = 0; i < ms.size(); ++i) {
            const Json& m = ms[i];
            const std::string where = "messages[" + std::to_string(i) + "]";
            if (!m.is_object() || !m["role"].is_string()) throw ApiError(400, where + ".role is required", "invalid_value");
            const std::string role = m["role"].as_string();
            const std::string text = content_text(m["content"], where.c_str());
            using harmony::Role;
            if (role == "system" || role == "developer") {
                msgs.push_back({role == "system" ? Role::System : Role::Developer, text});
            } else if (role == "user") {
                msgs.push_back({Role::User, text});
            } else if (role == "assistant") {
                std::string reasoning;
                if (m["reasoning_content"].is_string()) reasoning = m["reasoning_content"].as_string();
                else if (m["reasoning"].is_string()) reasoning = m["reasoning"].as_string();
                const Json& calls = m["tool_calls"];
                if (calls.is_array() && calls.size() > 0) {
                    if (!text.empty()) {
                        harmony::Message pre{Role::Assistant, text};
                        pre.channel = "commentary";
                        msgs.push_back(pre);
                    }
                    for (size_t k = 0; k < calls.size(); ++k) {
                        const Json& c = calls[k];
                        const Json& f = c["function"];
                        if (!f.is_object() || !f["name"].is_string())
                            throw ApiError(400, where + ".tool_calls[" + std::to_string(k) + "].function.name is required", "invalid_value");
                        const std::string name = f["name"].as_string();
                        if (c["id"].is_string()) call_names[c["id"].as_string()] = name;
                        if (k == 0 && !reasoning.empty()) {
                            harmony::Message a{Role::Assistant, reasoning};
                            a.channel = "analysis";
                            msgs.push_back(a);
                        }
                        harmony::Message call{Role::Assistant};
                        call.content = f["arguments"].is_string() ? f["arguments"].as_string()
                                       : f["arguments"].is_null() ? "{}" : f["arguments"].dump();
                        call.channel = "commentary";
                        call.recipient = name.rfind("functions.", 0) == 0 ? name : "functions." + name;
                        msgs.push_back(std::move(call));
                    }
                } else {
                    if (!reasoning.empty()) {
                        harmony::Message a{Role::Assistant, reasoning};
                        a.channel = "analysis";
                        msgs.push_back(a);
                    }
                    harmony::Message fin{Role::Assistant, text};
                    fin.channel = "final";
                    msgs.push_back(std::move(fin));
                }
            } else if (role == "tool" || role == "function") {
                harmony::Message t{Role::Tool, text};
                if (m["name"].is_string()) t.name = m["name"].as_string();
                else if (m["tool_call_id"].is_string()) {
                    auto it = call_names.find(m["tool_call_id"].as_string());
                    if (it != call_names.end()) t.name = it->second;
                }
                if (t.name && t.name->rfind("functions.", 0) == 0) t.name = t.name->substr(10);
                msgs.push_back(std::move(t));
            } else {
                throw ApiError(400, where + ": unknown role '" + role + "'", "invalid_value");
            }
        }
        try {
            g.prompt = harmony::render(tok_, msgs, ro);
        } catch (const std::invalid_argument& e) {
            throw ApiError(400, e.what(), "invalid_messages");
        }
        GenerationRequest req;
        size_request(g, req);
        req.stop_tokens = {ret_, call_};

        const std::string id = "chatcmpl-" + random_hex(24);
        const int64_t created = unix_now();
        Api* self = this;
        enqueue(r, g.stream, [self, g = std::move(g), req = std::move(req), id, created](HttpResponse& h, size_t ahead) {
            self->run_chat(g, req, id, created, h, ahead);
        });
    }

    void run_chat(const GenParams& g, const GenerationRequest& req, const std::string& id, int64_t created,
                  HttpResponse& h, size_t ahead) {
        const bool stream = g.stream;
        auto chunk = [&](Json delta, Json finish) {
            Json::Object choice{{"index", 0}, {"delta", std::move(delta)}, {"finish_reason", std::move(finish)},
                                {"logprobs", Json()}};
            sse(h, Json::Object{{"id", id}, {"object", "chat.completion.chunk"}, {"created", created},
                                {"model", opts_.model_name}, {"system_fingerprint", "coral"},
                                {"choices", Json::Array{Json(std::move(choice))}}});
        };
        if (stream) {
            if (!h.headers_sent()) sse_headers(h);
            chunk(Json::Object{{"role", "assistant"}, {"content", ""}}, Json());
        }

        std::string content, reasoning;
        Json::Array calls;
        StopFilter filter(g.stop);
        bool stopped_by_text = false;
        harmony::Parser parser(tok_);
        auto emit_content = [&](const std::string& s) {
            if (s.empty()) return;
            content += s;
            if (stream) chunk(Json::Object{{"content", s}}, Json());
        };
        auto cb = [&](int32_t t) -> bool {
            for (auto& ev : parser.push(t)) {
                using K = harmony::Event::Kind;
                if (ev.kind == K::Text) {
                    if (ev.channel == "analysis") {
                        reasoning += ev.text;
                        if (stream) chunk(Json::Object{{"reasoning_content", ev.text}}, Json());
                    } else {
                        emit_content(filter.push(ev.text));
                        if (filter.hit()) stopped_by_text = true;
                    }
                } else if (ev.kind == K::ChannelEnd) {
                    emit_content(filter.flush());
                } else if (ev.kind == K::ToolCall && !ev.recipient.empty()) {
                    std::string name = ev.recipient;
                    if (name.rfind("functions.", 0) == 0) name = name.substr(10);
                    Json call = Json::Object{{"index", int64_t(calls.size())}, {"id", "call_" + random_hex(24)},
                                             {"type", "function"},
                                             {"function", Json::Object{{"name", name}, {"arguments", ev.arguments}}}};
                    if (stream) chunk(Json::Object{{"tool_calls", Json::Array{call}}}, Json());
                    calls.push_back(std::move(call));
                }
            }
            return !stopped_by_text && !h.client_gone();
        };
        GenerationStats st;
        const FinishReason why = engine_.generate(req, cb, &st);
        emit_content(filter.flush());
        h.annotate(stats_note(st, ahead));
        if (h.client_gone()) { h.end(); return; }

        const std::string finish = !calls.empty() ? "tool_calls" : stopped_by_text ? "stop" : finish_name(why);
        if (stream) {
            chunk(Json::Object{}, finish);
            if (g.include_usage)
                sse(h, Json::Object{{"id", id}, {"object", "chat.completion.chunk"}, {"created", created},
                                    {"model", opts_.model_name}, {"choices", Json::Array{}}, {"usage", usage(st)}});
            h.write("data: [DONE]\n\n");
            h.end();
            return;
        }
        Json::Object msg{{"role", "assistant"},
                         {"content", content.empty() && !calls.empty() ? Json() : Json(content)}};
        if (!reasoning.empty()) msg["reasoning_content"] = reasoning;
        if (!calls.empty()) msg["tool_calls"] = Json(std::move(calls));
        json_reply(h, Json::Object{
            {"id", id}, {"object", "chat.completion"}, {"created", created}, {"model", opts_.model_name},
            {"system_fingerprint", "coral"},
            {"choices", Json::Array{Json::Object{{"index", 0}, {"message", Json(std::move(msg))},
                                                 {"finish_reason", finish}, {"logprobs", Json()}}}},
            {"usage", usage(st)}});
    }

    // ---- /v1/completions --------------------------------------------------
    void completions(const HttpRequest&, const Json& b, HttpResponse& r) {
        GenParams g;
        parse_common(b, g);
        const Json& p = b["prompt"];
        auto tokens_of = [&](const Json& arr) {
            std::vector<int32_t> ids;
            for (const Json& x : arr.as_array()) {
                if (!x.is_number() || x.as_double() < 0 || x.as_double() >= double(tok_.vocab_size()))
                    throw ApiError(400, "prompt token ids must be integers in [0, vocab)", "invalid_value");
                ids.push_back(int32_t(x.as_int()));
            }
            return ids;
        };
        if (p.is_string()) g.prompt = tok_.encode(p.as_string(), /*allow_special=*/true);
        else if (p.is_array() && p.size() > 0 && p[0].is_number()) g.prompt = tokens_of(p);
        else if (p.is_array() && p.size() == 1 && p[0].is_string()) g.prompt = tok_.encode(p[0].as_string(), true);
        else if (p.is_array() && p.size() == 1 && p[0].is_array()) g.prompt = tokens_of(p[0]);
        else throw ApiError(400, "'prompt' must be a string or an array of token ids (batches are not supported)", "invalid_value");
        const bool echo = b.get_bool("echo", false);
        const std::string echo_text = echo ? tok_.decode(g.prompt) : std::string();

        GenerationRequest req;
        if (!g.max_tokens) g.max_tokens = 16;   // OpenAI default for completions
        size_request(g, req);
        req.stop_tokens = {ret_, call_};
        if (eot_ >= 0) req.stop_tokens.push_back(eot_);

        const std::string id = "cmpl-" + random_hex(24);
        const int64_t created = unix_now();
        Api* self = this;
        enqueue(r, g.stream, [self, g = std::move(g), req = std::move(req), id, created, echo_text](HttpResponse& h, size_t ahead) {
            self->run_completion(g, req, id, created, echo_text, h, ahead);
        });
    }

    void run_completion(const GenParams& g, const GenerationRequest& req, const std::string& id, int64_t created,
                        const std::string& echo_text, HttpResponse& h, size_t ahead) {
        const bool stream = g.stream;
        auto chunk = [&](const std::string& text, Json finish) {
            sse(h, Json::Object{{"id", id}, {"object", "text_completion"}, {"created", created}, {"model", opts_.model_name},
                                {"choices", Json::Array{Json::Object{{"index", 0}, {"text", text}, {"logprobs", Json()},
                                                                     {"finish_reason", std::move(finish)}}}}});
        };
        if (stream && !h.headers_sent()) sse_headers(h);
        std::string text = echo_text;
        if (stream && !echo_text.empty()) chunk(echo_text, Json());
        StopFilter filter(g.stop);
        Utf8Streamer utf8;
        const std::unordered_set<int32_t> stops(req.stop_tokens.begin(), req.stop_tokens.end());
        auto emit = [&](const std::string& s) {
            if (s.empty()) return;
            text += s;
            if (stream) chunk(s, Json());
        };
        auto cb = [&](int32_t t) -> bool {
            if (stops.count(t)) return true;
            emit(filter.push(utf8.push(tok_.token_bytes(t))));
            return !filter.hit() && !h.client_gone();
        };
        GenerationStats st;
        const FinishReason why = engine_.generate(req, cb, &st);
        emit(filter.push(utf8.flush()));
        emit(filter.flush());
        h.annotate(stats_note(st, ahead));
        if (h.client_gone()) { h.end(); return; }
        const std::string finish = filter.hit() ? "stop" : finish_name(why);
        if (stream) {
            chunk("", finish);
            if (g.include_usage)
                sse(h, Json::Object{{"id", id}, {"object", "text_completion"}, {"created", created},
                                    {"model", opts_.model_name}, {"choices", Json::Array{}}, {"usage", usage(st)}});
            h.write("data: [DONE]\n\n");
            h.end();
            return;
        }
        json_reply(h, Json::Object{
            {"id", id}, {"object", "text_completion"}, {"created", created}, {"model", opts_.model_name},
            {"choices", Json::Array{Json::Object{{"index", 0}, {"text", text}, {"logprobs", Json()}, {"finish_reason", finish}}}},
            {"usage", usage(st)}});
    }

    Engine& engine_;
    const Tokenizer& tok_;
    ServerOptions opts_;
    int32_t ret_ = -1, call_ = -1, eot_ = -1;
    uint32_t max_pos_ = 0;
    Worker worker;   // last: destroyed (joined) first, while everything it uses is alive
};

} // namespace

void install_openai_api(HttpServer& server, Engine& engine, const ServerOptions& opts) {
    auto api = std::make_shared<Api>(engine, opts);
    api->install(server);
    server.attach(api);
}

} // namespace coral
