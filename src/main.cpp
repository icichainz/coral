// coral command-line entry point.
//
//   coral info                     GPU device capabilities
//   coral selftest                 compile kernels, run a dispatch, verify results
//   coral inspect <model_dir>      config + tensor inventory, bind weights to GPU
//   coral run <model_dir> [...]    generate from a prompt (Harmony chat or --raw)
//   coral bench <model_dir> [...]  prefill/decode throughput, achieved GB/s
//   coral serve <model_dir> [...]  OpenAI-compatible HTTP server

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <map>
#include <regex>
#include <string>
#include <vector>

#include "coral/config.h"
#include "coral/engine.h"
#include "coral/gpu.h"
#include "coral/harmony.h"
#include "coral/kernels.h"
#include "coral/model.h"
#include "coral/safetensors.h"
#include "coral/server.h"
#include "coral/tokenizer.h"

using namespace coral;

namespace {

int usage() {
    std::fprintf(stderr,
        "usage: coral <command> [args]\n"
        "  info                    GPU device capabilities\n"
        "  selftest                compile kernels and verify a dispatch\n"
        "  inspect <model_dir>     model config and tensor inventory\n"
        "  run <model_dir> [-p prompt] [-n max_tokens] [--temp t] [--top-p p] [--top-k k]\n"
        "                  [--seed s] [--rep-penalty r] [--ctx N] [--raw] [--reasoning low|medium|high]\n"
        "                  [--system text]\n"
        "                          generate text; prompt from -p or stdin. Harmony chat by\n"
        "                          default (analysis to stderr, final answer to stdout);\n"
        "                          --raw feeds the prompt tokens as-is and prints the continuation\n"
        "  bench <model_dir> [-n tokens] [--ctx N] [--report-every K]\n"
        "                          greedy decode benchmark: ms/token, tok/s, GB/s\n"
        "  serve <model_dir> [--host H] [--port P] [--model-name N] [--ctx N]\n"
        "                  [--reasoning low|medium|high] [--max-connections N]\n"
        "                          OpenAI-compatible HTTP server (/v1/chat/completions,\n"
        "                          /v1/completions, /v1/models, /health); Ctrl-C stops\n");
    return 2;
}

std::string gb(uint64_t b) { char buf[32]; std::snprintf(buf, sizeof buf, "%.2f GB", double(b) / 1e9); return buf; }

int cmd_info() {
    auto dev = gpu::Device::create();
    const auto& i = dev->info();
    std::printf("device                 %s\n", i.name.c_str());
    std::printf("metal 4                %s\n", i.metal4 ? "yes" : "no");
    std::printf("unified memory         %s\n", i.unified_memory ? "yes" : "no");
    std::printf("max buffer             %s\n", gb(i.max_buffer_bytes).c_str());
    std::printf("recommended working set%s\n", (" " + gb(i.recommended_working_set_bytes)).c_str());
    std::printf("simd width             %u\n", i.simd_width);
    std::printf("max threads/threadgroup%s\n", (" " + std::to_string(i.max_threads_per_threadgroup)).c_str());
    std::printf("threadgroup memory     %u bytes\n", i.max_threadgroup_memory_bytes);
    return 0;
}

int cmd_selftest() {
    auto dev = gpu::Device::create();
    double secs = load_kernels(*dev);
    std::printf("kernels compiled in %.3f s\n", secs);

    const uint32_t n = 1 << 20;
    auto a = dev->alloc(n * 2), b = dev->alloc(n * 2), out = dev->alloc(n * 2);
    auto* pa = a.as<uint16_t>(); auto* pb = b.as<uint16_t>();
    for (uint32_t i = 0; i < n; ++i) { pa[i] = f32_to_bf16(float(i % 100)); pb[i] = f32_to_bf16(1.0f); }

    struct { uint32_t n; float scale; } params{n, 2.0f};
    gpu::Kernel k = dev->kernel("smoke_axpy_bf16");
    auto cs = dev->stream();

    // Warm, then time a burst of dispatches to show per-dispatch host+GPU cost.
    cs.begin();
    cs.dispatch_threads(k, gpu::Args().buffer(0, a).buffer(1, b).buffer(2, out).value(3, params), {n}, {256});
    cs.submit_and_wait();

    const int reps = 200;
    auto t0 = std::chrono::steady_clock::now();
    cs.begin();
    for (int r = 0; r < reps; ++r)
        cs.dispatch_threads(k, gpu::Args().buffer(0, a).buffer(1, b).buffer(2, out).value(3, params), {n}, {256});
    double gpu_s = cs.submit_and_wait();
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    size_t bad = 0;
    auto* po = out.as<uint16_t>();
    for (uint32_t i = 0; i < n; ++i) {
        float want = 2.0f * float(i % 100) + 1.0f;
        if (bf16_to_f32(po[i]) != want) { if (bad < 5) std::printf("  mismatch at %u: %g != %g\n", i, bf16_to_f32(po[i]), want); ++bad; }
    }
    std::printf("%d dispatches of %u elements: %.3f ms total (%.1f us/dispatch incl. encode), gpu wait %.3f ms\n",
                reps, n, wall * 1e3, wall * 1e6 / reps, gpu_s * 1e3);
    std::printf("axpy check: %s\n", bad ? "FAILED" : "ok");
    return bad ? 1 : 0;
}

int cmd_inspect(const std::string& dir) {
    ModelConfig cfg = ModelConfig::load(dir);
    std::printf("%s\n\n", cfg.summary().c_str());

    auto t0 = std::chrono::steady_clock::now();
    auto st = Safetensors::open(dir);
    double open_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%zu shards, %zu tensors, %s (headers parsed in %.1f ms)\n",
                st->shards().size(), st->tensors().size(), gb(st->total_bytes()).c_str(), open_s * 1e3);

    // Layer-0 and global tensors, with dtype byte totals across the model.
    std::map<std::string, uint64_t> by_dtype;
    std::regex layer_re("^model\\.layers\\.(\\d+)\\.");
    std::printf("\n%-52s %-6s %-28s %12s\n", "tensor (layer 0 shown once)", "dtype", "shape", "bytes");
    for (const auto& [name, t] : st->tensors()) {
        by_dtype[dtype_name(t.dtype)] += t.nbytes;
        std::smatch m;
        if (std::regex_search(name, m, layer_re) && m[1] != "0") continue;
        std::string shape = "[";
        for (size_t i = 0; i < t.shape.size(); ++i) shape += (i ? "," : "") + std::to_string(t.shape[i]);
        shape += "]";
        std::printf("%-52s %-6s %-28s %12zu\n", name.c_str(), dtype_name(t.dtype), shape.c_str(), t.nbytes);
    }
    std::printf("\nbytes by dtype:\n");
    for (auto& [d, b] : by_dtype) std::printf("  %-6s %s\n", d.c_str(), gb(b).c_str());

    auto dev = gpu::Device::create();
    t0 = std::chrono::steady_clock::now();
    st->bind_gpu(*dev);
    double bind_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\nbound %zu shard(s) to GPU with no copy in %.1f ms\n", st->shards().size(), bind_s * 1e3);

    // Sanity: decode a few MXFP4 scales from layer 0 to confirm the format reading.
    if (st->has("model.layers.0.mlp.experts.gate_up_proj_scales")) {
        auto sc = st->get("model.layers.0.mlp.experts.gate_up_proj_scales");
        const uint8_t* s = sc.data;
        std::printf("layer0 gate_up scales[0..8]: ");
        for (int i = 0; i < 8; ++i) std::printf("2^%d ", int(s[i]) - 127);
        std::printf("\n");
    }
    return 0;
}

// ---------------------------------------------------------------------------
// run / bench
// ---------------------------------------------------------------------------
struct Loaded {
    std::unique_ptr<gpu::Device> dev;
    std::shared_ptr<Model> model;
    std::shared_ptr<Tokenizer> tok;
    std::unique_ptr<Engine> engine;
};

using clk = std::chrono::steady_clock;
double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }

Loaded load_all(const std::string& dir, bool quiet = false) {
    Loaded L;
    L.dev = gpu::Device::create();
    const double kc = load_kernels(*L.dev);
    auto t0 = clk::now();
    L.tok = Tokenizer::load(dir);
    L.model = Model::load(*L.dev, dir);
    const double lt = since(t0);
    L.engine = Engine::create(*L.dev, L.model, L.tok);
    t0 = clk::now();
    L.engine->warmup();
    if (!quiet)
        std::fprintf(stderr, "[coral] %s: kernels %.0f ms, load %.1f s, warmup %.0f ms\n",
                     L.dev->info().name.c_str(), kc * 1e3, lt, since(t0) * 1e3);
    return L;
}

// Tiny argv parser: --flag value / -f value / --switch.
struct Opts {
    std::vector<std::string> pos;
    std::map<std::string, std::string> kv;
    Opts(int argc, char** argv, int from, const std::vector<std::string>& switches) {
        for (int i = from; i < argc; ++i) {
            std::string a = argv[i];
            if (a.size() > 1 && a[0] == '-' && !(a.size() > 1 && std::isdigit(uint8_t(a[1])))) {
                if (std::find(switches.begin(), switches.end(), a) != switches.end()) { kv[a] = "1"; continue; }
                if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
                kv[a] = argv[++i];
            } else {
                pos.push_back(a);
            }
        }
    }
    bool has(const std::string& k) const { return kv.count(k) > 0; }
    std::string get(const std::string& a, const std::string& b, const std::string& def) const {
        if (auto it = kv.find(a); it != kv.end()) return it->second;
        if (auto it = kv.find(b); it != kv.end()) return it->second;
        return def;
    }
};

int cmd_run(int argc, char** argv) {
    Opts o(argc, argv, 2, {"--raw"});
    if (o.pos.size() != 1) return usage();
    const std::string dir = o.pos[0];
    std::string prompt = o.get("-p", "--prompt", "");
    if (prompt.empty() && !o.has("-p") && !o.has("--prompt")) {
        if (isatty(0)) std::fprintf(stderr, "[coral] reading prompt from stdin (end with Ctrl-D)\n");
        prompt.assign(std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>());
    }
    const bool raw = o.has("--raw");

    Loaded L = load_all(dir);
    const Tokenizer& tok = *L.tok;

    GenerationRequest req;
    req.max_new_tokens = uint32_t(std::stoul(o.get("-n", "--max-tokens", raw ? "256" : "2048")));
    req.sampling.temperature = std::stof(o.get("--temp", "-t", "0"));
    req.sampling.top_p = std::stof(o.get("--top-p", "--top_p", "1"));
    req.sampling.top_k = std::stoi(o.get("--top-k", "--top_k", "0"));
    req.sampling.repetition_penalty = std::stof(o.get("--rep-penalty", "--repetition-penalty", "1"));
    req.sampling.seed = std::stoull(o.get("--seed", "-s", "0"));
    req.kv_capacity = uint32_t(std::stoul(o.get("--ctx", "--kv-capacity", "0")));
    for (const char* s : {"<|return|>", "<|call|>"}) req.stop_tokens.push_back(tok.special(s));

    const bool color = isatty(2);
    const char* dim = color ? "\033[2m" : "";
    const char* reset = color ? "\033[0m" : "";

    std::string current_channel;
    harmony::Parser parser(tok);
    Utf8Streamer raw_stream;
    TokenCallback cb;
    if (raw) {
        req.prompt = tok.encode(prompt, /*allow_special=*/true);
        const int32_t eot = tok.special("<|endoftext|>");
        if (eot >= 0) req.stop_tokens.push_back(eot);
        cb = [&](int32_t t) {
            std::string s = raw_stream.push(tok.token_bytes(t));
            std::fwrite(s.data(), 1, s.size(), stdout);
            std::fflush(stdout);
            return true;
        };
    } else {
        harmony::RenderOptions ro;
        const std::string effort = o.get("--reasoning", "-r", "medium");
        if (effort == "low") ro.reasoning = harmony::ReasoningEffort::Low;
        else if (effort == "medium") ro.reasoning = harmony::ReasoningEffort::Medium;
        else if (effort == "high") ro.reasoning = harmony::ReasoningEffort::High;
        else throw std::invalid_argument("--reasoning must be low, medium or high");
        std::vector<harmony::Message> msgs;
        if (o.has("--system")) msgs.push_back({harmony::Role::Developer, o.get("--system", "", "")});
        msgs.push_back({harmony::Role::User, prompt});
        req.prompt = harmony::render(tok, msgs, ro);
        cb = [&](int32_t t) {
            for (const auto& ev : parser.push(t)) {
                using K = harmony::Event::Kind;
                if (ev.kind == K::ChannelStart) {
                    current_channel = ev.channel;
                    if (ev.channel != "final")
                        std::fprintf(stderr, "%s[%s%s%s] ", dim, ev.channel.c_str(),
                                     ev.recipient.empty() ? "" : " -> ", ev.recipient.c_str());
                } else if (ev.kind == K::Text) {
                    if (ev.channel == "final") { std::fwrite(ev.text.data(), 1, ev.text.size(), stdout); std::fflush(stdout); }
                    else { std::fwrite(ev.text.data(), 1, ev.text.size(), stderr); }
                } else if (ev.kind == K::ChannelEnd) {
                    if (ev.channel != "final") std::fprintf(stderr, "%s\n", reset);
                } else if (ev.kind == K::ToolCall) {
                    std::fprintf(stderr, "%s[tool call %s] %s%s\n", dim, ev.recipient.c_str(), ev.arguments.c_str(), reset);
                }
            }
            return true;
        };
    }

    GenerationStats st;
    const FinishReason why = L.engine->generate(req, cb, &st);
    if (raw) { std::string s = raw_stream.flush(); std::fwrite(s.data(), 1, s.size(), stdout); }
    std::fprintf(stdout, "\n");
    std::fflush(stdout);
    const char* reason = why == FinishReason::Stop ? "stop" : why == FinishReason::Length ? "length"
                       : why == FinishReason::Cancelled ? "cancelled" : "error";
    std::fprintf(stderr, "%s[coral] prompt %u tok in %.2f s (%.1f tok/s) | generated %u tok, %.1f tok/s "
                         "(%.2f ms/tok) | finish: %s%s\n",
                 dim, st.prompt_tokens, st.prefill_seconds, st.prompt_tokens / std::max(1e-9, st.prefill_seconds),
                 st.generated_tokens, st.tokens_per_second(),
                 st.decode_steps ? st.decode_seconds * 1e3 / st.decode_steps : 0.0, reason, reset);
    return 0;
}

int cmd_bench(int argc, char** argv) {
    Opts o(argc, argv, 2, {});
    if (o.pos.size() != 1) return usage();
    const uint32_t n = uint32_t(std::stoul(o.get("-n", "--tokens", "256")));
    const uint32_t ctx = uint32_t(std::stoul(o.get("--ctx", "-c", "0")));   // prompt length (0 = short prompt)
    const uint32_t every = uint32_t(std::stoul(o.get("--report-every", "-r", "0")));
    Loaded L = load_all(o.pos[0]);
    Model& m = *L.model;

    // Prompt: a short sentence, or text repeated to exactly `ctx` tokens.
    std::vector<int32_t> prompt = L.tok->encode("The history of the Roman Empire begins with");
    if (ctx > 0) {
        const auto chunk = L.tok->encode(" In the beginning the city was small, and its people farmed the hills along the river.");
        while (prompt.size() < ctx) prompt.insert(prompt.end(), chunk.begin(), chunk.end());
        prompt.resize(ctx);
    }
    const uint32_t cap = uint32_t(prompt.size()) + n + 1;
    KVCache cache = m.new_cache(cap);
    auto cs = L.dev->stream();
    gpu::Buffer logits;

    ForwardStats pf;
    auto t0 = clk::now();
    int32_t tok = m.prefill_argmax(cs, cache, prompt, logits, &pf);
    const double prefill_wall = since(t0);
    std::printf("device        %s\n", L.dev->info().name.c_str());
    std::printf("prefill       %zu tokens in %.1f ms = %.1f tok/s  (token-by-token decode path, %u submits; "
                "GEMM prefill is roadmap step 7)\n",
                prompt.size(), prefill_wall * 1e3, prompt.size() / prefill_wall, pf.submits);

    struct Window { double wall = 0, gpu = 0, enc = 0; uint64_t bytes = 0; size_t disp = 0; uint32_t n = 0; };
    auto report = [&](const char* label, const Window& w, uint32_t ctx_end) {
        if (!w.n) return;
        std::printf("%-13s %u tok @ctx %u: %.2f ms/tok wall = %.1f tok/s | gpu %.2f ms/tok = %.1f GB/s | "
                    "host encode %.0f us/tok | %zu dispatches/tok\n",
                    label, w.n, ctx_end, w.wall * 1e3 / w.n, w.n / w.wall, w.gpu * 1e3 / w.n,
                    double(w.bytes) / w.gpu / 1e9, w.enc * 1e6 / w.n, w.disp / w.n);
    };
    Window total, win;
    for (uint32_t i = 0; i < n; ++i) {
        ForwardStats fs;
        const uint32_t c = cache.length + 1;   // positions attended by this step
        auto t1 = clk::now();
        tok = m.decode_argmax(cs, cache, tok, logits, &fs);
        const double wall = since(t1);
        for (Window* w : {&total, &win}) {
            w->wall += wall; w->gpu += fs.gpu_seconds; w->enc += fs.encode_seconds;
            w->bytes += m.decode_bytes_per_token(c); w->disp += fs.dispatches; w->n++;
        }
        if (every && (i + 1) % every == 0) { report("  window", win, cache.length); win = Window{}; }
    }
    report("decode", total, cache.length);
    std::printf("bytes/token   %.2f GB at ctx %zu, %.2f GB at ctx %u (weights + KV)\n",
                m.decode_bytes_per_token(uint32_t(prompt.size()) + 1) / 1e9, prompt.size() + 1,
                m.decode_bytes_per_token(cache.length) / 1e9, cache.length);
    return 0;
}

int cmd_serve(int argc, char** argv) {
    Opts o(argc, argv, 2, {});
    if (o.pos.size() != 1) return usage();
    ServerOptions so;
    so.host = o.get("--host", "-H", "127.0.0.1");
    so.port = uint16_t(std::stoul(o.get("--port", "-P", "8080")));
    so.model_name = o.get("--model-name", "-m", "gpt-oss-20b");
    so.kv_capacity = uint32_t(std::stoul(o.get("--ctx", "-c", "0")));
    so.default_reasoning = o.get("--reasoning", "-r", "medium");
    so.max_connections = uint32_t(std::stoul(o.get("--max-connections", "", "256")));
    if (so.default_reasoning != "low" && so.default_reasoning != "medium" && so.default_reasoning != "high")
        throw std::invalid_argument("--reasoning must be low, medium or high");
    so.handle_signals = true;
    so.log_requests = true;

    auto server = HttpServer::create(so);   // bind first: fail fast on a busy port
    Loaded L = load_all(o.pos[0]);
    so.device_name = L.dev->info().name;
    install_openai_api(*server, *L.engine, so);
    const bool v6 = so.host.find(':') != std::string::npos;
    std::fprintf(stderr, "[coral] serving %s on http://%s%s%s:%u/v1  (Ctrl-C to stop)\n", so.model_name.c_str(),
                 v6 ? "[" : "", so.host.c_str(), v6 ? "]" : "", unsigned(server->port()));
    server->run();
    server.reset();   // joins the generation worker before the engine goes away
    std::fprintf(stderr, "[coral] stopped\n");
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    std::string cmd = argv[1];
    try {
        if (cmd == "info") return cmd_info();
        if (cmd == "selftest") return cmd_selftest();
        if (cmd == "inspect") { if (argc < 3) return usage(); return cmd_inspect(argv[2]); }
        if (cmd == "run") return cmd_run(argc, argv);
        if (cmd == "bench") return cmd_bench(argc, argv);
        if (cmd == "serve") return cmd_serve(argc, argv);
        return usage();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
