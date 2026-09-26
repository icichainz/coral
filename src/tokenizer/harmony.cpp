// Harmony chat format: render (mirrors chat_template.jinja) and parse.
#include "coral/harmony.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace coral::harmony {

namespace {

// ---------------------------------------------------------------------------
// Order-preserving JSON value: tool schemas are rendered in property order,
// and arguments are re-serialized like Python's json.dumps (the template's
// `tojson`), so coral::Json (sorted std::map) cannot be used here.
// ---------------------------------------------------------------------------
struct OJson {
    enum class T { Null, Bool, Number, String, Array, Object } t = T::Null;
    bool b = false;
    std::string s;  // String: decoded UTF-8; Number: source text
    std::vector<OJson> arr;
    std::vector<std::pair<std::string, OJson>> obj;

    const OJson* get(std::string_view k) const {
        if (t != T::Object) return nullptr;
        for (const auto& [key, v] : obj) if (key == k) return &v;
        return nullptr;
    }
    bool truthy() const {  // Jinja/Python truthiness
        switch (t) {
            case T::Null: return false;
            case T::Bool: return b;
            case T::Number: return std::strtod(s.c_str(), nullptr) != 0.0;
            case T::String: return !s.empty();
            case T::Array: return !arr.empty();
            case T::Object: return !obj.empty();
        }
        return false;
    }
    bool is_str(std::string_view v) const { return t == T::String && s == v; }
};

struct OParser {
    std::string_view s;
    size_t i = 0;
    [[noreturn]] void fail(const char* m) const { throw std::invalid_argument(std::string("json: ") + m + " at " + std::to_string(i)); }
    void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i; }
    static void put_utf8(std::string& o, uint32_t cp) {
        if (cp < 0x80) o += char(cp);
        else if (cp < 0x800) { o += char(0xC0 | (cp >> 6)); o += char(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { o += char(0xE0 | (cp >> 12)); o += char(0x80 | ((cp >> 6) & 0x3F)); o += char(0x80 | (cp & 0x3F)); }
        else { o += char(0xF0 | (cp >> 18)); o += char(0x80 | ((cp >> 12) & 0x3F)); o += char(0x80 | ((cp >> 6) & 0x3F)); o += char(0x80 | (cp & 0x3F)); }
    }
    uint32_t hex4() {
        if (i + 4 > s.size()) fail("truncated \\u");
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            char c = s[i++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= uint32_t(c - '0');
            else if (c >= 'a' && c <= 'f') v |= uint32_t(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= uint32_t(c - 'A' + 10);
            else fail("bad \\u");
        }
        return v;
    }
    std::string str() {
        if (s[i] != '"') fail("expected string");
        ++i;
        std::string o;
        while (true) {
            if (i >= s.size()) fail("unterminated string");
            char c = s[i++];
            if (c == '"') break;
            if (c != '\\') { o += c; continue; }
            if (i >= s.size()) fail("bad escape");
            char e = s[i++];
            switch (e) {
                case '"': o += '"'; break;
                case '\\': o += '\\'; break;
                case '/': o += '/'; break;
                case 'b': o += '\b'; break;
                case 'f': o += '\f'; break;
                case 'n': o += '\n'; break;
                case 'r': o += '\r'; break;
                case 't': o += '\t'; break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp <= 0xDBFF && i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u') {
                        size_t save = i; i += 2;
                        uint32_t lo = hex4();
                        if (lo >= 0xDC00 && lo <= 0xDFFF) cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        else i = save;
                    }
                    put_utf8(o, cp);
                    break;
                }
                default: fail("bad escape");
            }
        }
        return o;
    }
    OJson value() {
        ws();
        if (i >= s.size()) fail("unexpected end");
        OJson v;
        char c = s[i];
        if (c == '{') {
            v.t = OJson::T::Object; ++i; ws();
            if (i < s.size() && s[i] == '}') { ++i; return v; }
            while (true) {
                ws();
                std::string k = str();
                ws();
                if (i >= s.size() || s[i] != ':') fail("expected ':'");
                ++i;
                OJson val = value();
                bool replaced = false;  // Python dicts keep the last duplicate, at the first position
                for (auto& kv : v.obj) if (kv.first == k) { kv.second = val; replaced = true; }
                if (!replaced) v.obj.emplace_back(std::move(k), std::move(val));
                ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == '}') { ++i; return v; }
                fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            v.t = OJson::T::Array; ++i; ws();
            if (i < s.size() && s[i] == ']') { ++i; return v; }
            while (true) {
                v.arr.push_back(value());
                ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == ']') { ++i; return v; }
                fail("expected ',' or ']'");
            }
        }
        if (c == '"') { v.t = OJson::T::String; v.s = str(); return v; }
        if (s.substr(i, 4) == "true") { i += 4; v.t = OJson::T::Bool; v.b = true; return v; }
        if (s.substr(i, 5) == "false") { i += 5; v.t = OJson::T::Bool; return v; }
        if (s.substr(i, 4) == "null") { i += 4; return v; }
        if (c == '-' || (c >= '0' && c <= '9')) {
            size_t st = i;
            if (s[i] == '-') ++i;
            while (i < s.size() && ((s[i] >= '0' && s[i] <= '9') || s[i] == '.' || s[i] == 'e' || s[i] == 'E' || s[i] == '+' || s[i] == '-')) ++i;
            v.t = OJson::T::Number; v.s = std::string(s.substr(st, i - st));
            return v;
        }
        fail("unexpected character");
    }
};

OJson parse_ojson(std::string_view text) {
    OParser p{text};
    OJson v = p.value();
    p.ws();
    if (p.i != text.size()) p.fail("trailing characters");
    return v;
}

// Python float repr (shortest round-trip, 'r' format rules).
std::string py_float(double v) {
    if (std::isnan(v)) return "NaN";
    if (std::isinf(v)) return v < 0 ? "-Infinity" : "Infinity";
    if (v == 0) return std::signbit(v) ? "-0.0" : "0.0";
    char buf[64];
    int prec = 1;
    for (; prec <= 17; ++prec) {
        std::snprintf(buf, sizeof buf, "%.*e", prec - 1, v);
        if (std::strtod(buf, nullptr) == v) break;
    }
    // buf = [-]d[.ddd]e[+-]XX
    std::string m(buf);
    bool neg = m[0] == '-';
    if (neg) m.erase(0, 1);
    size_t epos = m.find('e');
    int exp = std::atoi(m.c_str() + epos + 1);
    std::string digits;
    for (size_t k = 0; k < epos; ++k) if (m[k] != '.') digits += m[k];
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    std::string out = neg ? "-" : "";
    if (exp >= -4 && exp < 16) {
        if (exp < 0) {
            out += "0." + std::string(size_t(-exp - 1), '0') + digits;
        } else if (size_t(exp) + 1 >= digits.size()) {
            out += digits + std::string(size_t(exp) + 1 - digits.size(), '0') + ".0";
        } else {
            out += digits.substr(0, size_t(exp) + 1) + "." + digits.substr(size_t(exp) + 1);
        }
    } else {
        out += digits.substr(0, 1);
        if (digits.size() > 1) out += "." + digits.substr(1);
        char e[16];
        std::snprintf(e, sizeof e, "e%c%02d", exp < 0 ? '-' : '+', std::abs(exp));
        out += e;
    }
    return out;
}

std::string py_number(const std::string& src) {
    bool is_float = src.find_first_of(".eE") != std::string::npos;
    if (!is_float) {
        std::string d = src;  // Python int: drop leading '-' of zero
        if (d == "-0") return "0";
        return d;
    }
    return py_float(std::strtod(src.c_str(), nullptr));
}

// json.dumps(str, ensure_ascii=False)
void py_quote(std::string& o, std::string_view s) {
    o += '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            case '\b': o += "\\b"; break;
            case '\f': o += "\\f"; break;
            default:
                if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o += char(c);
        }
    }
    o += '"';
}

// json.dumps(value, ensure_ascii=False) with default separators.
void py_dump(std::string& o, const OJson& v) {
    switch (v.t) {
        case OJson::T::Null: o += "null"; break;
        case OJson::T::Bool: o += v.b ? "true" : "false"; break;
        case OJson::T::Number: o += py_number(v.s); break;
        case OJson::T::String: py_quote(o, v.s); break;
        case OJson::T::Array:
            o += '[';
            for (size_t k = 0; k < v.arr.size(); ++k) { if (k) o += ", "; py_dump(o, v.arr[k]); }
            o += ']';
            break;
        case OJson::T::Object:
            o += '{';
            for (size_t k = 0; k < v.obj.size(); ++k) {
                if (k) o += ", ";
                py_quote(o, v.obj[k].first);
                o += ": ";
                py_dump(o, v.obj[k].second);
            }
            o += '}';
            break;
    }
}

std::string py_dumps(const OJson& v) { std::string o; py_dump(o, v); return o; }

// Jinja string conversion for `"..." + value` (strings as-is, else JSON-ish).
std::string jinja_str(const OJson& v) {
    if (v.t == OJson::T::String) return v.s;
    if (v.t == OJson::T::Null) return "None";
    if (v.t == OJson::T::Bool) return v.b ? "True" : "False";
    return py_dumps(v);
}

// ---------------------------------------------------------------------------
// TypeScript rendering of JSON schemas (render_typescript_type macro).
// ---------------------------------------------------------------------------
bool in_required(const OJson* req, const std::string& name) {
    if (!req || req->t != OJson::T::Array) return false;
    for (const auto& r : req->arr) if (r.is_str(name)) return true;
    return false;
}

std::string ts_type(const OJson& spec) {
    const OJson* type = spec.get("type");
    auto nullable = [&] { const OJson* n = spec.get("nullable"); return n && n->truthy(); };
    if (type && type->is_str("array")) {
        std::string out;
        const OJson* items = spec.get("items");
        if (items && items->truthy()) {
            const OJson* it = items->get("type");
            if (it && it->is_str("string")) out = "string[]";
            else if (it && it->is_str("number")) out = "number[]";
            else if (it && it->is_str("integer")) out = "number[]";
            else if (it && it->is_str("boolean")) out = "boolean[]";
            else {
                std::string inner = ts_type(*items);
                // Jinja's |length counts code points.
                size_t cps = 0;
                for (unsigned char c : inner) if ((c & 0xC0) != 0x80) ++cps;
                out = (inner == "object | object" || cps > 50) ? "any[]" : inner + "[]";
            }
        } else {
            out = "any[]";
        }
        if (nullable()) out += " | null";
        return out;
    }
    if (type && type->t == OJson::T::Array && !type->arr.empty()) {
        std::string out;
        for (size_t k = 0; k < type->arr.size(); ++k) { if (k) out += " | "; out += jinja_str(type->arr[k]); }
        return out;
    }
    const OJson* one_of = spec.get("oneOf");
    if (one_of && one_of->truthy()) {
        // The template's has_object_variants flag is set inside a for loop and
        // never escapes Jinja's loop scope, so the union is always rendered.
        std::string out;
        const auto& vars = one_of->arr;
        for (size_t k = 0; k < vars.size(); ++k) {
            const OJson& var = vars[k];
            out += ts_type(var);
            const OJson* d = var.get("description");
            if (d && d->truthy()) out += "// " + jinja_str(*d);
            if (const OJson* def = var.get("default")) out += "                    // default: " + py_dumps(*def);
            if (k + 1 < vars.size()) out += " | \n";
        }
        return out;
    }
    if (type && type->is_str("string")) {
        const OJson* en = spec.get("enum");
        if (en && en->truthy()) {
            std::string out = "\"";
            for (size_t k = 0; k < en->arr.size(); ++k) { if (k) out += "\" | \""; out += jinja_str(en->arr[k]); }
            return out + "\"";
        }
        return nullable() ? "string | null" : "string";
    }
    if (type && (type->is_str("number") || type->is_str("integer"))) return "number";
    if (type && type->is_str("boolean")) return "boolean";
    if (type && type->is_str("object")) {
        const OJson* props = spec.get("properties");
        if (!props || !props->truthy()) return "object";
        const OJson* req = spec.get("required");
        std::string out = "{\n";
        for (size_t k = 0; k < props->obj.size(); ++k) {
            const auto& [name, ps] = props->obj[k];
            out += name;
            if (!in_required(req, name)) out += "?";
            out += ": \n                ";
            out += ts_type(ps);
            if (k + 1 < props->obj.size()) out += ", ";
        }
        return out + "}";
    }
    return "any";
}

std::string render_tools_namespace(const std::vector<ToolSpec>& tools) {
    std::string o = "## functions\n\nnamespace functions {\n\n";
    for (const auto& tool : tools) {
        o += "// " + tool.description + "\n";
        o += "type " + tool.name + " = ";
        OJson params;
        if (!tool.parameters_json_schema.empty()) params = parse_ojson(tool.parameters_json_schema);
        const OJson* props = params.get("properties");
        if (params.truthy() && props && props->truthy()) {
            const OJson* req = params.get("required");
            o += "(_: {\n";
            for (const auto& [name, ps] : props->obj) {
                const OJson* d = ps.get("description");
                if (d && d->truthy()) o += "// " + jinja_str(*d) + "\n";
                o += name;
                if (!in_required(req, name)) o += "?";
                o += ": ";
                o += ts_type(ps);
                if (const OJson* def = ps.get("default")) {
                    const OJson* en = ps.get("enum");
                    const OJson* one_of = ps.get("oneOf");
                    if (en && en->truthy()) o += ", // default: " + jinja_str(*def);
                    else if (one_of && one_of->truthy()) o += "// default: " + jinja_str(*def);
                    else o += ", // default: " + py_dumps(*def);
                }
                o += ",\n";
            }
            o += "}) => any;\n\n";
        } else {
            o += "() => any;\n\n";
        }
    }
    o += "} // namespace functions";
    return o;
}

std::string today() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[16];
    std::strftime(buf, sizeof buf, "%Y-%m-%d", &tm);
    return buf;
}

const char* effort_name(ReasoningEffort e) {
    switch (e) {
        case ReasoningEffort::Low: return "low";
        case ReasoningEffort::Medium: return "medium";
        case ReasoningEffort::High: return "high";
    }
    return "medium";
}

// Collects text and special tokens; text between two specials is encoded as
// one segment (tiktoken splits on specials the same way).
struct Builder {
    const Tokenizer* tok;
    std::vector<int32_t> ids;
    std::string text, pending;
    void t(std::string_view s) { pending += s; text += s; }
    void sp(const char* name) {
        flush();
        text += name;
        if (!tok) return;
        int32_t id = tok->special(name);
        if (id < 0) throw std::runtime_error(std::string("harmony: tokenizer lacks special token ") + name);
        ids.push_back(id);
    }
    void flush() {
        if (tok && !pending.empty()) {
            auto v = tok->encode(pending, false);
            ids.insert(ids.end(), v.begin(), v.end());
        }
        pending.clear();
    }
};

bool is_tool_call(const Message& m) { return m.role == Role::Assistant && m.recipient && !m.recipient->empty(); }
bool is_analysis(const Message& m) { return m.role == Role::Assistant && !is_tool_call(m) && m.channel && *m.channel == "analysis"; }

std::string qualify(const std::string& name) {
    return name.find('.') == std::string::npos ? "functions." + name : name;
}

void build(Builder& b, const std::vector<Message>& msgs, const RenderOptions& opts) {
    // System message.
    b.sp("<|start|>"); b.t("system"); b.sp("<|message|>");
    b.t(opts.model_identity + "\n");
    b.t("Knowledge cutoff: " + opts.knowledge_cutoff + "\n");
    b.t("Current date: " + (opts.current_date.empty() ? today() : opts.current_date) + "\n\n");
    b.t(std::string("Reasoning: ") + effort_name(opts.reasoning) + "\n\n");
    b.t("# Valid channels: analysis, commentary, final. Channel must be included for every message.");
    if (!opts.tools.empty()) b.t("\nCalls to these tools must go to the commentary channel: 'functions'.");
    b.sp("<|end|>");

    // Developer message.
    size_t first = 0;
    std::string dev;
    if (!msgs.empty() && (msgs[0].role == Role::Developer || msgs[0].role == Role::System)) { dev = msgs[0].content; first = 1; }
    if (!dev.empty() || !opts.tools.empty()) {
        b.sp("<|start|>"); b.t("developer"); b.sp("<|message|>");
        if (!dev.empty()) b.t("# Instructions\n\n" + dev + "\n\n");
        if (!opts.tools.empty()) b.t("# Tools\n\n" + render_tools_namespace(opts.tools));
        b.sp("<|end|>");
    }

    std::string last_tool;  // qualified name of the most recent tool call
    for (size_t i = first; i < msgs.size(); ++i) {
        const Message& m = msgs[i];
        switch (m.role) {
            case Role::User:
                b.sp("<|start|>"); b.t("user"); b.sp("<|message|>"); b.t(m.content); b.sp("<|end|>");
                break;
            case Role::Assistant: {
                if (is_tool_call(m)) {
                    bool future_final = false;
                    for (size_t j = i + 1; j < msgs.size(); ++j)
                        if (msgs[j].role == Role::Assistant && !is_tool_call(msgs[j]) && !is_analysis(msgs[j])) future_final = true;
                    if (i > first && is_analysis(msgs[i - 1]) && !msgs[i - 1].content.empty() &&
                        (!future_final || !opts.include_final_only_in_history)) {
                        b.sp("<|start|>"); b.t("assistant"); b.sp("<|channel|>"); b.t("analysis");
                        b.sp("<|message|>"); b.t(msgs[i - 1].content); b.sp("<|end|>");
                    }
                    std::string recipient = qualify(*m.recipient);
                    b.sp("<|start|>"); b.t("assistant to=" + recipient); b.sp("<|channel|>");
                    std::string ctype = m.content_type ? *m.content_type : "json";
                    if (opts.constrain_token) { b.t("commentary "); b.sp("<|constrain|>"); b.t(ctype); }
                    else b.t("commentary " + ctype);
                    b.sp("<|message|>");
                    std::string args;
                    try { args = py_dumps(parse_ojson(m.content)); }
                    catch (const std::invalid_argument&) { py_quote(args, m.content); }  // not JSON: a string argument
                    b.t(args);
                    b.sp("<|call|>");
                    last_tool = recipient;
                } else if (is_analysis(m)) {
                    bool feeds_call = i + 1 < msgs.size() && is_tool_call(msgs[i + 1]);
                    if (!feeds_call && !opts.include_final_only_in_history) {
                        b.sp("<|start|>"); b.t("assistant"); b.sp("<|channel|>"); b.t("analysis");
                        b.sp("<|message|>"); b.t(m.content); b.sp("<|end|>");
                    }
                } else {
                    b.sp("<|start|>"); b.t("assistant"); b.sp("<|channel|>");
                    b.t(m.channel && !m.channel->empty() ? *m.channel : "final");
                    b.sp("<|message|>"); b.t(m.content); b.sp("<|end|>");
                    last_tool.clear();
                }
                break;
            }
            case Role::Tool: {
                std::string name = m.name && !m.name->empty() ? qualify(*m.name) : last_tool;
                if (name.empty()) throw std::invalid_argument("harmony: tool message without a preceding tool call");
                b.sp("<|start|>"); b.t(name + " to=assistant"); b.sp("<|channel|>"); b.t("commentary");
                b.sp("<|message|>");
                std::string q;
                py_quote(q, m.content);
                b.t(q);
                b.sp("<|end|>");
                break;
            }
            case Role::System:
            case Role::Developer:
                break;  // only a leading system/developer message is rendered (template behaviour)
        }
    }
    b.sp("<|start|>"); b.t("assistant");
    b.flush();
}

} // namespace

std::vector<int32_t> render(const Tokenizer& tok, const std::vector<Message>& messages, const RenderOptions& opts) {
    Builder b{&tok, {}, {}, {}};
    build(b, messages, opts);
    return std::move(b.ids);
}

std::string render_text(const std::vector<Message>& messages, const RenderOptions& opts) {
    Builder b{nullptr, {}, {}, {}};
    build(b, messages, opts);
    return std::move(b.text);
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------
Parser::Parser(const Tokenizer& tok)
    : tok_(tok),
      start_(tok.special("<|start|>")), end_(tok.special("<|end|>")), message_(tok.special("<|message|>")),
      channel_tok_(tok.special("<|channel|>")), constrain_(tok.special("<|constrain|>")),
      return_(tok.special("<|return|>")), call_(tok.special("<|call|>")) {}

void Parser::parse_header() {
    // header_: [role/recipient text] \x01C [channel text] \x01K [constrain text]
    std::string pre, chan, cons;
    std::string* cur = &pre;
    for (size_t i = 0; i < header_.size(); ++i) {
        if (header_[i] == '\x01' && i + 1 < header_.size()) {
            cur = header_[i + 1] == 'C' ? &chan : &cons;
            ++i;
            continue;
        }
        *cur += header_[i];
    }
    channel_.clear(); recipient_.clear(); content_type_.clear();
    auto words = [](const std::string& s) {
        std::vector<std::string> w;
        size_t i = 0;
        while (i < s.size()) {
            while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n')) ++i;
            size_t st = i;
            while (i < s.size() && !(s[i] == ' ' || s[i] == '\t' || s[i] == '\n')) ++i;
            if (i > st) w.push_back(s.substr(st, i - st));
        }
        return w;
    };
    for (const auto& w : words(pre))
        if (w.rfind("to=", 0) == 0) recipient_ = w.substr(3);
    for (const auto& w : words(chan)) {
        if (w.rfind("to=", 0) == 0) recipient_ = w.substr(3);
        else if (channel_.empty()) channel_ = w;
        else if (content_type_.empty()) content_type_ = w;
    }
    for (const auto& w : words(cons)) {
        if (w.rfind("to=", 0) == 0) recipient_ = w.substr(3);
        else if (content_type_.empty()) content_type_ = w;
    }
    header_.clear();
}

void Parser::end_message(std::vector<Event>& ev, bool tool_call) {
    if (recipient_.empty()) {
        std::string rest = text_.flush();
        if (!rest.empty()) ev.push_back({Event::Kind::Text, channel_, rest, {}, {}, {}});
    }
    ev.push_back({Event::Kind::ChannelEnd, channel_, {}, recipient_, {}, content_type_});
    if (tool_call || !recipient_.empty())
        ev.push_back({Event::Kind::ToolCall, channel_, {}, recipient_, args_, content_type_});
    text_ = Utf8Streamer{};
    args_.clear();
    state_ = State::Header;
    header_.clear();
}

std::vector<Event> Parser::push(int32_t token) {
    std::vector<Event> ev;
    if (finished_) return ev;
    if (token == return_ || token == call_) {
        if (state_ == State::Body) end_message(ev, token == call_);
        ev.push_back({Event::Kind::Stop, {}, {}, {}, {}, {}});
        finished_ = true;
        return ev;
    }
    if (state_ == State::Header) {
        if (token == start_) header_.clear();
        else if (token == channel_tok_) header_ += "\x01" "C";
        else if (token == constrain_) header_ += "\x01" "K";
        else if (token == message_) {
            parse_header();
            state_ = State::Body;
            ev.push_back({Event::Kind::ChannelStart, channel_, {}, recipient_, {}, content_type_});
        } else if (token == end_) {
            header_.clear();
        } else {
            header_ += tok_.token_bytes(token);
        }
        return ev;
    }
    // Body.
    if (token == end_) { end_message(ev, false); return ev; }
    if (token == start_) { end_message(ev, false); return ev; }  // missing <|end|>: recover
    std::string_view bytes = tok_.token_bytes(token);
    if (bytes.size() > 4 && bytes.substr(0, 2) == "<|" && tok_.special(bytes) == token) return ev;  // stray special: ignore
    if (!recipient_.empty()) { args_ += bytes; return ev; }
    std::string out = text_.push(bytes);
    if (!out.empty()) ev.push_back({Event::Kind::Text, channel_, std::move(out), {}, {}, {}});
    return ev;
}

} // namespace coral::harmony
