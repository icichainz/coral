#include "coral/json.h"

#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace coral {

namespace {

struct Parser {
    std::string_view s;
    size_t i = 0;

    [[noreturn]] void fail(const std::string& what) const {
        throw std::runtime_error("json: " + what + " at offset " + std::to_string(i));
    }
    void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i; }
    char peek() const { return i < s.size() ? s[i] : '\0'; }
    bool consume(char c) { if (peek() == c) { ++i; return true; } return false; }
    void expect(char c) { if (!consume(c)) fail(std::string("expected '") + c + "'"); }

    static void put_utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) out += char(cp);
        else if (cp < 0x800) { out += char(0xC0 | (cp >> 6)); out += char(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { out += char(0xE0 | (cp >> 12)); out += char(0x80 | ((cp >> 6) & 0x3F)); out += char(0x80 | (cp & 0x3F)); }
        else { out += char(0xF0 | (cp >> 18)); out += char(0x80 | ((cp >> 12) & 0x3F)); out += char(0x80 | ((cp >> 6) & 0x3F)); out += char(0x80 | (cp & 0x3F)); }
    }
    uint32_t hex4() {
        if (i + 4 > s.size()) fail("truncated \\u escape");
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            char c = s[i++]; v <<= 4;
            if (c >= '0' && c <= '9') v |= c - '0';
            else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
            else fail("bad hex digit");
        }
        return v;
    }

    std::string string() {
        expect('"');
        std::string out;
        while (true) {
            if (i >= s.size()) fail("unterminated string");
            char c = s[i++];
            if (c == '"') break;
            if (c != '\\') { out += c; continue; }
            if (i >= s.size()) fail("bad escape");
            char e = s[i++];
            switch (e) {
                case '"': out += '"'; break;   case '\\': out += '\\'; break; case '/': out += '/'; break;
                case 'b': out += '\b'; break;  case 'f': out += '\f'; break;  case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;  case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp <= 0xDBFF) {  // high surrogate
                        if (i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u') {
                            i += 2;
                            uint32_t lo = hex4();
                            if (lo >= 0xDC00 && lo <= 0xDFFF) cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            else fail("bad low surrogate");
                        } else fail("lone high surrogate");
                    }
                    put_utf8(out, cp);
                    break;
                }
                default: fail("bad escape character");
            }
        }
        return out;
    }

    Json number() {
        size_t start = i;
        if (peek() == '-') ++i;
        while (i < s.size() && ((s[i] >= '0' && s[i] <= '9') || s[i] == '.' || s[i] == 'e' || s[i] == 'E' || s[i] == '+' || s[i] == '-')) ++i;
        std::string tok(s.substr(start, i - start));
        char* end = nullptr;
        double v = std::strtod(tok.c_str(), &end);
        if (end == tok.c_str()) fail("bad number");
        return Json(v);
    }

    Json value(int depth = 0) {
        if (depth > 512) fail("nesting too deep");
        ws();
        char c = peek();
        if (c == '{') {
            ++i; Json::Object obj; ws();
            if (consume('}')) return Json(std::move(obj));
            while (true) {
                ws(); std::string k = string(); ws(); expect(':');
                obj.emplace(std::move(k), value(depth + 1));
                ws();
                if (consume(',')) continue;
                expect('}'); break;
            }
            return Json(std::move(obj));
        }
        if (c == '[') {
            ++i; Json::Array arr; ws();
            if (consume(']')) return Json(std::move(arr));
            while (true) {
                arr.push_back(value(depth + 1)); ws();
                if (consume(',')) continue;
                expect(']'); break;
            }
            return Json(std::move(arr));
        }
        if (c == '"') return Json(string());
        if (c == 't') { if (s.substr(i, 4) != "true") fail("bad literal"); i += 4; return Json(true); }
        if (c == 'f') { if (s.substr(i, 5) != "false") fail("bad literal"); i += 5; return Json(false); }
        if (c == 'n') { if (s.substr(i, 4) != "null") fail("bad literal"); i += 4; return Json(); }
        if (c == '-' || (c >= '0' && c <= '9')) return number();
        fail("unexpected character");
    }
};

void dump_string(std::string& out, const std::string& s) {
    out += '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break; case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break; case '\r': out += "\\r"; break; case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break; case '\f': out += "\\f"; break;
            default:
                if (c < 0x20) { char buf[8]; snprintf(buf, sizeof buf, "\\u%04x", c); out += buf; }
                else out += char(c);
        }
    }
    out += '"';
}

void dump_impl(const Json& j, std::string& out, int indent, int level) {
    auto nl = [&](int lvl) { if (indent >= 0) { out += '\n'; out.append(size_t(lvl * indent), ' '); } };
    switch (j.type()) {
        case Json::Type::Null: out += "null"; break;
        case Json::Type::Bool: out += j.as_bool() ? "true" : "false"; break;
        case Json::Type::Number: {
            double v = j.as_double();
            if (std::isfinite(v) && v == std::floor(v) && std::fabs(v) < 1e15) out += std::to_string(int64_t(v));
            else { char buf[32]; snprintf(buf, sizeof buf, "%.17g", v); out += buf; }
            break;
        }
        case Json::Type::String: dump_string(out, j.as_string()); break;
        case Json::Type::Array: {
            const auto& a = j.as_array();
            if (a.empty()) { out += "[]"; break; }
            out += '[';
            for (size_t k = 0; k < a.size(); ++k) { if (k) out += ','; nl(level + 1); dump_impl(a[k], out, indent, level + 1); }
            nl(level); out += ']';
            break;
        }
        case Json::Type::Object: {
            const auto& o = j.as_object();
            if (o.empty()) { out += "{}"; break; }
            out += '{'; bool first = true;
            for (const auto& [k, v] : o) {
                if (!first) out += ','; first = false;
                nl(level + 1); dump_string(out, k); out += indent >= 0 ? ": " : ":"; dump_impl(v, out, indent, level + 1);
            }
            nl(level); out += '}';
            break;
        }
    }
}

const Json& null_json() { static const Json n; return n; }

} // namespace

Json Json::parse(std::string_view text) {
    Parser p{text};
    Json v = p.value();
    p.ws();
    if (p.i != text.size()) p.fail("trailing characters");
    return v;
}

std::string Json::dump(int indent) const { std::string out; dump_impl(*this, out, indent, 0); return out; }

bool Json::as_bool() const { if (type_ != Type::Bool) throw std::runtime_error("json: not a bool"); return bool_; }
double Json::as_double() const { if (type_ != Type::Number) throw std::runtime_error("json: not a number"); return num_; }
int64_t Json::as_int() const { return int64_t(as_double()); }
const std::string& Json::as_string() const { if (type_ != Type::String) throw std::runtime_error("json: not a string"); return str_; }
const Json::Array& Json::as_array() const { if (type_ != Type::Array) throw std::runtime_error("json: not an array"); return *arr_; }
const Json::Object& Json::as_object() const { if (type_ != Type::Object) throw std::runtime_error("json: not an object"); return *obj_; }
Json::Array& Json::as_array() { if (type_ != Type::Array) throw std::runtime_error("json: not an array"); return *arr_; }
Json::Object& Json::as_object() { if (type_ != Type::Object) throw std::runtime_error("json: not an object"); return *obj_; }

const Json& Json::operator[](std::string_view key) const {
    if (type_ != Type::Object) return null_json();
    auto it = obj_->find(std::string(key));
    return it == obj_->end() ? null_json() : it->second;
}
const Json& Json::operator[](size_t i) const {
    if (type_ != Type::Array || i >= arr_->size()) return null_json();
    return (*arr_)[i];
}
Json& Json::operator[](const std::string& key) {
    if (type_ == Type::Null) { type_ = Type::Object; obj_ = std::make_shared<Object>(); }
    if (type_ != Type::Object) throw std::runtime_error("json: not an object");
    return (*obj_)[key];
}
bool Json::contains(std::string_view key) const { return type_ == Type::Object && obj_->count(std::string(key)) > 0; }
size_t Json::size() const {
    if (type_ == Type::Array) return arr_->size();
    if (type_ == Type::Object) return obj_->size();
    return 0;
}

int64_t Json::get_int(std::string_view k, int64_t d) const { const Json& v = (*this)[k]; return v.is_number() ? v.as_int() : d; }
double Json::get_double(std::string_view k, double d) const { const Json& v = (*this)[k]; return v.is_number() ? v.as_double() : d; }
bool Json::get_bool(std::string_view k, bool d) const { const Json& v = (*this)[k]; return v.is_bool() ? v.as_bool() : d; }
std::string Json::get_string(std::string_view k, const std::string& d) const { const Json& v = (*this)[k]; return v.is_string() ? v.as_string() : d; }

} // namespace coral
