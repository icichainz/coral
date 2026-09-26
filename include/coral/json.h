// Minimal JSON DOM: enough for safetensors headers, config.json,
// tokenizer.json and the OpenAI-compatible HTTP API. No external deps.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace coral {

class Json {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };
    using Array = std::vector<Json>;
    using Object = std::map<std::string, Json>;  // ordered for stable output

    Json() = default;
    Json(std::nullptr_t) {}
    Json(bool b) : type_(Type::Bool), bool_(b) {}
    Json(int v) : type_(Type::Number), num_(v) {}
    Json(int64_t v) : type_(Type::Number), num_(double(v)) {}
    Json(uint64_t v) : type_(Type::Number), num_(double(v)) {}
    Json(double v) : type_(Type::Number), num_(v) {}
    Json(const char* s) : type_(Type::String), str_(s) {}
    Json(std::string s) : type_(Type::String), str_(std::move(s)) {}
    Json(Array a) : type_(Type::Array), arr_(std::make_shared<Array>(std::move(a))) {}
    Json(Object o) : type_(Type::Object), obj_(std::make_shared<Object>(std::move(o))) {}

    static Json parse(std::string_view text);         // throws std::runtime_error
    std::string dump(int indent = -1) const;           // compact if indent < 0

    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::Null; }
    bool is_bool() const { return type_ == Type::Bool; }
    bool is_number() const { return type_ == Type::Number; }
    bool is_string() const { return type_ == Type::String; }
    bool is_array() const { return type_ == Type::Array; }
    bool is_object() const { return type_ == Type::Object; }

    bool               as_bool() const;
    double             as_double() const;
    int64_t            as_int() const;
    const std::string& as_string() const;
    const Array&       as_array() const;
    const Object&      as_object() const;
    Array&             as_array();
    Object&            as_object();

    // Object access. `operator[]` on a const object returns a shared Null on miss.
    const Json& operator[](std::string_view key) const;
    const Json& operator[](size_t i) const;
    const Json& operator[](int i) const { return (*this)[size_t(i)]; }
    Json&       operator[](size_t i) { return as_array().at(i); }        // throws if out of range
    Json&       operator[](int i) { return as_array().at(size_t(i)); }
    Json&       operator[](const std::string& key);   // creates key (object) as needed
    bool        contains(std::string_view key) const;
    size_t      size() const;

    // Typed getters with defaults.
    int64_t     get_int(std::string_view key, int64_t def) const;
    double      get_double(std::string_view key, double def) const;
    bool        get_bool(std::string_view key, bool def) const;
    std::string get_string(std::string_view key, const std::string& def) const;

private:
    Type type_ = Type::Null;
    bool bool_ = false;
    double num_ = 0.0;
    std::string str_;
    std::shared_ptr<Array> arr_;
    std::shared_ptr<Object> obj_;
};

} // namespace coral
