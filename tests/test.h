// Minimal test framework: no dependencies, one binary, filter by substring.
#pragma once

#include <cmath>
#include <cstdio>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace coral::test {

struct Case { std::string name; std::function<void()> fn; };
std::vector<Case>& registry();

struct Registrar { Registrar(const char* name, std::function<void()> fn) { registry().push_back({name, std::move(fn)}); } };

struct Failure : std::runtime_error { using std::runtime_error::runtime_error; };

#define CORAL_TEST(name) \
    static void test_##name(); \
    static ::coral::test::Registrar reg_##name(#name, test_##name); \
    static void test_##name()

#define CHECK(cond) do { if (!(cond)) { std::ostringstream _s; _s << __FILE__ << ":" << __LINE__ << " CHECK failed: " #cond; throw ::coral::test::Failure(_s.str()); } } while (0)
#define CHECK_EQ(a, b) do { auto _a = (a); auto _b = (b); if (!(_a == _b)) { std::ostringstream _s; _s << __FILE__ << ":" << __LINE__ << " CHECK_EQ failed: " #a " == " #b " (" << _a << " vs " << _b << ")"; throw ::coral::test::Failure(_s.str()); } } while (0)
#define CHECK_NEAR(a, b, tol) do { double _a = double(a), _b = double(b); if (std::fabs(_a - _b) > (tol)) { std::ostringstream _s; _s << __FILE__ << ":" << __LINE__ << " CHECK_NEAR failed: " #a " ~ " #b " (" << _a << " vs " << _b << ", tol " << (tol) << ")"; throw ::coral::test::Failure(_s.str()); } } while (0)
#define CHECK_THROWS(expr) do { bool _t = false; try { (void)(expr); } catch (...) { _t = true; } if (!_t) { std::ostringstream _s; _s << __FILE__ << ":" << __LINE__ << " expected exception: " #expr; throw ::coral::test::Failure(_s.str()); } } while (0)

// Skip the current test (e.g. model weights not present).
struct Skip : std::runtime_error { using std::runtime_error::runtime_error; };
#define SKIP(reason) throw ::coral::test::Skip(reason)

// Model directory for weight-dependent tests: $CORAL_MODEL_DIR or models/gpt-oss-20b.
std::string model_dir_or_skip();

} // namespace coral::test
