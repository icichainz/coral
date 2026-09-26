#include "test.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>

namespace coral::test {

std::vector<Case>& registry() { static std::vector<Case> r; return r; }

std::string model_dir_or_skip() {
    const char* env = std::getenv("CORAL_MODEL_DIR");
    std::string dir = env ? env : "models/gpt-oss-20b";
    if (!std::filesystem::exists(std::filesystem::path(dir) / "config.json"))
        SKIP("model not found at " + dir + " (set CORAL_MODEL_DIR)");
    return dir;
}

} // namespace coral::test

int main(int argc, char** argv) {
    const char* filter = argc > 1 ? argv[1] : nullptr;
    int passed = 0, failed = 0, skipped = 0;
    for (const auto& c : coral::test::registry()) {
        if (filter && c.name.find(filter) == std::string::npos) continue;
        auto t0 = std::chrono::steady_clock::now();
        try {
            c.fn();
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::printf("  ok    %-40s %8.1f ms\n", c.name.c_str(), ms);
            ++passed;
        } catch (const coral::test::Skip& s) {
            std::printf("  skip  %-40s (%s)\n", c.name.c_str(), s.what());
            ++skipped;
        } catch (const std::exception& e) {
            std::printf("  FAIL  %-40s\n        %s\n", c.name.c_str(), e.what());
            ++failed;
        }
    }
    std::printf("\n%d passed, %d failed, %d skipped\n", passed, failed, skipped);
    return failed ? 1 : 0;
}
