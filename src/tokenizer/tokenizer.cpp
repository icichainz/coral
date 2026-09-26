// Byte-level BPE tokenizer for o200k_harmony (tiktoken-compatible).
//
// Pipeline: [special-token split] -> hand-written o200k pre-tokenizer over
// UTF-8 code points -> whole-piece vocab lookup -> rank-based BPE merge.
// Ranks are token ids (as in tiktoken), so tokenizer.json's merge list is not
// needed: merging pair (a, b) has rank id(bytes(a) + bytes(b)).
#include "coral/tokenizer.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "coral/json.h"
#include "unicode_tables.h"

namespace coral {

// ---------------------------------------------------------------------------
// Utf8Streamer
// ---------------------------------------------------------------------------
namespace {
// Length of the UTF-8 sequence starting with lead byte `b`, or 0 if invalid.
int utf8_len(unsigned char b) {
    if (b < 0x80) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return 0;
}
} // namespace

std::string Utf8Streamer::push(std::string_view bytes) {
    pending_.append(bytes);
    std::string out;
    size_t i = 0;
    while (i < pending_.size()) {
        unsigned char b = pending_[i];
        int len = utf8_len(b);
        if (len == 0) { out += "\xEF\xBF\xBD"; ++i; continue; }         // stray continuation byte
        if (i + len > pending_.size()) {                                // incomplete: wait, unless already broken
            bool ok = true;
            for (size_t k = i + 1; k < pending_.size(); ++k) if ((static_cast<unsigned char>(pending_[k]) & 0xC0) != 0x80) { ok = false; break; }
            if (ok) break;
            out += "\xEF\xBF\xBD"; ++i; continue;
        }
        bool ok = true;
        for (int k = 1; k < len; ++k) if ((static_cast<unsigned char>(pending_[i + k]) & 0xC0) != 0x80) { ok = false; break; }
        if (!ok) { out += "\xEF\xBF\xBD"; ++i; continue; }
        out.append(pending_, i, size_t(len));
        i += size_t(len);
    }
    pending_.erase(0, i);
    return out;
}

std::string Utf8Streamer::flush() {
    std::string out;
    if (!pending_.empty()) { out = "\xEF\xBF\xBD"; pending_.clear(); }
    return out;
}

// ---------------------------------------------------------------------------
// Unicode classes for the pre-tokenizer
// ---------------------------------------------------------------------------
namespace {

constexpr uint32_t kInvalidCp = 0xFFFFFFFFu;  // undecodable byte: classified "other"

// Decode one code point at s[i]; sets len (>= 1). Invalid bytes decode to kInvalidCp.
inline uint32_t decode_cp(std::string_view s, size_t i, size_t& len) {
    unsigned char b0 = static_cast<unsigned char>(s[i]);
    if (b0 < 0x80) { len = 1; return b0; }
    int n = utf8_len(b0);
    if (n < 2 || i + size_t(n) > s.size()) { len = 1; return kInvalidCp; }
    uint32_t cp = b0 & (0x7F >> n);
    for (int k = 1; k < n; ++k) {
        unsigned char b = static_cast<unsigned char>(s[i + size_t(k)]);
        if ((b & 0xC0) != 0x80) { len = 1; return kInvalidCp; }
        cp = (cp << 6) | (b & 0x3F);
    }
    len = size_t(n);
    return cp;
}

// Category lookup: dense table for the BMP, binary search above it.
struct CatTable {
    uint8_t bmp[0x10000];
    CatTable() {
        std::memset(bmp, 0, sizeof bmp);
        for (uint32_t r = 0; r < unicode::kCatRangeCount; ++r) {
            const auto& cr = unicode::kCatRanges[r];
            if (cr.lo > 0xFFFF) break;
            for (uint32_t c = cr.lo; c <= std::min<uint32_t>(cr.hi, 0xFFFF); ++c) bmp[c] = cr.cat;
        }
    }
};
const CatTable& cat_table() { static const CatTable t; return t; }

inline uint8_t category(uint32_t cp) {
    if (cp <= 0xFFFF) return cat_table().bmp[cp];
    if (cp == kInvalidCp) return unicode::kOther;
    const auto* lo = unicode::kCatRanges;
    const auto* hi = unicode::kCatRanges + unicode::kCatRangeCount;
    const auto* it = std::upper_bound(lo, hi, cp, [](uint32_t v, const unicode::CatRange& r) { return v < r.lo; });
    if (it == lo) return unicode::kOther;
    --it;
    return cp <= it->hi ? it->cat : uint8_t(unicode::kOther);
}

inline bool is_ws(uint32_t c) {  // Unicode White_Space
    if (c <= 0x20) return c == 0x20 || (c >= 0x09 && c <= 0x0D);
    if (c < 0x85) return false;
    return c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 ||
           c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

// Class bits derived from the category.
enum : uint8_t { kU = 1, kLow = 2, kLetter = 4, kNum = 8, kWs = 16, kNl = 32 };

inline uint8_t classify(uint32_t cp) {
    uint8_t f = 0;
    switch (category(cp)) {
        case unicode::kLu: case unicode::kLt: f = kU | kLetter; break;
        case unicode::kLl: f = kLow | kLetter; break;
        case unicode::kLm: case unicode::kLo: f = kU | kLow | kLetter; break;
        case unicode::kM: f = kU | kLow; break;
        case unicode::kN: f = kNum; break;
        default: break;
    }
    if (is_ws(cp)) f |= kWs;
    if (cp == '\r' || cp == '\n') f |= kNl;
    return f;
}

struct AsciiClasses {
    uint8_t c[128];
    AsciiClasses() { for (uint32_t i = 0; i < 128; ++i) c[i] = classify(i); }
};
const AsciiClasses& ascii_classes() { static const AsciiClasses t; return t; }

// Cursor over the text that yields (class bits, byte length) per code point.
struct Cursor {
    std::string_view s;
    const uint8_t* ascii;
    inline uint8_t at(size_t i, size_t& len) const {
        unsigned char b = static_cast<unsigned char>(s[i]);
        if (b < 0x80) { len = 1; return ascii[b]; }
        return classify(decode_cp(s, i, len));
    }
    inline uint32_t cp_at(size_t i, size_t& len) const { return decode_cp(s, i, len); }
    // Start of the code point preceding byte offset i (i > 0).
    inline size_t prev(size_t i) const {
        size_t j = i - 1;
        size_t floor = i >= 4 ? i - 4 : 0;
        while (j > floor && (static_cast<unsigned char>(s[j]) & 0xC0) == 0x80) --j;
        size_t len;
        decode_cp(s, j, len);
        return (j + len == i) ? j : i - 1;  // invalid bytes are single units
    }
};

// Prefix class of branches 1/2: [^\r\n\p{L}\p{N}]
inline bool is_prefix(uint8_t f) { return !(f & (kNl | kLetter | kNum)); }

// (?i:'s|'t|'re|'ve|'m|'ll|'d) at e; returns end after the suffix (or e).
inline size_t match_suffix(const Cursor& c, size_t e) {
    const std::string_view s = c.s;
    if (e >= s.size() || s[e] != '\'') return e;
    auto lower = [&](size_t i) -> int {  // ASCII lowercase, 's' for U+017F, else -1
        if (i >= s.size()) return -1;
        unsigned char b = static_cast<unsigned char>(s[i]);
        if (b < 0x80) return (b >= 'A' && b <= 'Z') ? b + 32 : b;
        if (b == 0xC5 && i + 1 < s.size() && static_cast<unsigned char>(s[i + 1]) == 0xBF) return 0x17F;
        return -1;
    };
    int a = lower(e + 1);
    if (a == 's' || a == 0x17F) return e + 1 + (a == 's' ? 1 : 2);
    if (a == 't' || a == 'm' || a == 'd') return e + 2;
    int b = lower(e + 2);
    if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return e + 3;
    return e;
}

// Greedy run of code points whose class has any bit of `mask`; returns end.
inline size_t run(const Cursor& c, size_t i, uint8_t mask, size_t* count = nullptr) {
    size_t n = 0, len;
    while (i < c.s.size() && (c.at(i, len) & mask)) { i += len; ++n; }
    if (count) *count = n;
    return i;
}

// Branch 1 body: [U]*[L]+ suffix?   Returns end or 0 on failure (s > 0 on success).
inline size_t body1(const Cursor& c, size_t s) {
    size_t k = run(c, s, kU);
    for (size_t j = k;;) {
        size_t cnt;
        size_t e = run(c, j, kLow, &cnt);
        if (cnt > 0) return match_suffix(c, e);
        if (j == s) return 0;
        j = c.prev(j);
    }
}

// Branch 2 body: [U]+[L]* suffix?
inline size_t body2(const Cursor& c, size_t s) {
    size_t cnt;
    size_t k = run(c, s, kU, &cnt);
    if (cnt == 0) return 0;
    return match_suffix(c, run(c, k, kLow));
}

// Length-agnostic o200k pre-tokenizer: returns the end of the pre-token at p.
size_t next_pretoken(const Cursor& c, size_t p) {
    const size_t n = c.s.size();
    size_t len0;
    uint8_t f0 = c.at(p, len0);

    // 1 / 2: letters with optional one-char prefix.
    if (f0 & (kU | kLow) || (p + len0 < n && is_prefix(f0))) {
        size_t e;
        if (is_prefix(f0) && p + len0 < n && (e = body1(c, p + len0))) return e;
        if ((f0 & (kU | kLow)) && (e = body1(c, p))) return e;
        if (is_prefix(f0) && p + len0 < n && (e = body2(c, p + len0))) return e;
        if ((f0 & kU) && (e = body2(c, p))) return e;
    }
    // 3: \p{N}{1,3}
    if (f0 & kNum) {
        size_t i = p + len0, cnt = 1, len;
        while (cnt < 3 && i < n && (c.at(i, len) & kNum)) { i += len; ++cnt; }
        return i;
    }
    // 4:  ?[^\s\p{L}\p{N}]+[\r\n/]*
    {
        auto cls4 = [](uint8_t f) { return !(f & (kWs | kLetter | kNum)); };
        size_t q = p;
        if (c.s[p] == ' ' && p + 1 < n) q = p + 1;
        size_t len;
        if (q < n && cls4(c.at(q, len))) {
            size_t i = q;
            while (i < n && cls4(c.at(i, len))) i += len;
            while (i < n && (c.s[i] == '\r' || c.s[i] == '\n' || c.s[i] == '/')) ++i;
            return i;
        }
    }
    // 5/6/7: whitespace.
    if (f0 & kWs) {
        size_t last_nl = SIZE_MAX, i = p, cnt = 0, last_start = p, len;
        while (i < n) {
            uint8_t f = c.at(i, len);
            if (!(f & kWs)) break;
            if (f & kNl) last_nl = i;
            last_start = i;
            i += len; ++cnt;
        }
        if (last_nl != SIZE_MAX) return last_nl + 1;         // \s*[\r\n]+
        if (i == n) return i;                                // \s+(?!\S) at end of text
        if (cnt >= 2) return last_start;                     // \s+(?!\S): leave one for the next word
        return i;                                            // \s+
    }
    return p + len0;  // unreachable for valid input; keeps progress guaranteed
}

// GPT-2 byte <-> unicode mapping.
struct ByteMap {
    int32_t cp_to_byte[324];
    ByteMap() {
        std::fill(std::begin(cp_to_byte), std::end(cp_to_byte), -1);
        int extra = 0;
        for (int b = 0; b < 256; ++b) {
            bool direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            int cp = direct ? b : 256 + extra++;
            cp_to_byte[cp] = b;
        }
    }
};

inline uint64_t hash_bytes(std::string_view s) {
    uint64_t h = 0xcbf29ce484222325ull ^ (s.size() * 0x9E3779B97F4A7C15ull);
    size_t i = 0;
    for (; i + 8 <= s.size(); i += 8) {
        uint64_t w; std::memcpy(&w, s.data() + i, 8);
        h = (h ^ w) * 0x100000001b3ull; h ^= h >> 29;
    }
    uint64_t w = 0;
    std::memcpy(&w, s.data() + i, s.size() - i);
    h = (h ^ w) * 0x100000001b3ull;
    h ^= h >> 32; h *= 0xff51afd7ed558ccdull; h ^= h >> 33;
    return h;
}

std::string read_file(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + p.string());
    std::ostringstream ss; ss << f.rdbuf();
    return ss.str();
}

// ---------------------------------------------------------------------------
// BpeTokenizer
// ---------------------------------------------------------------------------
class BpeTokenizer final : public Tokenizer {
public:
    explicit BpeTokenizer(const std::string& model_dir) { load(model_dir); }

    std::vector<int32_t> encode(std::string_view text, bool allow_special) const override {
        std::vector<int32_t> out;
        out.reserve(text.size() / 3 + 4);
        if (!allow_special) { encode_ordinary(text, out); return out; }
        size_t start = 0, i = 0;
        while ((i = text.find("<|", i)) != std::string_view::npos) {
            size_t close = text.find("|>", i + 2);
            if (close == std::string_view::npos) break;
            auto it = special_ids_.find(std::string(text.substr(i, close + 2 - i)));
            if (it == special_ids_.end()) { i += 2; continue; }
            encode_ordinary(text.substr(start, i - start), out);
            out.push_back(it->second);
            i = start = close + 2;
        }
        encode_ordinary(text.substr(start), out);
        return out;
    }

    std::string decode(std::span<const int32_t> ids) const override {
        std::string raw;
        for (int32_t id : ids) raw.append(token_bytes(id));
        Utf8Streamer s;
        std::string out = s.push(raw);
        out += s.flush();
        return out;
    }

    std::string_view token_bytes(int32_t id) const override {
        if (id < 0 || size_t(id) >= offsets_.size() - 1) return {};
        return std::string_view(arena_).substr(offsets_[size_t(id)], offsets_[size_t(id) + 1] - offsets_[size_t(id)]);
    }

    int32_t special(std::string_view name) const override {
        auto it = special_ids_.find(std::string(name));
        return it == special_ids_.end() ? -1 : it->second;
    }

    size_t vocab_size() const override { return offsets_.size() - 1; }

private:
    std::string arena_;                       // concatenated token bytes
    std::vector<uint32_t> offsets_;           // id -> [offsets_[id], offsets_[id+1])
    std::vector<int32_t> table_;              // open-addressing hash: bytes -> ordinary id
    uint64_t mask_ = 0;
    std::unordered_map<std::string, int32_t> special_ids_;
    uint8_t ascii_[128];

    mutable std::mutex cache_mu_;
    mutable std::unordered_map<std::string, std::vector<int32_t>> cache_;  // slow-path pieces
    static constexpr size_t kCacheMax = 1 << 16;

    int32_t lookup(std::string_view b) const {
        for (uint64_t h = hash_bytes(b) & mask_;; h = (h + 1) & mask_) {
            int32_t id = table_[h];
            if (id < 0) return -1;
            if (token_bytes(id) == b) return id;
        }
    }

    void load(const std::string& model_dir) {
        namespace fs = std::filesystem;
        fs::path dir(model_dir);
        Json tj = Json::parse(read_file(dir / "tokenizer.json"));
        const Json& model = tj["model"];
        if (model.get_string("type", "") != "BPE") throw std::runtime_error("tokenizer.json: expected BPE model");

        ByteMap bm;
        const auto& vocab = model["vocab"].as_object();
        size_t n = 0;
        for (const auto& [k, v] : vocab) n = std::max<size_t>(n, size_t(v.as_int()) + 1);
        std::vector<std::string> bytes(n);
        std::vector<bool> ordinary(n, false);
        for (const auto& [k, v] : vocab) {
            std::string b;
            for (size_t i = 0; i < k.size();) {
                size_t len;
                uint32_t cp = decode_cp(k, i, len);
                if (cp >= 324 || bm.cp_to_byte[cp] < 0) throw std::runtime_error("tokenizer.json: non byte-level vocab entry");
                b.push_back(char(bm.cp_to_byte[cp]));
                i += len;
            }
            bytes[size_t(v.as_int())] = std::move(b);
            ordinary[size_t(v.as_int())] = true;
        }
        std::vector<std::pair<std::string, int32_t>> specials;
        for (const Json& t : tj["added_tokens"].as_array())
            specials.emplace_back(t.get_string("content", ""), int32_t(t.get_int("id", -1)));
        size_t total = n;
        for (auto& [name, id] : specials) total = std::max(total, size_t(id) + 1);
        // The model's embedding table may extend past the named specials
        // (o200k_harmony: 201088); those ids are <|reserved_N|>.
        if (fs::exists(dir / "config.json")) {
            Json cfg = Json::parse(read_file(dir / "config.json"));
            total = std::max(total, size_t(cfg.get_int("vocab_size", 0)));
        }
        bytes.resize(total);
        ordinary.resize(total, false);
        std::vector<bool> named(total, false);
        for (auto& [name, id] : specials) {
            if (id < 0) continue;
            bytes[size_t(id)] = name;
            named[size_t(id)] = true;
            ordinary[size_t(id)] = false;
            special_ids_[name] = id;
        }
        for (size_t id = n; id < total; ++id) {
            if (named[id]) continue;
            std::string name = "<|reserved_" + std::to_string(id) + "|>";
            bytes[id] = name;
            special_ids_.emplace(name, int32_t(id));
        }

        offsets_.resize(total + 1);
        size_t sz = 0;
        for (auto& b : bytes) sz += b.size();
        arena_.reserve(sz);
        for (size_t id = 0; id < total; ++id) { offsets_[id] = uint32_t(arena_.size()); arena_ += bytes[id]; }
        offsets_[total] = uint32_t(arena_.size());

        size_t cap = 1;
        while (cap < n * 2) cap <<= 1;
        table_.assign(cap, -1);
        mask_ = cap - 1;
        for (size_t id = 0; id < total; ++id) {
            if (!ordinary[id]) continue;
            std::string_view b = token_bytes(int32_t(id));
            uint64_t h = hash_bytes(b) & mask_;
            while (table_[h] >= 0) h = (h + 1) & mask_;
            table_[h] = int32_t(id);
        }
        for (int b = 0; b < 256; ++b) {
            char ch = char(b);
            if (lookup(std::string_view(&ch, 1)) < 0) throw std::runtime_error("tokenizer.json: missing single-byte token");
        }
        std::memcpy(ascii_, ascii_classes().c, sizeof ascii_);
    }

    void encode_ordinary(std::string_view text, std::vector<int32_t>& out) const {
        Cursor c{text, ascii_};
        size_t p = 0;
        while (p < text.size()) {
            size_t e = next_pretoken(c, p);
            encode_piece(text.substr(p, e - p), out);
            p = e;
        }
    }

    void encode_piece(std::string_view piece, std::vector<int32_t>& out) const {
        int32_t id = lookup(piece);
        if (id >= 0) { out.push_back(id); return; }
        {
            std::lock_guard<std::mutex> lk(cache_mu_);
            auto it = cache_.find(std::string(piece));
            if (it != cache_.end()) { out.insert(out.end(), it->second.begin(), it->second.end()); return; }
        }
        std::vector<int32_t> ids;
        if (piece.size() <= 64) bpe_small(piece, ids); else bpe_large(piece, ids);
        out.insert(out.end(), ids.begin(), ids.end());
        std::lock_guard<std::mutex> lk(cache_mu_);
        if (cache_.size() >= kCacheMax) cache_.clear();
        cache_.emplace(std::string(piece), std::move(ids));
    }

    uint32_t rank(std::string_view b) const {
        int32_t id = lookup(b);
        return id < 0 ? UINT32_MAX : uint32_t(id);
    }

    // tiktoken's _byte_pair_merge (quadratic; fine for short pieces).
    void bpe_small(std::string_view piece, std::vector<int32_t>& ids) const {
        struct Part { uint32_t start, rank; };
        Part parts[66];
        size_t np = 0;
        const size_t n = piece.size();
        for (size_t i = 0; i + 1 < n; ++i) parts[np++] = {uint32_t(i), rank(piece.substr(i, 2))};
        parts[np++] = {uint32_t(n - 1), UINT32_MAX};
        parts[np++] = {uint32_t(n), UINT32_MAX};
        auto get_rank = [&](size_t i) -> uint32_t {
            if (i + 3 < np) return rank(piece.substr(parts[i].start, parts[i + 3].start - parts[i].start));
            return UINT32_MAX;
        };
        for (;;) {
            uint32_t best = UINT32_MAX; size_t bi = 0;
            for (size_t i = 0; i + 1 < np; ++i) if (parts[i].rank < best) { best = parts[i].rank; bi = i; }
            if (best == UINT32_MAX) break;
            if (bi > 0) parts[bi - 1].rank = get_rank(bi - 1);
            parts[bi].rank = get_rank(bi);
            std::memmove(&parts[bi + 1], &parts[bi + 2], (np - bi - 2) * sizeof(Part));
            --np;
        }
        for (size_t i = 0; i + 1 < np; ++i) {
            int32_t id = lookup(piece.substr(parts[i].start, parts[i + 1].start - parts[i].start));
            ids.push_back(id);
        }
    }

    // Same merge order via a min-heap keyed (rank, start): O(n log n).
    void bpe_large(std::string_view piece, std::vector<int32_t>& ids) const {
        const size_t n = piece.size();
        std::vector<uint32_t> start(n + 1), next(n + 1), prev(n + 1), rk(n + 1, UINT32_MAX);
        std::vector<bool> alive(n + 1, true);
        for (size_t i = 0; i <= n; ++i) { start[i] = uint32_t(i); next[i] = uint32_t(i + 1); prev[i] = uint32_t(i ? i - 1 : 0); }
        auto pair_rank = [&](size_t i) -> uint32_t {  // merge node i with its successor
            if (i >= n) return UINT32_MAX;
            size_t j = next[i];
            if (j >= n) return UINT32_MAX;
            size_t k = next[j];
            return rank(piece.substr(start[i], start[k] - start[i]));
        };
        using E = std::pair<uint64_t, uint32_t>;  // (rank << 32 | start, node)
        std::priority_queue<E, std::vector<E>, std::greater<E>> pq;
        for (size_t i = 0; i < n; ++i) {
            rk[i] = pair_rank(i);
            if (rk[i] != UINT32_MAX) pq.push({(uint64_t(rk[i]) << 32) | start[i], uint32_t(i)});
        }
        while (!pq.empty()) {
            auto [key, i] = pq.top(); pq.pop();
            if (!alive[i] || rk[i] != uint32_t(key >> 32)) continue;
            size_t j = next[i];
            alive[j] = false;
            next[i] = next[j];
            if (next[j] <= n) prev[next[j]] = i;
            rk[i] = pair_rank(i);
            if (rk[i] != UINT32_MAX) pq.push({(uint64_t(rk[i]) << 32) | start[i], i});
            if (i > 0) {
                size_t p = prev[i];
                rk[p] = pair_rank(p);
                if (rk[p] != UINT32_MAX) pq.push({(uint64_t(rk[p]) << 32) | start[p], uint32_t(p)});
            }
        }
        for (size_t i = 0; i < n; i = next[i])
            ids.push_back(lookup(piece.substr(start[i], start[next[i]] - start[i])));
    }
};

} // namespace

std::unique_ptr<Tokenizer> Tokenizer::load(const std::string& model_dir) {
    return std::make_unique<BpeTokenizer>(model_dir);
}

} // namespace coral
