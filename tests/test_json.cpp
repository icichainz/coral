#include "test.h"
#include "coral/json.h"

using coral::Json;

CORAL_TEST(json_parse_basic) {
    Json j = Json::parse(R"({"a": 1, "b": [true, null, "x\n\u00e9\ud83d\ude00"], "c": {"d": -2.5e3}})");
    CHECK(j.is_object());
    CHECK_EQ(j["a"].as_int(), 1);
    CHECK(j["b"].is_array());
    CHECK_EQ(j["b"].size(), size_t(3));
    CHECK(j["b"][0].as_bool());
    CHECK(j["b"][1].is_null());
    CHECK_EQ(j["b"][2].as_string(), std::string("x\n\xC3\xA9\xF0\x9F\x98\x80"));
    CHECK_NEAR(j["c"]["d"].as_double(), -2500.0, 1e-9);
    CHECK(j["missing"].is_null());
    CHECK_EQ(j.get_int("a", 7), 1);
    CHECK_EQ(j.get_int("zz", 7), 7);
}

CORAL_TEST(json_roundtrip) {
    Json j = Json::parse(R"({"k":"v\"q","n":[1,2,3],"o":{}})");
    std::string s = j.dump();
    CHECK_EQ(s, std::string(R"({"k":"v\"q","n":[1,2,3],"o":{}})"));
    Json again = Json::parse(s);
    CHECK_EQ(again["n"][2].as_int(), 3);
}

CORAL_TEST(json_errors) {
    CHECK_THROWS(Json::parse("{"));
    CHECK_THROWS(Json::parse("[1,]"));
    CHECK_THROWS(Json::parse("tru"));
    CHECK_THROWS(Json::parse("{\"a\":1} x"));
    CHECK_THROWS(Json::parse("\"\\ud800\""));
}

CORAL_TEST(json_build) {
    Json j;
    j["model"] = "gpt-oss-20b";
    j["choices"] = Json::Array{Json(Json::Object{{"index", 0}, {"text", "hi"}})};
    CHECK_EQ(j.dump(), std::string(R"({"choices":[{"index":0,"text":"hi"}],"model":"gpt-oss-20b"})"));
}
