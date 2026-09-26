// Harmony chat format for gpt-oss: renders conversations to token ids and
// parses generated tokens back into channels (analysis / commentary / final)
// and tool calls.
//
// Special tokens (o200k_harmony):
//   <|start|>=200006  <|end|>=200007  <|message|>=200008  <|channel|>=200005
//   <|constrain|>=200003  <|return|>=200002  <|call|>=200012
// Generation stops on <|return|> (end of turn) or <|call|> (tool call).
//
// render() reproduces models/gpt-oss-20b/chat_template.jinja (as run by
// Hugging Face apply_chat_template with add_generation_prompt=True) token for
// token; tests/data/harmony_vectors.json holds the reference renders.
//
// Message mapping onto the template:
//   * A leading System or Developer message becomes the "# Instructions"
//     section of the developer message (later ones are ignored, as in the
//     template).
//   * Assistant with `recipient` = tool call; `content` is the raw JSON
//     arguments, re-serialized like Python json.dumps (", " / ": ",
//     key order kept). An immediately preceding assistant "analysis" message
//     is its reasoning, kept only while no later final answer exists.
//   * Assistant "analysis" messages are otherwise dropped when
//     include_final_only_in_history is set. Any other assistant message is
//     rendered on its channel (default "final") and ends with <|end|>.
//   * Tool messages render as `functions.NAME to=assistant` with the content
//     JSON-quoted (template's `|tojson`). NAME comes from `name` or the most
//     recent tool call.
#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "coral/tokenizer.h"

namespace coral::harmony {

enum class Role { System, Developer, User, Assistant, Tool };

struct Message {
    Role role = Role::User;
    std::string content;
    std::optional<std::string> channel;     // assistant only: analysis|commentary|final
    std::optional<std::string> recipient;   // tool calls: e.g. "functions.get_weather"
    std::optional<std::string> name;        // tool messages: tool name
    std::optional<std::string> content_type;// e.g. "json" for constrained tool output
};

enum class ReasoningEffort { Low, Medium, High };

struct ToolSpec {
    std::string name;
    std::string description;
    std::string parameters_json_schema;
};

struct RenderOptions {
    ReasoningEffort reasoning = ReasoningEffort::Medium;
    std::string model_identity = "You are ChatGPT, a large language model trained by OpenAI.";
    std::string knowledge_cutoff = "2024-06";
    std::string current_date;               // YYYY-MM-DD; empty = today
    std::vector<ToolSpec> tools;            // rendered in the developer message
    bool include_final_only_in_history = true;  // drop prior analysis turns
    // Tool-call header content type: false renders "<|channel|>commentary json"
    // exactly like chat_template.jinja; true renders
    // "<|channel|>commentary <|constrain|>json" like the openai-harmony library.
    bool constrain_token = false;
};

// Render a conversation and open the assistant turn (`<|start|>assistant`).
// Throws std::invalid_argument on malformed conversations (e.g. a tool
// message with no preceding tool call).
std::vector<int32_t> render(const Tokenizer& tok, const std::vector<Message>& messages,
                            const RenderOptions& opts);

// The same conversation as text, special tokens spelled out (for logs/tests).
std::string render_text(const std::vector<Message>& messages, const RenderOptions& opts);

// Incremental parser for the assistant's generated tokens, starting right
// after the rendered `<|start|>assistant`. Per message it emits ChannelStart
// (channel, recipient) once `<|message|>` arrives, Text chunks that are always
// complete UTF-8, and ChannelEnd on <|end|>/<|return|>/<|call|>. Messages
// addressed to a recipient (tool calls) are not streamed as Text; their body
// is delivered whole in the ToolCall event on <|call|>.
struct Event {
    enum class Kind { ChannelStart, Text, ChannelEnd, ToolCall, Stop } kind;
    std::string channel;      // ChannelStart/Text/ChannelEnd
    std::string text;         // Text
    std::string recipient;    // ToolCall (also set on ChannelStart/ChannelEnd of a tool call)
    std::string arguments;    // ToolCall (raw JSON)
    std::string content_type; // ToolCall/ChannelStart: e.g. "json" (from <|constrain|> or header)
};

class Parser {
public:
    explicit Parser(const Tokenizer& tok);
    // Feed one token; returns zero or more events. Returns a Stop event on
    // <|return|> / <|call|> and ignores anything after.
    std::vector<Event> push(int32_t token);
    bool finished() const { return finished_; }
private:
    void parse_header();
    void end_message(std::vector<Event>& ev, bool tool_call);

    const Tokenizer& tok_;
    int32_t start_, end_, message_, channel_tok_, constrain_, return_, call_;
    bool finished_ = false;
    std::string header_;      // header bytes; specials as \x01C (channel) / \x01K (constrain)
    std::string channel_, recipient_, content_type_, args_;
    Utf8Streamer text_;
    enum class State { Header, Body };
    State state_ = State::Header;
};

} // namespace coral::harmony
