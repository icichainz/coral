#!/usr/bin/env python3
"""
Produce tests/data/harmony_vectors.json: conversations rendered with the
model's own chat_template.jinja (environment mirrors Hugging Face
transformers' apply_chat_template: trim_blocks, lstrip_blocks, loopcontrols,
json.dumps-based tojson, raise_exception, strftime_now) and tokenized with
tiktoken o200k_harmony (special tokens allowed). The JSON is committed; the
C++ tests never run Python.

    uv run --python 3.13 --with tiktoken --with jinja2 python3 tests/reference/gen_harmony_vectors.py

Each case carries the conversation twice: `hf_messages` / `hf_tools` in the
shape the template expects, and `messages` / `tools` / `options` in coral's
harmony::Message / ToolSpec / RenderOptions shape (tool parameter schemas are
kept as JSON strings so property order survives).
"""
import json
import os
from datetime import datetime

import jinja2
import tiktoken
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

DATE = datetime(2025, 1, 15)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEMPLATE = os.path.join(ROOT, "models", "gpt-oss-20b", "chat_template.jinja")
OUT = os.path.join(ROOT, "tests", "data", "harmony_vectors.json")


def make_env():
    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: DATE.strftime(fmt)
    return env


WEATHER_PARAMS = {
    "type": "object",
    "properties": {
        "location": {"type": "string", "description": "City and country, e.g. Paris, France"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
        "days": {"type": "integer", "description": "Forecast length", "default": 3},
        "hourly": {"type": "boolean"},
    },
    "required": ["location"],
}

SEARCH_PARAMS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search terms"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "scores": {"type": "array", "items": {"type": "number"}, "nullable": True},
        "points": {"type": "array", "items": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}, "required": ["x"]}},
        "matrix": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
        "anything": {"type": "array"},
        "filter": {
            "type": "object",
            "description": "Optional filter",
            "properties": {"lang": {"type": "string", "nullable": True}, "safe": {"type": "boolean", "default": True}},
            "required": ["safe"],
        },
        "opts": {"type": "object"},
        "mode": {"type": ["string", "null"]},
        "single": {"type": ["integer"]},
        "choice": {"oneOf": [{"type": "string", "description": "a name"}, {"type": "integer"}]},
        "objchoice": {"oneOf": [{"type": "object"}, {"type": "string"}]},
        "limit": {"type": "number", "default": 2.5},
        "extra": {"description": "untyped", "default": None},
        "label": {"type": "string", "default": "héllo \"q\""},
    },
}

NOARG = {"type": "object", "properties": {}}

EDGE_PARAMS = {
    "type": "object",
    "properties": {
        "pick": {"oneOf": [{"type": "string", "description": "by name", "default": "a"}, {"type": "number", "default": 1}],
                 "default": "b", "description": "Either form"},
        "level": {"type": "integer", "enum": [1, 2, 3]},
        "color": {"type": "string", "enum": ["red", "green"], "default": "green", "description": "Hue"},
        "objs": {"type": "array", "items": {"type": "object"}},
        "unions": {"type": "array", "items": {"type": ["string", "number"]}},
        "short_obj": {"type": "array", "items": {"type": "object", "properties": {"k": {"type": "string"}}}},
        "long_unicode": {"type": "array", "items": {"type": "string", "enum": ["ééééééééééééééé", "ààààààààààààààà", "üü"]}},
        "nested_nullable": {"type": "array", "items": {"type": "array"}, "nullable": True},
        "ratio": {"type": "number", "default": 0.0001},
        "big": {"type": "number", "default": 1e16},
        "tiny": {"type": "number", "default": 1.5e-7},
        "neg": {"type": "integer", "default": -12},
        "obj_default": {"type": "object", "default": {"a": [1, True, None], "b": "x\ny"}},
        "empty_desc": {"type": "string", "description": ""},
    },
    "required": ["pick", "level"],
}


def fn(name, desc, params):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": params}}


def spec(name, desc, params):
    return {"name": name, "description": desc, "parameters_json_schema": json.dumps(params, separators=(",", ":"))}


def msg(role, content, **kw):
    m = {"role": role, "content": content}
    m.update(kw)
    return m


CASES = [
    {
        "name": "user_only_default",
        "kwargs": {},
        "hf_messages": [msg("user", "Hello!")],
        "messages": [msg("user", "Hello!")],
    },
    {
        "name": "developer_low_identity",
        "kwargs": {"reasoning_effort": "low", "model_identity": "You are Coral, a fast local assistant."},
        "hf_messages": [msg("developer", "Answer in French.\nBe concise."), msg("user", "What is the capital of Japan?")],
        "messages": [msg("developer", "Answer in French.\nBe concise."), msg("user", "What is the capital of Japan?")],
    },
    {
        "name": "system_as_developer_high",
        "kwargs": {"reasoning_effort": "high"},
        "hf_messages": [msg("system", "You are a pirate."), msg("user", "Tell me a joke  \n\n  please")],
        "messages": [msg("system", "You are a pirate."), msg("user", "Tell me a joke  \n\n  please")],
    },
    {
        "name": "multi_turn_drops_analysis",
        "kwargs": {"reasoning_effort": "medium"},
        "hf_messages": [
            msg("user", "2+2?"),
            {"role": "assistant", "thinking": "Simple arithmetic.", "content": "4"},
            msg("user", "Et en français ? «quatre» 😀"),
        ],
        "messages": [
            msg("user", "2+2?"),
            msg("assistant", "Simple arithmetic.", channel="analysis"),
            msg("assistant", "4", channel="final"),
            msg("user", "Et en français ? «quatre» 😀"),
        ],
    },
    {
        "name": "tools_pending_call_keeps_analysis",
        "kwargs": {},
        "hf_tools": [fn("get_weather", "Get the current weather", WEATHER_PARAMS), fn("get_time", "Current UTC time", NOARG)],
        "tools": [spec("get_weather", "Get the current weather", WEATHER_PARAMS), spec("get_time", "Current UTC time", NOARG)],
        "hf_messages": [
            msg("developer", "Use tools when helpful."),
            msg("user", "Weather in Paris?"),
            {"role": "assistant", "thinking": "Need to call get_weather.",
             "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"location": "Paris, France", "unit": "celsius", "days": 2}}}]},
            msg("tool", "{\"temp\": 21, \"sky\": \"clear\"}"),
        ],
        "messages": [
            msg("developer", "Use tools when helpful."),
            msg("user", "Weather in Paris?"),
            msg("assistant", "Need to call get_weather.", channel="analysis"),
            msg("assistant", "{\"location\":\"Paris, France\",\"unit\":\"celsius\",\"days\":2}", channel="commentary", recipient="functions.get_weather"),
            msg("tool", "{\"temp\": 21, \"sky\": \"clear\"}", name="get_weather"),
        ],
    },
    {
        "name": "tools_completed_call_drops_analysis",
        "kwargs": {"reasoning_effort": "low"},
        "hf_tools": [fn("get_weather", "Get the current weather", WEATHER_PARAMS)],
        "tools": [spec("get_weather", "Get the current weather", WEATHER_PARAMS)],
        "hf_messages": [
            msg("user", "Weather in Oslo?"),
            {"role": "assistant", "thinking": "Call the tool.",
             "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"location": "Oslo"}}}]},
            msg("tool", "-3C, snow"),
            {"role": "assistant", "content": "It is -3°C and snowing in Oslo."},
            msg("user", "Thanks! And tomorrow?"),
        ],
        "messages": [
            msg("user", "Weather in Oslo?"),
            msg("assistant", "Call the tool.", channel="analysis"),
            msg("assistant", "{\"location\": \"Oslo\"}", recipient="functions.get_weather"),
            msg("tool", "-3C, snow"),
            msg("assistant", "It is -3°C and snowing in Oslo.", channel="final"),
            msg("user", "Thanks! And tomorrow?"),
        ],
    },
    {
        "name": "complex_schema",
        "kwargs": {"reasoning_effort": "high"},
        "hf_tools": [fn("search", "Search the index", SEARCH_PARAMS), fn("noop", "Does nothing", {"type": "object"})],
        "tools": [spec("search", "Search the index", SEARCH_PARAMS), spec("noop", "Does nothing", {"type": "object"})],
        "hf_messages": [msg("user", "Find \"coral\" docs")],
        "messages": [msg("user", "Find \"coral\" docs")],
    },
    {
        "name": "edge_schema_and_multi_calls",
        "kwargs": {"reasoning_effort": "medium", "model_identity": "Tu es Corail. 🪸"},
        "hf_tools": [fn("edge", "Edge cases: «quotes» and ümlauts", EDGE_PARAMS), fn("ping", "Ping", NOARG)],
        "tools": [spec("edge", "Edge cases: «quotes» and ümlauts", EDGE_PARAMS), spec("ping", "Ping", NOARG)],
        "hf_messages": [
            msg("user", "Run edge then ping."),
            {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "edge", "arguments": {"pick": "b", "level": 3, "ratio": 0.25, "list": [1, 2.0, "x"], "nested": {"z": None, "a": False}}}}]},
            msg("tool", "multi\nline\tresult \"quoted\" é"),
            {"role": "assistant", "thinking": "Now ping.", "tool_calls": [{"type": "function", "function": {"name": "ping", "arguments": {}}}]},
            msg("tool", "pong"),
        ],
        "messages": [
            msg("user", "Run edge then ping."),
            msg("assistant", "{ \"pick\" : \"b\", \"level\": 3, \"ratio\": 0.250, \"list\": [1, 2.0, \"x\"], \"nested\": {\"z\": null, \"a\": false} }", recipient="functions.edge"),
            msg("tool", "multi\nline\tresult \"quoted\" é"),
            msg("assistant", "Now ping.", channel="analysis"),
            msg("assistant", "{}", recipient="ping"),
            msg("tool", "pong", name="ping"),
        ],
    },
    {
        "name": "analysis_only_turns_dropped",
        "kwargs": {"reasoning_effort": "low"},
        "hf_messages": [
            msg("developer", ""),
            msg("user", "Hi"),
            {"role": "assistant", "thinking": "Greet back.", "content": "Hello! How can I help?"},
            msg("user", "Bye"),
        ],
        "messages": [
            msg("developer", ""),
            msg("user", "Hi"),
            msg("assistant", "Greet back.", channel="analysis"),
            msg("assistant", "Hello! How can I help?"),
            msg("user", "Bye"),
        ],
    },
    {
        "name": "code_and_whitespace",
        "kwargs": {},
        "hf_messages": [msg("user", "Fix this:\n```python\ndef f(x):\n\treturn x*2  \n```\n\n")],
        "messages": [msg("user", "Fix this:\n```python\ndef f(x):\n\treturn x*2  \n```\n\n")],
    },
]


def main():
    env = make_env()
    with open(TEMPLATE, encoding="utf-8") as f:
        tmpl = env.from_string(f.read())
    enc = tiktoken.get_encoding("o200k_harmony")
    out_cases = []
    for c in CASES:
        kwargs = dict(c["kwargs"])
        text = tmpl.render(messages=c["hf_messages"], tools=c.get("hf_tools"), add_generation_prompt=True,
                           bos_token="<|startoftext|>", eos_token="<|return|>", pad_token="<|endoftext|>", **kwargs)
        ids = enc.encode(text, allowed_special="all")
        options = {"current_date": DATE.strftime("%Y-%m-%d"), "reasoning": kwargs.get("reasoning_effort", "medium")}
        if "model_identity" in kwargs:
            options["model_identity"] = kwargs["model_identity"]
        out_cases.append({
            "name": c["name"],
            "options": options,
            "messages": c["messages"],
            "tools": c.get("tools", []),
            "text": text,
            "ids": ids,
        })
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"template": "models/gpt-oss-20b/chat_template.jinja", "cases": out_cases}, f, ensure_ascii=True, indent=1)
        f.write("\n")
    print(f"wrote {OUT}: {len(out_cases)} cases")


if __name__ == "__main__":
    main()
