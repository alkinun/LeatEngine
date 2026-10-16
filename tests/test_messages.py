import pytest

from leatengine.chat import Reply
from leatengine.messages import as_chat, content, stop_reason, tool_use_block

CITY = {"type": "object", "properties": {"city": {"type": "string"}}}
WEATHER = {"name": "weather", "description": "The weather in a city now", "input_schema": CITY}


def test_text():
    # the system prompt as a system message, then the messages as they are
    chat = as_chat({"system": "Be brief.", "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 9, "model": "claude"})  # fmt: skip
    assert chat == {"messages": [{"role": "system", "content": "Be brief."},
                                 {"role": "user", "content": "Hi"}],
                    "max_tokens": 9}  # fmt: skip


def test_blocks():
    # a conversation sent back whole: a system prompt of blocks, text and images, the model's
    # thinking and the tools it used, their results first in the next message, before its text
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA"}}

    def use(i: int, city: str) -> dict:
        return {"type": "tool_use", "id": f"toolu_{i}", "name": "weather", "input": {"city": city}}

    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Weather?"}, image]},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "Check both.", "signature": ""},
            {"type": "text", "text": "Checking."}, use(1, "Oslo"), use(2, "Rome")]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Snow"},
            {"type": "tool_result", "tool_use_id": "toolu_2",
             "content": [{"type": "text", "text": "Sun"}], "is_error": False},
            {"type": "text", "text": "And now?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Mixed."}]},
    ]  # fmt: skip
    system = [{"type": "text", "text": "Use tools.", "cache_control": {"type": "ephemeral"}}]
    calls = [{"id": f"toolu_{i}", "type": "function",
              "function": {"name": "weather", "arguments": f'{{"city": "{city}"}}'}}
             for i, city in ((1, "Oslo"), (2, "Rome"))]  # fmt: skip
    assert as_chat({"system": system, "messages": messages})["messages"] == [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": [
            {"type": "text", "text": "Weather?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]},
        {"role": "assistant", "content": "Checking.", "tool_calls": calls,
         "reasoning_content": "Check both."},
        {"role": "tool", "tool_call_id": "toolu_1", "content": "Snow"},
        {"role": "tool", "tool_call_id": "toolu_2", "content": "Sun"},
        {"role": "user", "content": [{"type": "text", "text": "And now?"}]},
        {"role": "assistant", "content": "Mixed."},
    ]  # fmt: skip


def test_calls_alone():
    # a message of tool uses alone has no content, as chat's calls, and a tool result of an image
    # keeps it as a part
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA"}}
    used = {"type": "tool_use", "id": "t", "name": "f", "input": {}}
    result = {"type": "tool_result", "tool_use_id": "t", "content": [image]}
    messages = [{"role": "assistant", "content": [used]}, {"role": "user", "content": [result]}]
    first, second = as_chat({"messages": messages})["messages"]
    assert first["content"] is None and first["tool_calls"][0]["function"]["arguments"] == "{}"
    assert second == {"role": "tool", "tool_call_id": "t", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}  # fmt: skip


def test_system_messages():
    # where they are, as Claude Code says its environment after the user's first words
    said = [{"type": "text", "text": "# Environment"}]
    messages = [{"role": "user", "content": "Fix it."}, {"role": "system", "content": said}]
    assert as_chat({"messages": messages})["messages"] == [
        {"role": "user", "content": "Fix it."},
        {"role": "system", "content": "# Environment"},
    ]


def test_claude_codes_user():
    # of its device and account, not its session, so that its sessions share their prefix
    said = '{"device_id": "d1", "account_uuid": "a1", "session_id": "s1"}'
    asked = {"messages": [{"role": "user", "content": "x"}]}
    assert as_chat(asked | {"metadata": {"user_id": said}})["user"] == "d1 a1"
    assert as_chat(asked | {"metadata": {"user_id": "[1]"}})["user"] == "[1]"
    assert as_chat(asked | {"metadata": {"user_id": "u" * 300}})["user"] == "u" * 256


def test_claude_codes_billing_header():
    # its hash of each request, in the system prompt's first line, the same in every request
    asked = "x-anthropic-billing-header: cc_version=2.1.101.e51; cc_entrypoint=cli; cch=a5145;"
    chat = as_chat({"system": [{"type": "text", "text": asked}, {"type": "text", "text": "\nHi"}],
                    "messages": [{"role": "user", "content": "x"}]})  # fmt: skip
    assert chat["messages"][0]["content"] == asked.replace("a5145", "0") + "\nHi"


def test_options():
    # client tools as chat tools, server tools left out; the tool choices, thinking, stop
    # sequences and the user's id as chat's, and the options both APIs share
    body = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 64,
            "tools": [WEATHER | {"type": "custom"}, {"type": "web_search_20250305", "name": "s"}],
            "tool_choice": {"type": "any"}, "thinking": {"type": "enabled", "budget_tokens": 2048},
            "stop_sequences": ["END"], "temperature": 0.2, "top_k": 5, "stream": True,
            "metadata": {"user_id": "u1"}, "service_tier": "auto"}  # fmt: skip
    function = {"name": "weather", "description": WEATHER["description"], "parameters": CITY}
    assert as_chat(body) == {
        "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 64,
        "tools": [{"type": "function", "function": function}], "tool_choice": "required",
        "reasoning_effort": "medium", "stop": ["END"], "temperature": 0.2, "top_k": 5,
        "stream": True, "user": "u1",
    }  # fmt: skip
    asked = {"messages": [{"role": "user", "content": "Hi"}]}
    named = as_chat(asked | {"tool_choice": {"type": "tool", "name": "weather"}})["tool_choice"]
    assert named == {"type": "function", "function": {"name": "weather"}}
    assert as_chat(asked | {"tool_choice": {"type": "none"}})["tool_choice"] == "none"

    def effort(**kwargs) -> str | None:
        return as_chat(asked | kwargs).get("reasoning_effort")

    # thinking off, or as much as asked, or of thinking enabled, the middle; else the model's
    assert effort(thinking={"type": "disabled"}, output_config={"effort": "high"}) == "none"
    assert effort(thinking={"type": "enabled", "budget_tokens": 1024},
                  output_config={"effort": "max"}) == "xhigh"  # fmt: skip
    assert effort(thinking={"type": "adaptive"}, output_config={"effort": "low"}) == "low"
    assert effort(thinking={"type": "adaptive"}) is effort(thinking={"type": "between_tools"})
    assert effort(thinking={"type": "adaptive"}) is None
    form = {"type": "json_schema", "schema": CITY}
    formatted = as_chat(asked | {"output_config": {"format": form}})["response_format"]
    schema = {"name": "output", "schema": CITY, "strict": True}
    assert formatted == {"type": "json_schema", "json_schema": schema}


@pytest.mark.parametrize("body, error", [
    ({}, "a message needs messages"),
    ({"messages": [{"role": "tool", "content": "x"}]}, "role is user, assistant or system"),
    ({"messages": [{"role": "system", "content": [{"type": "image"}]}]}, "a system message must"),
    ({"messages": [{"role": "user", "content": [{"type": "document"}]}]}, "'document'"),
    ({"messages": [{"role": "assistant", "content": [{"type": "image"}]}]}, "assistant's"),
    ({"messages": [], "system": [{"type": "image"}]}, "system must be text or a list of text"),
    ({"messages": [], "tool_choice": {"type": "all"}}, "tool_choice must be of type"),
    ({"messages": [], "thinking": True}, "thinking must be of type"),
    ({"messages": [], "output_config": {"effort": "huge"}}, "output_config.effort must be"),
    ({"messages": [], "output_config": {"format": {"type": "xml"}}}, "output_config.format must"),
])  # fmt: skip
def test_refused(body, error):
    with pytest.raises(ValueError, match=error):
        as_chat(body)


def test_content():
    # thinking, then text unless the reply only called tools, then each call, its arguments an
    # object
    call = {"id": "call_1", "type": "function",
            "function": {"name": "weather", "arguments": '{"city": "Paris"}'}}  # fmt: skip
    used = {"type": "tool_use", "id": "toolu_1", "name": "weather", "input": {"city": "Paris"}}
    thinking = {"type": "thinking", "thinking": "Rain?", "signature": ""}
    assert content(Reply(reasoning="Rain?"), "", [call]) == [thinking, used]
    assert content(Reply(), "Hi", [call]) == [{"type": "text", "text": "Hi"}, used]
    assert content(Reply(), "", []) == [{"type": "text", "text": ""}]
    assert tool_use_block(call, whole=False)["input"] == {}
    broken = call | {"function": {"name": "weather", "arguments": "{"}}
    assert tool_use_block(broken)["input"] == {}


def test_stop_reason():
    call = [{"id": "call_1"}]
    assert stop_reason("stop", call, None) == "tool_use"
    assert stop_reason("length", [], None) == "max_tokens"
    assert stop_reason("stop", [], "END") == "stop_sequence"
    assert stop_reason("stop", [], None) == "end_turn"
