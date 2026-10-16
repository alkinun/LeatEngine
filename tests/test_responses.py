import pytest

from leatengine.chat import Reply
from leatengine.responses import as_chat, echoed, output

CITY = {"type": "object", "properties": {"city": {"type": "string"}}}
WEATHER = {"type": "function", "name": "weather", "description": "The weather in a city now",
           "parameters": CITY}  # fmt: skip


def test_input_of_text():
    # instructions as a system message, then the input as the user's
    chat = as_chat({"instructions": "Be brief.", "input": "Hi", "max_output_tokens": 9})
    assert chat == {"messages": [{"role": "system", "content": "Be brief."},
                                 {"role": "user", "content": "Hi"}],
                    "max_completion_tokens": 9}  # fmt: skip


def test_input_of_items():
    # a conversation sent back whole: a developer's message as the system's, parts of text and
    # images, the model's reasoning with the calls it made after it, and the calls' outputs
    image = "data:image/png;base64,AAAA"

    def call(i: int, city: str) -> dict:
        return {"type": "function_call", "call_id": f"call_{i}", "name": "weather",
                "arguments": f'{{"city": "{city}"}}'}  # fmt: skip

    items = [
        {"role": "developer", "content": "Use tools."},
        {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "Weather?"},
            {"type": "input_image", "image_url": image}]},
        {"type": "reasoning", "id": "rs_1", "summary": [],
         "content": [{"type": "reasoning_text", "text": "Check both."}]},
        call(1, "Oslo"), call(2, "Rome"),
        {"type": "function_call_output", "call_id": "call_1", "output": "Snow"},
        {"type": "function_call_output", "call_id": "call_2", "output": "Sun"},
        {"type": "message", "role": "assistant",
         "content": [{"type": "output_text", "text": "Mixed."}]},
    ]  # fmt: skip
    calls = [{"id": f"call_{i}", "type": "function",
              "function": {"name": "weather", "arguments": f'{{"city": "{city}"}}'}}
             for i, city in ((1, "Oslo"), (2, "Rome"))]  # fmt: skip
    assert as_chat({"input": items})["messages"] == [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": [{"type": "text", "text": "Weather?"},
                                     {"type": "image_url", "image_url": {"url": image}}]},
        {"role": "assistant", "content": None, "tool_calls": calls,
         "reasoning_content": "Check both."},
        {"role": "tool", "tool_call_id": "call_1", "content": "Snow"},
        {"role": "tool", "tool_call_id": "call_2", "content": "Sun"},
        {"role": "assistant", "content": [{"type": "text", "text": "Mixed."}]},
    ]  # fmt: skip


def test_options():
    # function tools as chat tools, a text format as a response format, an effort, and the
    # options both APIs share
    schema = {"type": "object"}
    chat = as_chat({
        "input": "Hi", "tools": [WEATHER], "tool_choice": "none", "temperature": 0.2,
        "text": {"format": {"type": "json_schema", "name": "x", "schema": schema, "strict": True}},
        "reasoning": {"effort": "low"}, "stream": True, "store": False, "include": [],
    })  # fmt: skip
    function = {k: WEATHER[k] for k in ("name", "description", "parameters")}
    assert chat == {
        "messages": [{"role": "user", "content": "Hi"}],
        "tools": [{"type": "function", "function": function}], "tool_choice": "none",
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "x", "schema": schema, "strict": True}},
        "reasoning_effort": "low", "temperature": 0.2, "stream": True,
    }  # fmt: skip
    assert as_chat({"input": "Hi", "text": {"format": {"type": "json_object"}}})[
        "response_format"] == {"type": "json_object"}  # fmt: skip


@pytest.mark.parametrize("body, error", [
    ({}, "a response needs input"),
    ({"input": "x", "previous_response_id": "resp_1"}, "previous_response_id is not supported"),
    ({"input": "x", "conversation": "conv_1"}, "conversation is not supported"),
    ({"input": "x", "background": True}, "background is not supported"),
    ({"input": 3}, "input must be a string or a list of items"),
    ({"input": [{"type": "item_reference", "id": "x"}]}, "items of type 'item_reference' are not"),
    ({"input": [{"role": "tool", "content": "x"}]}, "an input message's role is user, system,"),
    ({"input": [{"role": "user", "content": [{"type": "input_file"}]}]}, "'input_file' are not"),
    ({"input": "x", "tools": [{"type": "web_search"}]}, "tools must be a list of function tools"),
    ({"input": "x", "instructions": 1}, "instructions must be a string"),
    ({"input": "x", "text": "json"}, "text must be an object"),
    ({"input": "x", "reasoning": "high"}, "reasoning must be an object"),
])  # fmt: skip
def test_refused(body, error):
    with pytest.raises(ValueError, match=error):
        as_chat(body)


def test_output():
    # reasoning, then text unless the reply only called tools, then each call
    call = {"id": "call_7", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    items = output(Reply(content="Hi", reasoning="Think."), "Hi", [call])
    assert [item["type"] for item in items] == ["reasoning", "message", "function_call"]
    assert items[0]["content"] == [{"type": "reasoning_text", "text": "Think."}]
    assert items[1]["content"][0]["text"] == "Hi" and items[1]["role"] == "assistant"
    made = {"type": "function_call", "id": "", "call_id": "call_7", "name": "f",
            "arguments": "{}", "status": "completed"}  # fmt: skip
    assert items[2]["id"].startswith("fc_") and items[2] | {"id": ""} == made
    assert [i["type"] for i in output(Reply(), "", [call])] == ["function_call"]
    assert [i["type"] for i in output(Reply(), "", [])] == ["message"]


def test_echoed():
    # what a response says of its request: whatever it asked, nothing stored
    said = echoed({"input": "x", "store": True, "tools": [WEATHER], "max_output_tokens": 5})
    assert said["store"] is False and said["tools"] == [WEATHER]
    assert (said["max_output_tokens"], said["text"]) == (5, {"format": {"type": "text"}})
