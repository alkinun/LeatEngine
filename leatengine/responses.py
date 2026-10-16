"""OpenAI's Responses API, a stateless subset: a request as the chat completion it asks for, and
its reply as a response's output items.

A response keeps no state: no previous_response_id or conversation, so a client sends the whole
conversation as input each time, its past output items too, as stateless clients do with store
false. Function tools alone; reasoning comes as reasoning items of their text, which a client may
send back for a model, as gpt-oss, whose template keeps a turn's reasoning with its calls.
"""

import uuid
from typing import Any

from leatengine.chat import Reply

# what a request may ask for that no chat completion does, which is refused rather than ignored
_STATEFUL = ("previous_response_id", "conversation", "prompt")


def as_chat(body: Any) -> dict[str, Any]:
    """The chat completion request a Responses request asks for: its instructions and input as
    messages, its function tools as chat tools, its text format as a response format, and the
    options both share; or a ValueError saying what it asks for that is not supported."""
    if not isinstance(body, dict) or body.get("input") is None:
        raise ValueError("a response needs input")
    for key in _STATEFUL:
        if body.get(key) is not None:
            raise ValueError(f"{key} is not supported: this server keeps no responses, so send "
                             "the conversation whole as input")  # fmt: skip
    if body.get("background"):
        raise ValueError("background is not supported: this server keeps no responses")
    instructions = body.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise ValueError("instructions must be a string")
    messages: list[dict[str, Any]] = (
        [{"role": "system", "content": instructions}] if instructions else []
    )
    given = body["input"]
    items = [{"role": "user", "content": given}] if isinstance(given, str) else given
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ValueError("input must be a string or a list of items")
    reasoning = ""  # of the next assistant message
    for item in items:
        kind = item.get("type", "message")
        if kind == "reasoning":
            parts = item.get("content") or item.get("summary") or []
            reasoning += "".join(p.get("text", "") for p in parts if isinstance(p, dict))
            continue
        if kind == "message":
            role = item.get("role")
            if role not in ("user", "system", "developer", "assistant"):
                raise ValueError("an input message's role is user, system, developer or assistant")
            message = {"role": "system" if role == "developer" else role}
            message["content"] = _content(item.get("content"))
        elif kind == "function_call":
            function = {"name": item.get("name"), "arguments": item.get("arguments")}
            call = {"id": item.get("call_id"), "type": "function", "function": function}
            last = messages[-1] if messages else {}
            if last.get("role") == "assistant":  # the calls of the message before
                last.setdefault("tool_calls", []).append(call)
                continue
            message = {"role": "assistant", "content": None, "tool_calls": [call]}
        elif kind == "function_call_output":
            output = item.get("output")
            content = output if isinstance(output, str) else _content(output)
            message = {"role": "tool", "tool_call_id": item.get("call_id"), "content": content}
        else:
            raise ValueError(f"input items of type {kind!r} are not supported")
        if message["role"] == "assistant" and reasoning:
            message["reasoning_content"], reasoning = reasoning, ""
        messages.append(message)
    chat: dict[str, Any] = {"messages": messages}
    tools = body.get("tools") or []
    if not isinstance(tools, list) or not all(
        isinstance(t, dict) and t.get("type") == "function" for t in tools
    ):
        raise ValueError("tools must be a list of function tools: other kinds are not supported")
    if tools:
        keys = ("name", "description", "parameters")
        chat["tools"] = [{"type": "function", "function": {k: t[k] for k in keys if k in t}}
                         for t in tools]  # fmt: skip
    if (form := (body.get("text") or {}).get("format")) is not None:
        if not isinstance(form, dict):
            raise ValueError("text.format must be an object")
        if form.get("type") == "json_schema":
            schema = {k: form.get(k) for k in ("name", "schema", "strict")}
            chat["response_format"] = {"type": "json_schema", "json_schema": schema}
        else:
            chat["response_format"] = {"type": form.get("type")}
    if (effort := (body.get("reasoning") or {}).get("effort")) is not None:
        chat["reasoning_effort"] = effort
    if body.get("max_output_tokens") is not None:
        chat["max_completion_tokens"] = body["max_output_tokens"]
    shared = ("tool_choice", "temperature", "top_p", "top_k", "min_p", "presence_penalty",
              "frequency_penalty", "repetition_penalty", "seed", "stop", "stream", "user",
              "safety_identifier", "service_tier", "chat_template_kwargs")  # fmt: skip
    return chat | {k: body[k] for k in shared if body.get(k) is not None}


def _content(content: Any) -> str | list[dict[str, Any]]:
    # an input message's content as a chat message's: its text, input or output, and its images
    if isinstance(content, str):
        return content
    if not isinstance(content, list) or not all(isinstance(p, dict) for p in content):
        raise ValueError("a message's content must be text or a list of parts")
    parts: list[dict[str, Any]] = []
    for part in content:
        kind = part.get("type")
        if kind in ("input_text", "output_text", "text"):
            parts.append({"type": "text", "text": part.get("text")})
        elif kind == "input_image":
            parts.append({"type": "image_url", "image_url": {"url": part.get("image_url")}})
        else:
            raise ValueError(f"content parts of type {kind!r} are not supported")
    return parts


def echoed(body: dict[str, Any]) -> dict[str, Any]:
    """What a response says of the request it answers, as OpenAI's does."""
    return {
        "instructions": body.get("instructions"),
        "max_output_tokens": body.get("max_output_tokens"),
        "metadata": body.get("metadata") or {},
        "parallel_tool_calls": body.get("parallel_tool_calls", True),
        "previous_response_id": None,
        "reasoning": {"effort": (body.get("reasoning") or {}).get("effort"), "summary": None},
        "store": False,  # whatever it asked: this server keeps none
        "text": {"format": (body.get("text") or {}).get("format") or {"type": "text"}},
        "tool_choice": body.get("tool_choice") or "auto",
        "tools": body.get("tools") or [],
        "truncation": body.get("truncation") or "disabled",
    }


def item_id(kind: str) -> str:
    """A new output item's id, of its kind's prefix: rs, msg or fc."""
    return f"{kind}_{uuid.uuid4().hex}"


def reasoning_item(id: str, text: str) -> dict[str, Any]:
    return {"type": "reasoning", "id": id, "summary": [],
            "content": [{"type": "reasoning_text", "text": text}]}  # fmt: skip


def message_item(id: str, text: str | None, status: str = "completed") -> dict[str, Any]:
    """A message of the reply's text, of no part before it has any."""
    parts = [] if text is None else [output_text(text)]
    return {"type": "message", "id": id, "status": status, "role": "assistant", "content": parts}


def output_text(text: str) -> dict[str, Any]:
    return {"type": "output_text", "text": text, "annotations": [], "logprobs": []}


def function_item(call: dict[str, Any], arguments: str | None = None) -> dict[str, Any]:
    """A call as a function_call item, of a chat completion's call: its arguments, unless given
    others, as the stream's first item has none yet."""
    function = call["function"]
    return {"type": "function_call", "id": f"fc_{call['id'].removeprefix('call_')}",
            "call_id": call["id"], "name": function["name"],
            "arguments": function["arguments"] if arguments is None else arguments,
            "status": "completed"}  # fmt: skip


def output(reply: Reply, content: str, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A finished reply's output items: its reasoning, its text unless it only called tools,
    and its calls, as chat completions' tool_calls have them."""
    items = [reasoning_item(item_id("rs"), reply.reasoning)] if reply.reasoning else []
    if content or not calls:
        items.append(message_item(item_id("msg"), content))
    return items + [function_item(call) for call in calls]
