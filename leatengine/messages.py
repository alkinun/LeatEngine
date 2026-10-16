"""Anthropic's Messages API, as Claude Code and Anthropic's SDKs speak it: a request as the chat
completion it asks for, and its reply as a message's content blocks.

Text, images, tool use and its results, and thinking, which a client sends back for a model whose
template keeps a turn's reasoning with its calls; client tools alone, Anthropic's server tools, as
web search, left out; thinking and an effort as a reasoning effort, and a format's schema as a
response format. Thinking blocks come with an empty signature: no model here signs its reasoning,
and no client checks it. The sampling controls Anthropic's API no longer takes, as temperature,
are taken as older clients send them.
"""

import json
import re
import uuid
from typing import Any

from leatengine.chat import Reply

# Claude Code's first line of its system prompt: a hash of the request, `cch=...;`, that changes
# with each, which would keep the prefix cache from reaching past it; as llama.cpp's server does,
# the same in every request
_BILLING = re.compile(r"^(x-anthropic-billing-header:[^\n]*?\bcch=)\w+;")


def as_chat(body: Any) -> dict[str, Any]:
    """The chat completion request a Messages request asks for: its system prompt and messages as
    chat messages, its tools and tool choice as chat's, and the options both share; or a
    ValueError saying what it asks for that is not supported."""
    if not isinstance(body, dict) or body.get("messages") is None:
        raise ValueError("a message needs messages")
    messages: list[dict[str, Any]] = []
    if (system := body.get("system")) is not None:
        text = system if isinstance(system, str) else _text(system, "system")
        if text:
            messages.append({"role": "system", "content": _BILLING.sub(r"\g<1>0;", text)})
    given = body["messages"]
    if not isinstance(given, list) or not all(isinstance(m, dict) for m in given):
        raise ValueError("messages must be a list of messages")
    for message in given:
        role, content = message.get("role"), message.get("content")
        if role not in ("user", "assistant", "system"):
            raise ValueError("a message's role is user, assistant or system")
        if role == "system":  # as Claude Code says its environment, after the user's first words
            content = content if isinstance(content, str) else _text(content, "a system message")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list) or not all(isinstance(b, dict) for b in content):
            raise ValueError("a message's content must be text or a list of blocks")
        messages += _assistant(content) if role == "assistant" else _user(content)
    chat: dict[str, Any] = {"messages": messages}
    tools = body.get("tools") or []
    if not isinstance(tools, list) or not all(isinstance(t, dict) for t in tools):
        raise ValueError("tools must be a list of tools")
    # client tools alone: Anthropic's server tools, of a type as web_search_20250305, which no
    # server but Anthropic's runs, are left out, the model not offered them
    if functions := [t for t in tools if t.get("type", "custom") == "custom"]:
        chat["tools"] = [{"type": "function", "function": {
            "name": t.get("name"), "description": t.get("description", ""),
            "parameters": t.get("input_schema"),
        }} for t in functions]  # fmt: skip
    choice = body.get("tool_choice")
    if choice is not None:
        kind = choice.get("type") if isinstance(choice, dict) else None
        if kind not in ("auto", "any", "tool", "none"):
            raise ValueError('tool_choice must be of type "auto", "any", "tool" or "none"')
        named = {"type": "function", "function": {"name": choice.get("name")}}
        chat["tool_choice"] = named if kind == "tool" else "required" if kind == "any" else kind
    config = body.get("output_config") or {}
    if not isinstance(config, dict):
        raise ValueError("output_config must be an object")
    thinking, effort = body.get("thinking"), config.get("effort")
    kind = thinking.get("type") if isinstance(thinking, dict) else thinking
    if kind not in (None, "enabled", "disabled", "adaptive", "between_tools"):
        raise ValueError('thinking must be of type "enabled", "disabled", "adaptive" or '
                         '"between_tools"')  # fmt: skip
    if effort not in (None, "low", "medium", "high", "xhigh", "max"):
        raise ValueError('output_config.effort must be "low", "medium", "high", "xhigh" or "max"')
    # no reasoning, or as much as asked, or of thinking enabled, the middle; else, adaptive, the
    # model's own, as its template has it
    if kind == "disabled":
        chat["reasoning_effort"] = "none"
    elif effort is not None:
        chat["reasoning_effort"] = "xhigh" if effort == "max" else effort
    elif kind == "enabled":
        chat["reasoning_effort"] = "medium"
    if (form := config.get("format")) is not None:
        if not (isinstance(form, dict) and form.get("type") == "json_schema"
                and isinstance(form.get("schema"), dict)):  # fmt: skip
            raise ValueError('output_config.format must be {"type": "json_schema", "schema": ...}')
        schema = {"name": "output", "schema": form["schema"], "strict": True}
        chat["response_format"] = {"type": "json_schema", "json_schema": schema}
    if body.get("stop_sequences") is not None:
        chat["stop"] = body["stop_sequences"]
    user = (body.get("metadata") or {}).get("user_id")
    if isinstance(user, str) and user:  # whose prompts share cached prefixes with its own alone
        chat["user"] = _end_user(user)
    shared = ("max_tokens", "temperature", "top_p", "top_k", "stream")
    return {k: body[k] for k in shared if body.get(k) is not None} | chat


def _end_user(user_id: str) -> str:
    # the user a request is for: Claude Code's id is JSON of its device, account and session, and
    # the device's and account's alone let a user's sessions share the prefix of its system prompt
    # and tools, some 12k tokens
    try:
        said = json.loads(user_id)
    except ValueError:
        said = None
    if isinstance(said, dict) and isinstance(device := said.get("device_id"), str):
        user_id = f"{device} {said.get('account_uuid') or ''}".strip()
    return user_id[:256]


def _text(blocks: Any, of: str) -> str:
    # the text of a list of text blocks, as a system prompt's
    if not isinstance(blocks, list) or not all(
        isinstance(b, dict) and b.get("type") == "text" for b in blocks
    ):
        raise ValueError(f"{of} must be text or a list of text blocks")
    return "".join(str(b.get("text", "")) for b in blocks)


def _user(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # a user's message: its tool results each a tool message, first, as they answer the calls of
    # the message before; then the rest, its text and images, as the user's
    results, parts = [], []
    for block in blocks:
        kind = block.get("type")
        if kind == "tool_result":
            content = block.get("content")
            if not isinstance(content, str):
                content = _parts(content or [])
                if all(p["type"] == "text" for p in content):
                    content = "".join(p["text"] for p in content)
            results.append({"role": "tool", "tool_call_id": block.get("tool_use_id"),
                            "content": content})  # fmt: skip
        else:
            parts += _parts([block])
    return results + ([{"role": "user", "content": parts}] if parts else [])


def _assistant(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # the model's message: its text, its thinking as reasoning, its tool uses as calls
    text, reasoning, calls = "", "", []
    for block in blocks:
        kind = block.get("type")
        if kind == "text":
            text += str(block.get("text", ""))
        elif kind == "thinking":
            reasoning += str(block.get("thinking", ""))
        elif kind == "redacted_thinking":
            continue  # Anthropic's models' own, encrypted, of no use to another
        elif kind == "tool_use":
            arguments = json.dumps(block.get("input") or {}, ensure_ascii=False)
            function = {"name": block.get("name"), "arguments": arguments}
            calls.append({"id": block.get("id"), "type": "function", "function": function})
        else:
            raise ValueError(f"an assistant's content blocks of type {kind!r} are not supported")
    message: dict[str, Any] = {"role": "assistant", "content": text or (None if calls else "")}
    if calls:
        message["tool_calls"] = calls
    if reasoning:
        message["reasoning_content"] = reasoning
    return [message]


def _parts(blocks: Any) -> list[dict[str, Any]]:
    # content blocks of text and images as chat's parts
    if not isinstance(blocks, list) or not all(isinstance(b, dict) for b in blocks):
        raise ValueError("a tool result's content must be text or a list of blocks")
    parts: list[dict[str, Any]] = []
    for block in blocks:
        kind = block.get("type")
        if kind == "text":
            parts.append({"type": "text", "text": str(block.get("text", ""))})
        elif kind == "image":
            source, url = block.get("source") or {}, None
            if source.get("type") == "base64":
                url = f"data:{source.get('media_type')};base64,{source.get('data')}"
            elif source.get("type") == "url":
                url = source.get("url")  # which leatengine does not fetch, and so refuses
            else:
                raise ValueError("an image's source must be of type base64 or url")
            parts.append({"type": "image_url", "image_url": {"url": url}})
        else:
            raise ValueError(f"content blocks of type {kind!r} are not supported")
    return parts


def message_id() -> str:
    return f"msg_{uuid.uuid4().hex}"


def thinking_block(text: str) -> dict[str, Any]:
    return {"type": "thinking", "thinking": text, "signature": ""}


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def tool_use_block(call: dict[str, Any], whole: bool = True) -> dict[str, Any]:
    """A chat completion's call as a tool_use block: of its arguments as an object, or of none
    as a stream's first holds before its arguments come as JSON."""
    function = call["function"]
    try:
        given = json.loads(function["arguments"]) if whole else {}
    except ValueError:
        given = None
    return {"type": "tool_use", "id": f"toolu_{call['id'].removeprefix('call_')}",
            "name": function["name"],
            "input": given if isinstance(given, dict) else {}}  # fmt: skip


def content(reply: Reply, text: str, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A finished reply's content blocks: its thinking, its text unless it only called tools, and
    its calls, as chat completions' tool_calls have them."""
    blocks = [thinking_block(reply.reasoning)] if reply.reasoning else []
    if text or not calls:
        blocks.append(text_block(text))
    return blocks + [tool_use_block(call) for call in calls]


def stop_reason(finish: str, calls: list[dict[str, Any]], stopped_by: str | None) -> str:
    """Why a reply ended, as a message says: it called tools, met max_tokens or the context's
    end, met a stop sequence, or ended its turn."""
    if calls:
        return "tool_use"
    if finish == "length":
        return "max_tokens"
    return "stop_sequence" if stopped_by is not None else "end_turn"


def error_type(status: int) -> str:
    """An error's type, as Anthropic's API names that of a status."""
    kinds = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
             404: "not_found_error", 413: "request_too_large", 429: "rate_limit_error",
             503: "overloaded_error"}  # fmt: skip
    return kinds.get(status, "invalid_request_error" if status < 500 else "api_error")
