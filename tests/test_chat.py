import json

import jinja2
import pytest

from leatengine.chat import (
    EFFORTS,
    IMAGE,
    ChatTemplate,
    Reply,
    images,
    parse_tool_calls,
    split_reply,
    tool_call_start,
    typed_arguments,
)
from leatengine.gguf import GGUF
from leatengine.tokenizer import USER_DEFINED, Tokenizer
from tests.helpers import ids, tiny_metadata

TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}{{ m['role'] }}:{{ m['content'] }}{{ eos_token }}"
    "{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}"
)


def chat(template: str = TEMPLATE) -> tuple[ChatTemplate, Tokenizer]:
    metadata = tiny_metadata(**{"tokenizer.chat_template": template})
    tok = Tokenizer(metadata)
    return ChatTemplate(metadata, tok), tok


def test_render():
    c, _ = chat()
    assert c.render([{"role": "user", "content": "hi"}]) == "<s>user:hi<|eot|>assistant:"
    assert (
        c.render([{"role": "user", "content": "hi"}], add_generation_prompt=False)
        == "<s>user:hi<|eot|>"
    )


def test_bos_exactly_once():
    msgs = [{"role": "user", "content": "ab"}]
    for template in (TEMPLATE, TEMPLATE.removeprefix("{{ bos_token }}")):
        c, tok = chat(template)
        encoded = c.encode(msgs)
        assert encoded[0] == tok.bos_id and encoded.count(tok.bos_id) == 1
        assert ids(tok, "<|eot|>")[0] in encoded  # control text from the template is parsed


def test_template_helpers():
    c, _ = chat("{{ raise_exception('no tools') if tools else messages | tojson }}")
    assert c.render([{"content": "<é>"}]) == '[{"content": "<é>"}]'  # no HTML escaping
    with pytest.raises(jinja2.TemplateError, match="no tools"):
        c.render([], tools=[{"name": "f"}])


def test_openai_messages():
    # text parts are joined, and tool calls are left out where there are none and otherwise have
    # their JSON arguments decoded, as templates expect
    c, _ = chat(
        "{% for m in messages %}{{ m.content or '' }}"
        "{% if 'tool_calls' in m %}{{ m.tool_calls[0].function.arguments.city }}{% endif %};"
        "{% endfor %}"
    )
    function = {"name": "f", "arguments": '{"city": "Oslo"}'}
    call = {"id": "1", "type": "function", "function": function}
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "assistant", "content": "c", "tool_calls": None},
    ]
    assert c.render(messages) == "a\nb;Oslo;c;"
    with pytest.raises(ValueError, match="text, image or image_url"):
        c.render([{"role": "user", "content": [{"type": "input_audio"}]}])
    for content in ({"text": "a"}, 5, True):  # of none of OpenAI's kinds
        with pytest.raises(ValueError, match="content must be text, a list of parts, or null"):
            c.render([{"role": "user", "content": content}])
    # a call's content of None, as OpenAI's clients send it: as text to templates that take text,
    # as Qwen2.5's; None to those that tell it from text, as DeepSeek-R1's distillations', unless
    # they read it as text too, as Qwen3's splits it
    texts = chat("{% for m in messages %}{{ m.content + '|' }}{% endfor %}")[0]
    assert texts.render(messages[1:2]) == "|"
    told = chat("{% for m in messages %}{{ m.content is none }}{% endfor %}")[0]
    assert told.render(messages[1:2]) == "True"
    for text in ("m.content.split('x')", "m.content + 'x'"):
        both = "{% for m in messages %}{{ m.content is none }}{{ " + text + " }}{% endfor %}"
        assert chat(both)[0].render(messages[1:2]).startswith("False")


def test_generation_blocks():
    # transformers' {% generation %} blocks, which mark the assistant's text for training, render
    # as their bodies, as LFM2.5's template has them
    c, _ = chat(
        "{% for m in messages %}{% if m.role == 'assistant' %}{% generation %}{{ m.content }}"
        "{% endgeneration %}{% else %}{{ m.content }}{% endif %};{% endfor %}"
    )
    messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    assert c.render(messages) == "a;b;"


def test_undefined_joins_as_no_text():
    # a tool's description, which OpenAI's API leaves out at will, joined to text as gpt-oss's
    # template joins it; any other use of what is not there fails as before
    c, _ = chat("{% for t in tools %}{{ '// ' + t.description + '!' }}{% endfor %}")
    assert c.render([], tools=[{"name": "f"}, {"name": "g", "description": "G"}]) == "// !// G!"
    with pytest.raises(jinja2.UndefinedError):
        chat("{{ tools[0].description + 1 }}")[0].render([], tools=[{"name": "f"}])


def test_images():
    # an image part stands as IMAGE in the text, joined to its neighbours without a newline, and
    # its tokens take its place; data: URLs give its bytes, other URLs none
    c, tok = chat("{% for m in messages %}{{ m.content }};{% endfor %}")
    png = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0="}}
    parts = [{"type": "text", "text": "a"}, png, {"type": "text", "text": f"b{IMAGE}"},
             {"type": "text", "text": "c"}]  # fmt: skip
    text = c.render([{"role": "user", "content": parts}])
    assert text == f"a{IMAGE}b\nc;"
    # transformers' image parts stand for images too, and text of its own holds no IMAGE
    other = [{"type": "image"}, {"type": "text", "text": "d"}]
    assert c.render([{"content": f"e{IMAGE}"}, {"content": other}]) == f"e;{IMAGE}d;"
    # nor do a tool call's arguments, or what else the template is given
    called = {"function": {"name": "f", "arguments": json.dumps({"x": IMAGE})}}
    shown = chat("{{ messages[0].tool_calls[0].function.arguments.x }}{{ tools[0] }}")[0]
    assert shown.render([{"tool_calls": [called]}], tools=[IMAGE]) == ""
    assert c.tokens(text, [[-5, -5]]) == tok.encode("a") + [-5, -5] + tok.encode("b\nc;", bos=False)
    with pytest.raises(ValueError, match="1 images, 0 are given"):
        c.tokens(text)
    assert images([{"role": "user", "content": parts}, {"content": "d"}]) == [b"\x89PNG\r"]
    for url in ("https://example.com/a.png", "data:image/png;base64,not base64!"):
        with pytest.raises(ValueError, match="data: URL|not base64"):
            images([{"content": [{"type": "image_url", "image_url": {"url": url}}]}])
    with pytest.raises(ValueError, match="must be an image_url part"):  # an image of no data
        images([{"content": other}])


def with_marker(marker: str) -> ChatTemplate:
    # chat()'s template, of a vocab that holds a reasoning marker, user-defined as in those models
    metadata = tiny_metadata(**{"tokenizer.chat_template": TEMPLATE})
    metadata["tokenizer.ggml.tokens"] = [*metadata["tokenizer.ggml.tokens"], marker]
    metadata["tokenizer.ggml.token_type"] = [*metadata["tokenizer.ggml.token_type"], USER_DEFINED]
    return ChatTemplate(metadata, Tokenizer(metadata))


def test_form():
    # how replies mark their reasoning, by the vocab's markers or the template's, whether a prompt
    # opened a block of it, and what opens an answer
    think = chat(TEMPLATE + "<think>")[0]
    assert chat()[0].form is None and think.form == "think" and think.opens_thinking("a<think>\n")
    harmony = with_marker("<|channel|>")
    assert harmony.form == "harmony" and harmony.answer == "<|channel|>final<|message|>"
    assert chat()[0].answer == ""  # of a format that marks no answer
    gemma = with_marker("<|channel>")
    assert gemma.form == "gemma4" and gemma.opens_thinking("<|turn>model\n<|channel>thought\n")
    atem = chat(TEMPLATE + "{# <atem:function_calls> #}")[0]
    assert atem.form == "atem" and atem.answer == " to=user<|message|>"
    assert not gemma.opens_thinking("<|turn>model\n<|channel>thought\n<channel|>")


def test_preamble():
    # what a reply holds before its answer, as grammar.Matcher takes it: a block of reasoning it
    # may open, the end of the one its prompt opened, or the marker of its answer, unless the
    # prompt wrote it; of a format that marks neither, nothing
    think = chat(TEMPLATE + "<think>")[0]
    assert think.preamble("a") == ("<think>", "</think>")
    assert think.preamble("a<think>\n") == "</think>" and chat()[0].preamble("a") is None
    gemma = with_marker("<|channel>")
    assert gemma.preamble("<|turn>model\n") == ("<|channel>thought", "<channel|>")
    harmony = with_marker("<|channel|>")
    assert harmony.preamble("a") == harmony.answer and harmony.preamble(harmony.answer) is None


def test_thinking():
    # reasoning as gpt-oss's template reads it too, of a message without text, which it would
    # take for the reasoning and refuse both
    c, _ = chat("{% for m in messages %}{{ m.thinking }}|{% endfor %}")
    call = {"type": "function", "function": {"name": "f", "arguments": "{}"}}
    messages = [
        {"role": "assistant", "content": "", "reasoning_content": "Hm.", "tool_calls": [call]},
        {"role": "assistant", "content": "Hi.", "reasoning_content": "Hm."},
        {"role": "assistant", "content": "", "reasoning_content": "Hm.", "thinking": "Own."},
    ]
    assert c.render(messages, add_generation_prompt=False) == "Hm.||Own.|"


GPT_OSS = (
    '{%- if reasoning_effort is not defined %}{%- set reasoning_effort = "medium" %}{%- endif %}'
)
MISTRAL = (
    "{%- set reasoning_effort = reasoning_effort | default('none') %}"
    "{%- if reasoning_effort not in ['none', 'high'] %}{{ raise_exception('no') }}{% endif %}"
)
LEVELS = (
    "{%- if reasoning_effort == 'xhigh' %}x{% elif 'low' == reasoning_effort %}l{% endif %}"
    '{%- set reasoning_effort = reasoning_effort | default("medium") %}'
)
GEMMA = "{%- set enable_thinking = enable_thinking | default(false) -%}"
QWEN = "{%- if enable_thinking is defined and enable_thinking is false %}<think></think>{% endif %}"
MUSE = (
    "{% set rs = reasoning_strength if reasoning_strength is defined and reasoning_strength "
    "else 'high' %}{{ rs }}"
)
COHERE = '{% if reasoning_effort | lower == "none" %}{% endif %}'
SOLAR = (
    '{% set reasoning_effort = reasoning_effort if reasoning_effort is defined else "high" %}'
    '{% if reasoning_effort in ["low", "minimal"] %}{% endif %}'
)


@pytest.mark.parametrize(
    "template, efforts, default",
    [
        (TEMPLATE, (), None),  # never reasons
        (TEMPLATE + "<think>", (), None),  # always does, and cannot be told
        (GPT_OSS + TEMPLATE, ("low", "medium", "high"), "medium"),  # compares it with nothing
        (MISTRAL + TEMPLATE, ("none", "high"), "none"),  # of the values it compares it with
        (LEVELS + TEMPLATE, ("low", "medium", "xhigh"), "medium"),
        (GEMMA + TEMPLATE, ("none", "high"), "none"),  # off by default, as it says
        (QWEN + TEMPLATE, ("none", "high"), "high"),  # on, unless turned off
        (COHERE + TEMPLATE, ("none", "high"), "high"),  # "none" alone compared: off or on
        (SOLAR + TEMPLATE, ("minimal", "low", "high"), "high"),  # a conditional default
        (MUSE + TEMPLATE, ("low", "medium", "high"), "high"),  # Muse Glimmer's own name for it
    ],
)
def test_efforts(template, efforts, default):
    # the efforts a template's model reasons at, and its default, of what the template reads
    c = chat(template)[0]
    assert (c.efforts, c.default_effort) == (efforts, default)


def test_effort():
    # a request's effort, as the template is told it: the nearest of its levels, the lesser of two
    # as near; on or off; or nothing
    levels, toggle = chat(LEVELS + TEMPLATE)[0], chat(QWEN + TEMPLATE)[0]
    told = {e: levels.effort(e)["reasoning_effort"] for e in EFFORTS}
    assert told == {"none": "low", "minimal": "low", "low": "low", "medium": "medium",
                    "high": "medium", "xhigh": "xhigh"}  # fmt: skip
    turned = [toggle.effort(e)["enable_thinking"] for e in ("none", "minimal", "xhigh")]
    assert turned == [False, True, True]
    assert chat()[0].effort("high") == {}
    shown = chat(GPT_OSS + "Reasoning: {{ reasoning_effort }}")[0]
    assert shown.render([], **shown.effort("xhigh")) == "Reasoning: high"
    assert shown.render([]) == "Reasoning: medium"
    # told by Muse Glimmer's reasoning_strength
    muse = chat(MUSE)[0]
    assert muse.render([], **muse.effort("low")) == "low" and muse.render([]) == "high"


def test_typed_arguments():
    # ATEM's values of text, as the tool's schema types them: text where it takes strings, else
    # JSON where the value is
    city = {"type": "string"}
    schema = {"properties": {"city": city, "days": {"type": "integer"}, "tags": {}}}
    tools = [{"type": "function", "function": {"name": "weather", "parameters": schema}}]
    call = {"name": "weather", "arguments": {"city": "42", "days": "3", "tags": '["a"]'}}
    assert typed_arguments(call, tools) == {"city": "42", "days": 3, "tags": ["a"]}


WEATHER = [{"type": "function", "function": {"name": "weather"}}]
PARIS = {"name": "weather", "arguments": {"city": "Paris"}}


@pytest.mark.parametrize(
    "reply, text",
    [
        (' {"name": "weather", "parameters": {"city": "Paris"}} ', ""),  # Llama 3
        ('\n{"name": "weather", "arguments": {"city": "Paris"}}\n', ""),
        ('Let me see.\n<tool_call>\n{"name": "weather", "arguments": {"city": "Paris"}}\n'
         "</tool_call>", "Let me see."),  # Qwen3
        ("<tool_call>\n<function=weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n"
         "</tool_call>", ""),  # Qwen3.5
        ('<|tool_call>call:weather{city:<|"|>Paris<|"|>}<tool_call|>', ""),  # Gemma 4
        ('<|tool_call>call:weather{ city : <|"|>Paris<|"|> }<tool_call|>', ""),  # with spaces
        ("Checking.<|tool_call_start|>[weather(city='Paris')]<|tool_call_end|>", "Checking."),
        ("<tool_call>weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
         ""),  # GLM-4.7
        ('<|tool_call_start|>weather(city="Paris")<|tool_call_end|>', ""),  # LFM2, not a list
    ],
)  # fmt: skip
def test_tool_calls(reply, text):
    assert parse_tool_calls(reply, WEATHER) == (text, [PARIS])


def test_glm_tool_call_arguments():
    # GLM-4.7's values, typed as the tool's schema has them: text, or JSON of other types
    schema = {"properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}
    tools = [{"type": "function", "function": {"name": "weather", "parameters": schema}}]
    reply = ("<tool_call>weather\n<arg_key>city</arg_key>\n<arg_value>42</arg_value>\n"
             "<arg_key>days</arg_key><arg_value>3</arg_value>\n</tool_call>")  # fmt: skip
    assert parse_tool_calls(reply, tools)[1] == [
        {"name": "weather", "arguments": {"city": "42", "days": 3}}
    ]
    malformed = "<tool_call>weather<arg_key>city</arg_key>Paris</tool_call>"
    assert parse_tool_calls(malformed, tools) == (malformed, [])
    # nor is Qwen's JSON that falls short of a call one of GLM's, named by its text
    qwen = 'hi <tool_call>{"name":"f"}</tool_call>'
    assert parse_tool_calls(qwen, tools) == (qwen, [])


def test_python_tool_calls():
    # LFM2's: several calls, of Python's literals and JSON's, dotted names, and negative numbers;
    # none where anything is not a call of literal keyword arguments, which is never run
    weather = "weather(city='Paris', days=-3, hourly=true, at=None)"
    search = 'tools.search(q="a\'b", tags=["x", 1.5], opts={"k": false})'
    _, calls = parse_tool_calls(
        f"<|tool_call_start|>[{weather}, {search}]<|tool_call_end|>", WEATHER
    )
    days = {"city": "Paris", "days": -3, "hourly": True, "at": None}
    query = {"q": "a'b", "tags": ["x", 1.5], "opts": {"k": False}}
    assert calls == [{"name": "weather", "arguments": days},
                     {"name": "tools.search", "arguments": query}]  # fmt: skip
    for bad in ("[weather('Paris')]", "[weather(city=open('x'))]", "[weather(**kw)]", "[1 +",
                "[f()(x=1)]", "[weather(city=-'a')]", "[f(a={[1]: 2})]", "[f(a=1j)]",
                "[f(a=b'x')]"):  # fmt: skip
        reply = f"<|tool_call_start|>{bad}<|tool_call_end|>"
        assert parse_tool_calls(reply, WEATHER) == (reply, [])
    assert tool_call_start("Checking.<|tool_call_st") == len("Checking.")


def test_tool_call_arguments():
    # Gemma 4's syntax: quoted strings, which may hold its other marks, bare or quoted keys,
    # nested objects and lists, and JSON's other values; several calls in a row
    q = '<|"|>'
    args = f"{{a:{q}x, y: {{z}}{q},b:[1,-2.5,true,null],c:{{{q}d{q}:false}}}}"
    reply = f"<|tool_call>call:weather{args}<tool_call|><|tool_call>call:weather{{}}<tool_call|>"
    arguments = {"a": "x, y: {z}", "b": [1, -2.5, True, None], "c": {"d": False}}
    assert parse_tool_calls(reply, WEATHER) == (
        "",
        [PARIS | {"arguments": arguments}, PARIS | {"arguments": {}}],
    )


def test_tool_call_schema_as_written():
    # a tool's parameters as its client wrote them, a JSON schema or not: values as JSON but where
    # the schema says string
    call = "<tool_call>\n<function=f>\n<parameter=n>\n5\n</parameter>\n</function>\n</tool_call>"
    for parameters in ("x", {"properties": "x"}, {"properties": {"n": "x"}}):
        tools = [{"type": "function", "function": {"name": "f", "parameters": parameters}}]
        assert parse_tool_calls(call, tools) == ("", [{"name": "f", "arguments": {"n": 5}}])


def test_tool_call_parameters():
    # Qwen3.5's syntax: values of several lines, as text where the tool takes a string and else
    # as JSON, or as text where that fails; several calls in a row
    properties = {
        "q": {"type": "string"},
        "n": {"type": "integer"},
        "tags": {"type": "array"},
        "id": {"type": ["string", "null"]},
        "k": {"type": ["integer", "string"]},
    }
    tools = [{"type": "function", "function": {"name": "search", "parameters": {
        "type": "object", "properties": properties}}}]  # fmt: skip
    call = (
        "<tool_call>\n<function=search>\n<parameter=q>\n12\nmore lines\n</parameter>\n"
        '<parameter=n>\n5\n</parameter>\n<parameter=tags>\n["a"]\n</parameter>\n'
        "<parameter=other>\nnot json\n</parameter>\n<parameter=id>\n123\n</parameter>\n"
        "<parameter=k>\n7\n</parameter>\n</function>\n</tool_call>"
    )
    # of several types, one of them string, JSON only of another: as llama.cpp's parser
    arguments = {
        "q": "12\nmore lines",
        "n": 5,
        "tags": ["a"],
        "other": "not json",
        "id": "123",
        "k": 7,
    }
    assert parse_tool_calls("Searching.\n\n" + call + "\n" + call, tools) == (
        "Searching.",
        [{"name": "search", "arguments": arguments}] * 2,
    )


@pytest.mark.parametrize(
    "reply",
    [
        "It is sunny.",
        '{"name": "news", "parameters": {}}',
        '{"name": "weather"}',
        '<tool_call>{"name": "weather", "arguments": {}</tool_call>',
        "<tool_call>\n<function=weather>\nParis\n</function>\n</tool_call>",
        "<|tool_call>call:weather{city:Paris}<tool_call|>",
        # what is no call however malformed, nested too deep to parse even
        '{"name": [], "parameters": {}}',
        "[" * 100000,
        "<|tool_call>call:weather{city:" + "[" * 100000 + "<tool_call|>",
    ],
)
def test_text_is_not_a_tool_call(reply):
    assert parse_tool_calls(reply, WEATHER) == (reply, [])


def test_call_of_a_tool_not_declared():
    # a call between a call's markers is one whatever tool it names, as a model trained in
    # another harness calls its own, for the client to say there is none; JSON alone is not
    replies = [
        '<tool_call>{"name": "list", "arguments": {"path": "."}}</tool_call>',
        "<tool_call>\n<function=list>\n<parameter=path>\n.\n</parameter>\n</function>\n</tool_call>",
        "<|tool_call>call:list{path:<|\"|>.<|\"|>}<tool_call|>",
    ]  # fmt: skip
    listed = {"name": "list", "arguments": {"path": "."}}
    for reply in replies:
        assert parse_tool_calls(reply, WEATHER) == ("", [listed])
    assert parse_tool_calls('{"name": "list", "arguments": {}}', WEATHER)[1] == []


def test_tool_call_start():
    # what follows may yet be a tool call: all of a reply that may be JSON, from a marker on, or
    # the start of a marker at the end, and whitespace before those
    replies = ("", ' {"na', " It", "Hi\n<tool_call>{", "Hi <|tool", "Hi ")
    assert [tool_call_start(t) for t in replies] == [0, 0, 3, 2, 2, 2]


def test_no_template():
    with pytest.raises(ValueError, match="no chat template"):
        ChatTemplate(tiny_metadata(), Tokenizer(tiny_metadata()))


@pytest.mark.model
def test_llama3(model_path):
    metadata = GGUF.open(model_path).metadata
    if "<|start_header_id|>" not in metadata.get("tokenizer.chat_template", ""):
        pytest.skip("checks Llama 3's template")
    c = ChatTemplate(metadata, Tokenizer(metadata))
    messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]
    # Llama 3.1's template dates the system prompt 26 Jul 2024, 3.2's today, unless told
    text = c.render(messages, date_string="26 Jul 2024")
    assert text == (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nBe brief.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


HARMONY = (
    "<|channel|>analysis<|message|>The user wants weather.<|end|><|start|>assistant"
    "<|channel|>final<|message|>It is sunny."
)
HARMONY_CALL = (
    "<|channel|>analysis<|message|>Need a tool.<|end|><|start|>assistant<|channel|>commentary "
    'to=functions.weather <|constrain|>json<|message|>{"city": "Paris"}'
)
# Muse Glimmer's ATEM messages after the prompt's <|start|>assistant: reasoning to itself, then an
# answer to the user, or a call; <|eom|> decoding as text or not
ATEM = (
    " to=self<|message|>The user wants weather.<|eom|><|start|>assistant to=user<|message|>Sunny."
)
ATEM_CALL = (
    " to=self<|message|>Need a tool.<|start|>assistant to=weather<|message|><atem:function_calls>\n"
    '<atem:invoke name="weather">\n<atem:parameter name="city">Paris</atem:parameter>\n'
    '<atem:parameter name="days">3</atem:parameter>\n</atem:invoke>\n</atem:function_calls>'
)


@pytest.mark.parametrize(
    "text, form, thinking, reply",
    [
        ("<think>\nhmm</think>\n\nHi.", "think", False, Reply("hmm", "Hi.")),
        ("hmm</think>Hi.", "think", True, Reply("hmm", "Hi.")),  # the prompt opened it
        ("Hi.", "think", False, Reply(content="Hi.")),
        ("<think>still", "think", False, Reply("still")),
        ("a <think>b", None, False, Reply(content="a <think>b")),
        # Gemma 4's thought channel
        ("<|channel>thought\nhmm<channel|>Hi.", "gemma4", False, Reply("hmm", "Hi.")),
        ("Hi.", "gemma4", False, Reply(content="Hi.")),
        (HARMONY, "harmony", False, Reply("The user wants weather.", "It is sunny.")),
        (HARMONY_CALL, "harmony", False,
         Reply("Need a tool.", "", [{"name": "weather", "arguments": '{"city": "Paris"}'}])),
        (ATEM, "atem", False, Reply("The user wants weather.", "Sunny.")),
        (ATEM_CALL, "atem", False, Reply("Need a tool.", "", [
            {"name": "weather", "arguments": {"city": "Paris", "days": "3"}, "typed": False}])),
    ],
)  # fmt: skip
def test_split_reply(text, form, thinking, reply):
    assert split_reply(text, form, thinking) == reply


def test_split_reply_done():
    # an end that may begin a marker waits for more, but not once the reply is done
    assert split_reply("<think>a <", "think") == Reply("a ")
    assert split_reply("<think>a <", "think", done=True) == Reply("a <")
    assert split_reply("x </", None) == Reply(content="x ")
    assert split_reply("x </", None, done=True) == Reply(content="x </")


@pytest.mark.parametrize(
    "text, form",
    [
        (HARMONY, "harmony"),
        (ATEM, "atem"),
        ("<think>a b</think> c d", "think"),
        ("\n<think>a</think> b", "think"),
        ("<|channel>thought\na b<channel|> c d", "gemma4"),
    ],
)
def test_split_reply_streams(text, form):
    # every prefix splits into prefixes of the whole reply's parts: what a stream sent stays
    whole = split_reply(text, form)
    for n in range(len(text)):
        part = split_reply(text[:n], form)
        assert whole.reasoning.startswith(part.reasoning) and whole.content.startswith(part.content)
