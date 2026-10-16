import collections
import contextlib
import json
import socket
import statistics
import threading
import time
import urllib.error
import urllib.request
import weakref
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import openai
import pytest

from leatengine import server as server_module
from leatengine.chat import ChatTemplate, Reply
from leatengine.defaults import OPENAI, Overrides
from leatengine.engine import Engine
from leatengine.gguf import GGUF
from leatengine.grammar import Grammar, Matcher
from leatengine.keys import Keys
from leatengine.sampler import Sampling
from leatengine.server import Server, _calls, _Completion, _completion, _Load, _reasons, _Writer
from leatengine.tokenizer import Tokenizer
from tests.helpers import CONTEXT, Oracle, chat_template, reference_embedding

WEATHER = {
    "type": "function",
    "function": {
        "name": "weather",
        "description": "The weather in a city now",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


@contextlib.contextmanager
def serving(*models: Path, **options) -> Iterator[Server]:
    # a server of the models on a free port, none loaded
    with Server(models, port=0, **options) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        yield server
        server.shutdown()


def connect(server: Server) -> openai.OpenAI:
    url = f"http://127.0.0.1:{server.server_port}/v1"
    return openai.OpenAI(base_url=url, api_key="unused", max_retries=0)


def load(client: openai.OpenAI, model: str) -> dict:
    return client.post("/models/load", cast_to=object, body={"model": model})


@pytest.fixture(scope="module")
def server(tiny_model) -> Iterator[Server]:
    with serving(tiny_model[0], max_context=CONTEXT, prefill_chunk=8, slots=2) as server:
        server.load("tiny")
        yield server


@pytest.fixture(scope="module")
def engine(server) -> Engine:
    return server.loaded.engine


@pytest.fixture(scope="module")
def client(server) -> openai.OpenAI:
    return connect(server)


@pytest.fixture(scope="module")
def expected(tiny_model):
    # the greedy reply to a message, from an engine of its own
    engine = Engine(tiny_model[0], max_context=CONTEXT, prefill_chunk=8)

    def reply(content: str, max_tokens: int) -> str:
        prompt = engine.tokenizer.encode(content)  # the tiny model's template is the content
        return engine.tokenizer.decode(list(engine.generate(prompt, max_tokens)))

    return reply


def chat(client: openai.OpenAI, content: str, **kwargs):
    messages = [{"role": "user", "content": content}]
    return client.chat.completions.create(model="tiny", messages=messages, **kwargs)


def complete(client: openai.OpenAI, content: str, **kwargs) -> tuple[str, str]:
    # a reply's text and finish reason, streamed or whole
    response = chat(client, content, **kwargs)
    if not kwargs.get("stream"):
        return response.choices[0].message.content, response.choices[0].finish_reason
    choices = [chunk.choices[0] for chunk in response if chunk.choices]
    return "".join(c.delta.content or "" for c in choices), choices[-1].finish_reason


def test_models(client):
    (model,) = client.models.list()
    assert (model.id, model.status, model.max_context) == ("tiny", "loaded", CONTEXT)


def test_load(tiny_model, tmp_path, capsys):
    # models load on request, each in place of the last, saying its context, and answer whatever
    # model a request names
    (other := tmp_path / "other.gguf").symlink_to(tiny_model[0])
    with serving(tiny_model[0], other, max_context=CONTEXT) as server:
        client = connect(server)
        with pytest.raises(openai.InternalServerError, match="503.*no model is loaded"):
            chat(client, "hello")  # the server's state, as /health says it, for a client to retry
        engines = []
        for name, unloaded in (("tiny", "other"), ("other", "tiny"), ("other", "tiny")):
            assert load(client, name)["status"] == "loaded"
            statuses = {model.id: model.status for model in client.models.list()}
            assert statuses == {name: "loaded", unloaded: "unloaded"}
            assert chat(client, "hello", max_tokens=2).model == name
            engines.append(weakref.ref(server.loaded.engine))
        assert f"other: a context of {CONTEXT} tokens, as given" in capsys.readouterr().err
        # the first is freed before the second loads, as a GPU holds one model at most; loading
        # the second again keeps it
        first, second, again = (engine() for engine in engines)
        assert first is None and second is again is server.loaded.engine
        with pytest.raises(openai.NotFoundError, match="there is no model 'tinier'"):
            load(client, "tinier")


def test_completion_waits_for_a_load(tiny_model, tmp_path, monkeypatch):
    # a completion asked for once a model is to load waits for it, rather than finding none
    # loaded, and the models asked for meanwhile say it is loading, as when leatengine serve starts
    (other := tmp_path / "other.gguf").symlink_to(tiny_model[0])
    warm_up = Engine.warm_up
    monkeypatch.setattr(Engine, "warm_up", lambda self: time.sleep(0.5) or warm_up(self))
    with serving(tiny_model[0], other, max_context=CONTEXT) as server:
        client = connect(server)
        loading = threading.Thread(target=server.load, args=("tiny",))
        loading.start()
        while server.ready.is_set():
            time.sleep(0.001)
        assert chat(client, "hello", max_tokens=2).model == "tiny"
        loading.join()
        loading = threading.Thread(target=load, args=(client, "other"))
        loading.start()
        while server.loading is None:
            time.sleep(0.01)
        statuses = {model.id: model.status for model in client.models.list()}
        assert statuses == {"tiny": "unloaded", "other": "loading"}
        loading.join()


def test_completion_waits_for_every_load(tiny_model, tmp_path, monkeypatch):
    # a completion asked for while two loads wait, the first done, waits for the second too, and
    # is the second's, rather than rendered by the first's template and failing once it is gone
    (other := tmp_path / "other.gguf").symlink_to(tiny_model[0])
    warm_up = Engine.warm_up
    monkeypatch.setattr(Engine, "warm_up", lambda self: time.sleep(0.5) or warm_up(self))
    with serving(tiny_model[0], other, max_context=CONTEXT) as server:
        client = connect(server)
        loads = [threading.Thread(target=server.load, args=(name,)) for name in ("tiny", "other")]
        for loading in loads:
            loading.start()
            while not loading.is_alive() or server.ready.is_set():
                time.sleep(0.001)
        while server.loading != "other":  # the first done
            time.sleep(0.001)
        assert chat(client, "hello", max_tokens=2).model == "other"
        for loading in loads:
            loading.join()


def health(server: Server) -> tuple[int, dict]:
    url = f"http://127.0.0.1:{server.server_port}/health"
    try:
        with urllib.request.urlopen(url) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_health(tiny_model, tmp_path, monkeypatch):
    # unready, 503, till a model is loaded and warmed up, and ready, 200, then; asked with no key
    # of a server that has keys, as a supervisor that started it may hold none
    (keys := Keys(tmp_path / "keys.json")).add("alkin")
    warming, warm, warm_up = threading.Event(), threading.Event(), Engine.warm_up

    def held(self: Engine) -> None:  # till the test lets it go on
        warming.set()
        warm.wait(60)
        warm_up(self)

    monkeypatch.setattr(Engine, "warm_up", held)
    with serving(tiny_model[0], max_context=CONTEXT, keys=keys) as server:
        status, body = health(server)
        assert (status, body["error"]["message"]) == (503, "no model is loaded")
        loading = threading.Thread(target=server.load, args=("tiny",))
        loading.start()
        try:
            assert warming.wait(60)
            status, body = health(server)
            assert (status, body["error"]["message"]) == (503, "a model is loading")
        finally:
            warm.set()
            loading.join()
        assert health(server) == (200, {"status": "ok"})


def test_api_alone(server):
    # only API routes are served
    with pytest.raises(urllib.error.HTTPError, match="404"):
        urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/")


def test_hang_up_is_quiet(server, capsys):
    # a client that hung up before its answer was written, while waiting for model discovery
    # does while the engine compiles, is no error to print; any other error is
    for error in (BrokenPipeError(), ConnectionResetError(), ValueError("a bug")):
        try:
            raise error
        except Exception:
            server.handle_error(None, ("127.0.0.1", 1))
    printed = capsys.readouterr().err
    assert "ValueError: a bug" in printed and "Broken" not in printed and "Reset" not in printed


def check_timings(timings: dict, usage) -> None:
    # llama.cpp's timings of a reply, as its usage counts its tokens
    prompt = (timings["cache_n"], timings["prompt_n"])
    assert prompt == (usage.prompt_tokens_details.cached_tokens, usage.prompt_tokens - prompt[0])
    assert timings["predicted_n"] == usage.completion_tokens and timings["tokenize_ms"] > 0
    for name in ("prompt", "predicted"):
        n, ms = timings[f"{name}_n"], timings[f"{name}_ms"]
        if name == "predicted" and n == 1:  # one token takes no step, of no rate
            assert timings["predicted_per_token_ms"] == timings["predicted_per_second"] == 0
            continue
        assert ms > 0 and timings[f"{name}_per_token_ms"] == pytest.approx(ms / n)
        assert timings[f"{name}_per_second"] == pytest.approx(1e3 * n / ms)


def test_reply(client, expected):
    response = chat(client, "hello", max_tokens=8, temperature=0)
    assert response.choices[0].message.content == expected("hello", 8)
    assert response.choices[0].finish_reason == "length"
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (5, 8)
    check_timings(response.model_extra["timings"], response.usage)
    response = chat(client, "hello", max_tokens=1)
    check_timings(response.model_extra["timings"], response.usage)


def test_stream(client, expected):
    usage = {"include_usage": True}
    stream = chat(client, "hello", max_tokens=8, temperature=0, stream=True, stream_options=usage)
    chunks = list(stream)
    assert chunks[0].choices[0].delta.role == "assistant"
    assert "".join(c.choices[0].delta.content or "" for c in chunks[:-1]) == expected("hello", 8)
    assert chunks[-2].choices[0].finish_reason == "length"
    assert chunks[-1].choices == [] and chunks[-1].usage.completion_tokens == 8
    # the timings on the last chunk, as llama.cpp's: the usage's, or else the finish's
    assert all("timings" not in c.model_extra for c in chunks[:-1])
    check_timings(chunks[-1].model_extra["timings"], chunks[-1].usage)
    *_, last = chat(client, "hello", max_tokens=8, temperature=0, stream=True)
    assert last.choices[0].finish_reason == "length"
    assert last.model_extra["timings"]["predicted_n"] == 8


def test_timings_leave_out_the_wait(client, engine, monkeypatch):
    # a request that waits for a slot, both taken by longer replies, is timed from when it gets
    # one: the wait is in the time its client takes to see the first token, not in its timings.
    # Each step takes 10 ms more, so that the wait is long beside a GPU's few milliseconds.
    step = engine.step
    monkeypatch.setattr(engine, "step", lambda: time.sleep(0.01) or step())
    contents = ("a long reply", "another long reply")
    streams = [chat(client, content, max_tokens=30, stream=True) for content in contents]
    for stream in streams:  # both are generating
        next(c for c in stream if c.choices and c.choices[0].delta.content)
    start = time.perf_counter()
    response = chat(client, "a short one", max_tokens=2)
    elapsed = 1e3 * (time.perf_counter() - start)
    for stream in streams:
        assert [c for c in stream if c.choices][-1].choices[0].finish_reason == "length"
    timings = response.model_extra["timings"]
    assert timings["prompt_ms"] + timings["predicted_ms"] < elapsed / 4


@pytest.mark.parametrize("stream", [False, True])
def test_stop(client, expected, stream):
    full = expected("stop me", 16)
    stops = ["never there", full[6:9]]
    reply = complete(client, "stop me", max_tokens=16, temperature=0, stop=stops, stream=stream)
    assert reply == (full[: full.index(stops[1])], "stop")


def text(client: openai.OpenAI, prompt: str, **kwargs) -> tuple[str, str]:
    # a text completion's text and finish reason, streamed or whole
    response = client.completions.create(model="tiny", prompt=prompt, **kwargs)
    if not kwargs.get("stream"):
        return response.choices[0].text, response.choices[0].finish_reason
    choices = [chunk.choices[0] for chunk in response if chunk.choices]
    return "".join(c.text for c in choices), choices[-1].finish_reason


@pytest.mark.parametrize("stream", [False, True])
def test_text_completion(client, expected, stream):
    # the text that goes on from a prompt, as the engine generates it: 16 tokens unless asked for
    # more, as OpenAI's API has it; or cut where a stop string begins
    assert text(client, "hello", temperature=0, stream=stream) == (expected("hello", 16), "length")
    full = expected("stop me", 16)
    reply = text(client, "stop me", temperature=0, stop=full[6:9], stream=stream)
    assert reply == (full[: full.index(full[6:9])], "stop")


def test_text_completion_usage(client):
    # as OpenAI's completions say it, with llama.cpp's timings, of a reply whole or streamed
    response = client.completions.create(model="tiny", prompt="hello", max_tokens=8)
    assert response.object == "text_completion" and response.id.startswith("cmpl-")
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (5, 8)
    check_timings(response.model_extra["timings"], response.usage)
    stream = client.completions.create(
        model="tiny", prompt="hello", max_tokens=8, stream=True,
        stream_options={"include_usage": True},
    )  # fmt: skip
    *chunks, last = list(stream)
    assert chunks[-1].choices[0].finish_reason == "length" and last.choices == []
    assert all(c.object == "text_completion" for c in chunks) and last.usage.completion_tokens == 8
    check_timings(last.model_extra["timings"], last.usage)


def test_text_completion_prompts(server, engine, monkeypatch):
    # a prompt of text, whose control tokens' text is theirs, as llama.cpp's server reads it; of
    # tokens; and with a suffix, between the model's fill-in-the-middle tokens, neither's text a
    # token's, as llama.cpp's /infill puts them
    tok = engine.tokenizer

    def asked(**body) -> list[int]:
        return server_module._text_completion(body, server, server.loaded).prompt

    eot = tok.encode("<|eot|>", bos=False, special=True)
    assert asked(prompt="hi<|eot|>") == tok.encode("hi") + eot
    assert asked(prompt=["hi"]) == tok.encode("hi") and asked(prompt=[3, 1, 2]) == [3, 1, 2]
    with pytest.raises(ValueError, match="tiny has no fill-in-the-middle tokens: it takes no"):
        asked(prompt="a", suffix="b")
    monkeypatch.setattr(tok, "fim", (7, 8, 9))
    a, b = tok.encode("a<|eot|>", bos=False), tok.encode("b", bos=False)
    assert asked(prompt="a<|eot|>", suffix="b") == [7, *a, 8, *b, 9]
    monkeypatch.setattr(tok, "bos_id", 6)  # first, of a tokenizer that adds one
    assert asked(prompt=[3], suffix="") == [6, 7, 3, 8, 9]


def test_logit_bias(client, engine):
    # a token biased up is all a greedy reply takes, of a chat or of text
    seven = engine.tokenizer.decode([7] * 3)
    assert complete(client, "hello", max_tokens=3, temperature=0, logit_bias={"7": 100}) == (
        seven, "length")  # fmt: skip
    reply = text(client, "hello", max_tokens=3, temperature=0, logit_bias={"7": 100})
    assert reply == (seven, "length")


def check_scores(tokens: list[str], chosen: list[float], likeliest: list[list]) -> None:
    # the log-probabilities of 6 greedy tokens: each token the likeliest, the likeliest in order
    assert len(tokens) == len(chosen) == len(likeliest) == 6
    for token, logprob, top in zip(tokens, chosen, likeliest, strict=True):
        assert len(top) == 3 and top[0] == (token, logprob) and logprob <= 0
        assert [v for _, v in top] == sorted((v for _, v in top), reverse=True)


@pytest.mark.parametrize("stream", [False, True])
def test_logprobs(client, stream):
    # each token's log-probability and the 3 likeliest tokens', of a chat whole or streamed
    response = chat(client, "hello", max_tokens=6, temperature=0, logprobs=True, top_logprobs=3,
                    stream=stream)  # fmt: skip
    if stream:
        choices = [chunk.choices[0] for chunk in response if chunk.choices]
        scores = [e for c in choices if c.logprobs for e in c.logprobs.content]
        text = "".join(c.delta.content or "" for c in choices)
    else:
        scores, text = response.choices[0].logprobs.content, response.choices[0].message.content
    # the tokens' bytes the reply's, but for the space SentencePiece puts before the first
    spelled = b"".join(bytes(e.bytes) for e in scores).decode(errors="replace")
    assert spelled in (text, " " + text)
    likeliest = [[(t.token, t.logprob) for t in e.top_logprobs] for e in scores]
    check_scores([e.token for e in scores], [e.logprob for e in scores], likeliest)
    # none unless asked for
    assert chat(client, "hello", max_tokens=2).choices[0].logprobs is None


@pytest.mark.parametrize("stream", [False, True])
def test_text_logprobs(client, stream):
    # as OpenAI's text completions give them: by token, each where its text is in the reply's
    response = client.completions.create(
        model="tiny", prompt="hello", max_tokens=6, temperature=0, logprobs=3, stream=stream
    )
    choices = [chunk.choices[0] for chunk in response] if stream else response.choices
    tokens = [t for c in choices if c.logprobs for t in c.logprobs.tokens]
    chosen = [v for c in choices if c.logprobs for v in c.logprobs.token_logprobs]
    tops = [t for c in choices if c.logprobs for t in c.logprobs.top_logprobs]
    offsets = [o for c in choices if c.logprobs for o in c.logprobs.text_offset]
    # where each token's text begins in the reply's: one of partial UTF-8 none of its own
    text = "".join(c.text for c in choices)
    assert offsets[0] == 0 and offsets == sorted(offsets) and offsets[-1] <= len(text)
    likeliest = [sorted(top.items(), key=lambda kv: -kv[1]) for top in tops]
    check_scores(tokens, chosen, likeliest)


def _asked(body: dict, server: Server) -> _Completion:
    # what a request asks for of the server's loaded model
    return _completion(body, server, server.loaded)


def test_completion(server):
    # what a request asks for: OpenAI's fields and vLLM's, top_k of -1 keeping every token as 0
    messages = [{"role": "user", "content": "hi"}]
    body = {"messages": messages, "temperature": 0.5, "top_k": -1, "top_p": 0.9, "min_p": 0.05,
            "presence_penalty": 1.5, "frequency_penalty": -0.5, "repetition_penalty": 1.1,
            "max_tokens": 9, "max_completion_tokens": 7, "stop": "x", "seed": 3}  # fmt: skip
    asked = _asked(body, server)
    assert asked.sampling == Sampling(0.5, 0, 0.9, 0.05, 1.5, -0.5, 1.1)
    hi = server.loaded.engine.tokenizer.encode("hi")
    assert (asked.prompt, asked.max_tokens, asked.stop, asked.seed) == (hi, 7, ["x"], 3)
    # and by default, OpenAI's temperature of 1, up to the context and without empty stops
    asked = _asked({"messages": messages, "stop": ["", "y"], "top_p": None}, server)
    assert (asked.sampling, asked.max_tokens, asked.stop) == (Sampling(1.0), CONTEXT, ["y"])


def test_sampling_defaults(server, client, monkeypatch, tmp_path):
    # what a request leaves out of its sampling: the model's makers' set as it reasons, or as it
    # does not, under the operator's of every model and of it; listed with the model
    thinks, plain = {"temperature": 0.6, "top_k": 20}, {"temperature": 0.7, "top_k": 20}
    monkeypatch.setattr("leatengine.server.recommended", lambda metadata: (
        OPENAI | thinks, OPENAI | plain))  # fmt: skip
    messages = [{"role": "user", "content": "hi"}]
    on = {"chat_template_kwargs": {"enable_thinking": True}}
    assert _asked({"messages": messages} | on, server).sampling == Sampling(0.6, 20)
    # the tiny model's, which never reasons
    assert _asked({"messages": messages}, server).sampling == Sampling(0.7, 20)
    path = tmp_path / "sampling.json"
    path.write_text(json.dumps({"*": {"min_p": 0.05}, "tiny": {"top_k": 40}}))
    monkeypatch.setattr(server, "overrides", Overrides(path))
    asked = _asked({"messages": messages, "top_k": 5}, server)
    assert asked.sampling == Sampling(0.7, 5, 1.0, 0.05)  # the request's over all
    assert _asked({"messages": messages}, server).sampling == Sampling(0.7, 40, 1.0, 0.05)
    (model,) = client.models.list()
    assert model.sampling["plain"] == OPENAI | plain | {"top_k": 40, "min_p": 0.05}
    path.write_text("[]")  # broken: said to the client, the server going on
    with pytest.raises(openai.InternalServerError, match="sampling cannot be read"):
        chat(client, "hi", max_tokens=1)


@pytest.mark.parametrize("stream", [False, True])
def test_speculative_steps(tiny, tiny_assistant, stream):
    # a step that gives several tokens ends the reply at max_tokens or a stop string within them
    path = tiny("gemma4")[0]
    plain = Engine(path, max_context=CONTEXT, prefill_chunk=8)
    prompt = plain.tokenizer.encode("go on")
    tokens = list(plain.generate(prompt, 16))
    with serving(path, max_context=CONTEXT, prefill_chunk=8, draft=tiny_assistant[0]) as server:
        server.load("tiny")
        server.loaded.engine.drafter = Oracle(prompt + tokens)  # every step gives 4 tokens
        client = connect(server)
        reply = complete(client, "go on", max_tokens=6, temperature=0, stream=stream)
        assert reply == (plain.tokenizer.decode(tokens[:6]), "length")
        full = plain.tokenizer.decode(tokens)
        cut = len(plain.tokenizer.decode(tokens[:3]))  # within the first speculative step
        stop = full[cut : cut + 3]
        reply = complete(client, "go on", max_tokens=16, temperature=0, stop=[stop], stream=stream)
        assert reply == (full[: full.index(stop)], "stop")


def test_end_of_generation(client, engine, monkeypatch):
    # a reply ends at an end-of-generation token, which its text leaves out
    tokens = list(engine.generate(engine.tokenizer.encode("hello"), 8))
    monkeypatch.setattr(engine.tokenizer, "eog_ids", {tokens[3]})
    text = engine.tokenizer.decode(tokens[: tokens.index(tokens[3])])
    assert complete(client, "hello", max_tokens=8, temperature=0) == (text, "stop")


def test_seed(client):
    # sampled, as temperature is 1 unless given
    first, again = (complete(client, "seeded", max_tokens=8, seed=3) for _ in range(2))
    assert first == again
    # and cut, as Qwen3.6 recommends: top_k and min_p in OpenAI's client's extra_body
    options = {"top_p": 0.95, "presence_penalty": 1.5, "extra_body": {"top_k": 20, "min_p": 0.0}}
    first, again = (complete(client, "cut", max_tokens=8, seed=3, **options) for _ in range(2))
    assert first == again


def test_cached_tokens(client):
    chat(client, "a shared start, then one end", max_tokens=2)
    response = chat(client, "a shared start, then another", max_tokens=2)
    assert response.usage.prompt_tokens_details.cached_tokens == len("a shared start, then ")


def test_chat_template_kwargs(client):
    # options for the template, which the tiny model's writes before the last message
    response = chat(
        client, "hello", max_tokens=1, extra_body={"chat_template_kwargs": {"prefix": "ab"}}
    )
    assert response.usage.prompt_tokens == len("abhello")


def test_reasoning_effort(client, server, monkeypatch):
    # a model's efforts listed with it, and a request's told its template as it is told: the nearest
    # of its levels, over which chat_template_kwargs win; none of a model that takes none
    (model,) = client.models.list()
    assert getattr(model, "reasoning", None) is None
    chat_template, rendered = server.loaded.chat, []
    render = chat_template.render
    monkeypatch.setattr(chat_template, "_efforts", ("low", "medium", "high"))
    monkeypatch.setattr(chat_template, "_default", "medium")
    monkeypatch.setattr(chat_template, "_told", "reasoning_effort")
    monkeypatch.setattr(chat_template, "render", lambda m, **o: rendered.append(o) or render(m))
    (model,) = client.models.list()
    assert model.reasoning == {"efforts": ["low", "medium", "high"], "default": "medium"}
    for body in (
        {"reasoning_effort": "none"},
        {"reasoning_effort": "xhigh"},
        {},
        {"reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": "high"}},
    ):
        chat(client, "hello", max_tokens=1, extra_body=body)  # fmt: skip
    assert [o.get("reasoning_effort") for o in rendered] == ["low", "high", None, "high"]


def schema_format(schema: dict) -> dict:
    # a response format of a schema's JSON, as OpenAI's SDK writes one
    return {
        "type": "json_schema",
        "json_schema": {"name": "reply", "schema": schema, "strict": True},
    }


DEEP: dict = {"type": "null"}
for _ in range(64):
    DEEP = {"type": "array", "items": DEEP}
PICK = {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"enum": ["x", "yz"]}},
        "required": ["a", "b"], "additionalProperties": False}  # fmt: skip


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("schema", [PICK, None])
def test_response_format(client, stream, schema):
    # sampled replies held to a schema's JSON, or any object's: each whole, and so ended, with
    # "stop", as the tiny model has no end token, or cut short by max_tokens, with "length"
    response_format = {"type": "json_object"} if schema is None else schema_format(schema)
    matcher, whole = Matcher(Grammar(schema)), 0
    for seed in range(6):
        text, reason = complete(
            client, "hello", max_tokens=40, temperature=1, seed=seed, stream=stream,
            response_format=response_format,
        )  # fmt: skip
        state = matcher.feed(matcher.start, text.encode())
        assert state and matcher.done(state) == (reason == "stop"), (text, reason)
        if reason == "stop":
            value = json.loads(text)
            assert schema is None or value["a"] == int(value["a"]) and value["b"] in ("x", "yz")
            whole += 1
    assert whole or schema is None


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reply, called", [
    ('<tool_call>\n{"name": "weather", "arguments": {"city": "Paris"}}\n</tool_call>', True),
    ('{"name": "weather", "parameters": {"city": "Paris"}}', True),
    ('{"a": 1, "b": "x"}', False),
])  # fmt: skip
def test_response_format_beside_tools(client, replies_with, stream, reply, called):
    # a reply held to a schema's JSON, of tools it may call: a call, of text or of JSON, in its
    # place, or the JSON
    replies_with(reply)
    response = chat(client, "Weather in Paris?", tools=[WEATHER], stream=stream,
                    response_format=schema_format(PICK))  # fmt: skip
    if stream:
        choices = [chunk.choices[0] for chunk in response if chunk.choices]
        calls = [call for c in choices for call in c.delta.tool_calls or []]
        content, reason = "".join(c.delta.content or "" for c in choices), choices[-1].finish_reason
    else:
        message, reason = response.choices[0].message, response.choices[0].finish_reason
        calls, content = message.tool_calls or [], message.content
    if called:
        (call,) = calls
        assert reason == "tool_calls" and json.loads(call.function.arguments) == {"city": "Paris"}
    else:
        assert reason == "stop" and json.loads(content) == {"a": 1, "b": "x"} and not calls


def test_response_format_or_call(client):
    # sampled replies of the tiny model held to a schema's JSON or a call of JSON, as Llama 3's,
    # its vocab having no marker of a call: each whole one the one or the other
    seen = set()
    for seed in range(8):
        choice = chat(client, "hello", max_tokens=40, temperature=1, seed=seed, tools=[WEATHER],
                      response_format=schema_format(PICK)).choices[0]  # fmt: skip
        if choice.finish_reason == "tool_calls":
            assert choice.message.tool_calls[0].function.name == "weather"
        elif choice.finish_reason == "stop":
            value = json.loads(choice.message.content)
            assert value["a"] == int(value["a"]) and value["b"] in ("x", "yz")
        seen.add(choice.finish_reason)
    assert seen & {"stop", "tool_calls"}


def test_response_formats_compiled_once(client, server, monkeypatch):
    # a schema compiles once for the loaded model, its masks shared, until others more recently
    # asked for crowd it out
    compiled = []
    monkeypatch.setattr(server_module, "_GRAMMARS", 2)
    monkeypatch.setattr(
        server_module, "Grammar", lambda s, calls, **o: compiled.append(s) or Grammar(s, calls, **o)
    )
    for t in ("integer", "integer", "string", "boolean", "string", "integer"):
        schema = {"type": "object", "properties": {"v": {"type": t}}}
        chat(client, "hello", max_tokens=2, response_format=schema_format(schema))
    assert [s["properties"]["v"]["type"] for s in compiled] == [
        "integer", "string", "boolean", "integer"
    ]  # fmt: skip
    # properties in another order are another schema's
    for properties in ({"a": {}, "b": {}}, {"b": {}, "a": {}}):
        schema = {"type": "object", "properties": properties}
        chat(client, "hello", max_tokens=2, response_format=schema_format(schema))
    assert [list(s["properties"]) for s in compiled[-2:]] == [["a", "b"], ["b", "a"]]


@pytest.fixture
def replies_with(engine, monkeypatch):
    # makes the engine reply with the given text, which the tiny model would never write: each
    # step gives every active sequence its next token, and ends it at the last, an end of
    # generation unless the reply is to be cut off, as by max_tokens
    end = int(engine.model.output.shape[0]) - 1  # an end of generation, the tiny vocab has none
    monkeypatch.setattr(engine.tokenizer, "eog_ids", engine.tokenizer.eog_ids | {end})

    def reply_with(text: str, cut_off: bool = False) -> None:
        tokens = engine.tokenizer.encode(text) + ([] if cut_off else [end])

        def step():
            out = []
            for sequence in list(engine.active):
                sequence.tokens.append(token := tokens[len(sequence.tokens)])
                out.append((sequence, token))
                if len(sequence.tokens) == len(tokens):
                    engine.cancel(sequence)
            return out

        monkeypatch.setattr(engine, "step", step)

    return reply_with


@pytest.mark.parametrize("stream", [False, True])
def test_response(client, expected, stream):
    # a reply as a response of the Responses API: its text a message, cut off by
    # max_output_tokens as incomplete; streamed, each event numbered, the text in deltas
    asked = {"model": "tiny", "input": "hello", "max_output_tokens": 8, "temperature": 0}
    if stream:
        events = list(client.responses.create(**asked, stream=True))
        kinds = [e.type for e in events]
        assert kinds[:4] == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
        ]
        assert kinds[-4:] == ["response.output_text.done", "response.content_part.done",
                              "response.output_item.done", "response.incomplete"]  # fmt: skip
        assert [e.sequence_number for e in events] == list(range(len(events)))
        response = events[-1].response
        deltas = "".join(e.delta for e in events if e.type == "response.output_text.delta")
        assert deltas == response.output_text
    else:
        response = client.responses.create(**asked)
    assert response.output_text == expected("hello", 8) and response.id.startswith("resp_")
    assert response.status == "incomplete"
    assert response.incomplete_details.reason == "max_output_tokens"
    assert (response.usage.input_tokens, response.usage.output_tokens) == (5, 8)


@pytest.mark.parametrize("stream", [False, True])
def test_response_reasons_and_calls(client, replies_with, monkeypatch, stream):
    # Qwen3's reasoning, text and a call, as a reasoning item, a message and a function_call;
    # streamed, each item added and done in turn
    monkeypatch.setattr(ChatTemplate, "form", property(lambda self: "think"))
    replies_with("<think>Rain, I fear.</think>Checking.\n<tool_call>\n"
                 '{"name": "weather", "arguments": {"city": "Paris"}}\n</tool_call>')  # fmt: skip
    asked = {"model": "tiny", "input": "Weather in Paris?",
             "tools": [{"type": "function", **WEATHER["function"]}]}  # fmt: skip
    if stream:
        events = list(client.responses.create(**asked, stream=True))
        deltas = [e.delta for e in events if e.type == "response.reasoning_text.delta"]
        assert "".join(deltas) == "Rain, I fear."
        added = [e.output_index for e in events if e.type == "response.output_item.added"]
        done = [e.output_index for e in events if e.type == "response.output_item.done"]
        assert added == done == [0, 1, 2] and events[-1].type == "response.completed"
        response = events[-1].response
    else:
        response = client.responses.create(**asked)
    reasoning, message, call = response.output
    assert reasoning.content[0].text == "Rain, I fear." and message.content[0].text == "Checking."
    assert call.name == "weather" and json.loads(call.arguments) == {"city": "Paris"}
    assert call.call_id.startswith("call_") and response.status == "completed"


def test_response_text_format(client):
    # a text format of a schema, held to as a response_format's
    formatted = {"format": {"type": "json_schema", "name": "pick", "schema": PICK, "strict": True}}
    whole = 0
    for seed in range(6):
        response = client.responses.create(
            model="tiny", input="hello", max_output_tokens=40, temperature=1, text=formatted,
            extra_body={"seed": seed},
        )  # fmt: skip
        if response.status == "completed":
            assert json.loads(response.output_text)["b"] in ("x", "yz")
            whole += 1
    assert whole


@pytest.mark.parametrize("kwargs, error", [
    ({"previous_response_id": "resp_1"}, "previous_response_id is not supported"),
    ({"tools": [{"type": "function"}]}, "tools must be a list of objects, each"),
    ({"input": [{"type": "item_reference", "id": "x"}]}, "items of type 'item_reference'"),
    # of the fields as a response's request names them
    ({"max_output_tokens": 0}, "max_output_tokens must be a positive integer"),
    ({"text": {"format": {"type": "xml"}}}, 'text.format must be {"type": "text"}'),
])  # fmt: skip
def test_bad_response(client, kwargs, error):
    with pytest.raises(openai.BadRequestError, match=error):
        client.responses.create(**({"model": "tiny", "input": "hi"} | kwargs))


# as Llama 3, Qwen3, Gemma 4 and Qwen3.5 call tools; the text before a call is the content
CALLS = [
    (' {"name": "weather", "parameters": {"city": "Paris"}}', None),
    ('Checking.\n<tool_call>\n{"name": "weather", "arguments": {"city": "Paris"}}\n</tool_call>',
     "Checking."),
    ('<|tool_call>call:weather{city:<|"|>Paris<|"|>}<tool_call|>', None),
    ("Checking.\n\n<tool_call>\n<function=weather>\n<parameter=city>\nParis\n</parameter>\n"
     "</function>\n</tool_call>", "Checking."),
]  # fmt: skip


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reply, content", CALLS)
def test_tool_call(client, replies_with, stream, reply, content):
    replies_with(reply)
    response = chat(client, "Weather in Paris?", tools=[WEATHER], stream=stream)
    if stream:
        choices = [chunk.choices[0] for chunk in response]
        assert "".join(c.delta.content or "" for c in choices) == (content or "")
        (call,), reason = choices[-2].delta.tool_calls, choices[-1].finish_reason
    else:
        message, reason = response.choices[0].message, response.choices[0].finish_reason
        assert message.content == content
        (call,) = message.tool_calls
    assert call.type == "function" and call.id.startswith("call_") and reason == "tool_calls"
    assert call.function.name == "weather"
    assert json.loads(call.function.arguments) == {"city": "Paris"}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "form, reply",
    [("think", "<think>Sunny, I recall.</think>It is sunny."),
     ("think", "\n<think>Sunny, I recall.</think>It is sunny."),  # its newline not the text's
     ("gemma4", "<|channel>thought\nSunny, I recall.<channel|>It is sunny."),
     ("harmony", "<|channel|>analysis<|message|>Sunny, I recall.<|end|><|start|>assistant"
                 "<|channel|>final<|message|>It is sunny."),
     ("atem", " to=self<|message|>Sunny, I recall.<|eom|><|start|>assistant to=user<|message|>"
              "It is sunny.")],
)  # fmt: skip
def test_reasoning(client, replies_with, monkeypatch, stream, form, reply):
    # Qwen3's and gpt-oss's reasoning, apart from the text
    monkeypatch.setattr(ChatTemplate, "form", property(lambda self: form))
    replies_with(reply)
    response = chat(client, "Weather in Paris?", stream=stream)
    if stream:
        deltas = [chunk.choices[0].delta for chunk in response if chunk.choices]
        reasoning = "".join(getattr(d, "reasoning_content", None) or "" for d in deltas)
        content = "".join(d.content or "" for d in deltas)
    else:
        message = response.choices[0].message
        reasoning, content = getattr(message, "reasoning_content", None), message.content
    assert (reasoning, content) == ("Sunny, I recall.", "It is sunny.")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "form, reply, content",
    [(None, "if a <", "if a <"), ("think", "<think>b</think>a </", "a </"),
     ("harmony", "<|channel|>final<|message|>a <|", "a <|")],
)  # fmt: skip
def test_reply_ends_as_a_marker_begins(
    client, replies_with, monkeypatch, stream, form, reply, content
):
    # an end that may begin a marker waits for more only until the reply is done
    monkeypatch.setattr(ChatTemplate, "form", property(lambda self: form))
    replies_with(reply)
    assert complete(client, "Weather in Paris?", stream=stream)[0] == content


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "header, body, name, arguments",
    [
        ("commentary to=functions.weather <|constrain|>json", '{"city": "Paris"}', "weather",
         '{"city": "Paris"}'),
        # passed on as the model made it, as OpenAI's API does, for the client to say what is
        # wrong, rather than the reply ending with nothing: a tool not given, gpt-oss's own
        # python, and arguments that are no JSON object
        ("commentary to=functions.python", "print(1)", "python", "print(1)"),
        ("analysis to=python code", "print(1)", "python", "print(1)"),
        ("commentary to=functions.weather", '{"city": Paris}', "weather", '{"city": Paris}'),
        # a tool's name run into its channel, the <|channel|> between left out, as gpt-oss writes
        ("to=functions.weathercommentary json", '{"city": "Paris"}', "weather",
         '{"city": "Paris"}'),
    ],
)  # fmt: skip
def test_harmony_tool_call(client, replies_with, monkeypatch, stream, header, body, name,
                           arguments):  # fmt: skip
    monkeypatch.setattr(ChatTemplate, "form", property(lambda self: "harmony"))
    replies_with(f"<|channel|>{header}<|message|>{body}")
    response = chat(client, "Weather in Paris?", tools=[WEATHER], stream=stream)
    if stream:
        choices = [chunk.choices[0] for chunk in response]
        (call,), reason = choices[-2].delta.tool_calls, choices[-1].finish_reason
        content = "".join(c.delta.content or "" for c in choices)
    else:
        (call,), reason = response.choices[0].message.tool_calls, response.choices[0].finish_reason
        content = response.choices[0].message.content or ""
    assert reason == "tool_calls" and (call.function.name, call.function.arguments) == (
        name, arguments)  # fmt: skip
    assert content == ""


@pytest.mark.parametrize("stream", [False, True])
def test_atem_tool_call(client, replies_with, monkeypatch, stream):
    # Muse Glimmer's call: an ATEM block to the tool, its text values typed as the tool takes them
    monkeypatch.setattr(ChatTemplate, "form", property(lambda self: "atem"))
    replies_with(" to=self<|message|>Need a tool.<|eom|><|start|>assistant to=weather<|message|>"
                 '<atem:function_calls>\n<atem:invoke name="weather">\n<atem:parameter name="city">'
                 "Paris</atem:parameter>\n</atem:invoke>\n</atem:function_calls>")  # fmt: skip
    response = chat(client, "Weather in Paris?", tools=[WEATHER], stream=stream)
    if stream:
        choices = [chunk.choices[0] for chunk in response]
        (call,), reason = choices[-2].delta.tool_calls, choices[-1].finish_reason
    else:
        (call,), reason = response.choices[0].message.tool_calls, response.choices[0].finish_reason
    assert reason == "tool_calls" and call.function.name == "weather"
    assert json.loads(call.function.arguments) == {"city": "Paris"}


def test_tool_choice_required(client, server, monkeypatch):
    # sampled replies of the tiny model, its vocab without a call's marker, each a call of JSON, as
    # Llama 3's, or one cut short; of a function named, of it, its tool alone declared. Not of
    # harmony, whose calls come in its channels.
    time = {"type": "function", "function": {"name": "time", "parameters": {"type": "object"}}}
    called = set()
    for seed in range(6):
        for choice in ("required", {"type": "function", "function": {"name": "time"}}):
            reply = chat(client, "hello", max_tokens=40, temperature=1, seed=seed,
                         tools=[WEATHER, time], tool_choice=choice).choices[0]  # fmt: skip
            assert reply.finish_reason in ("tool_calls", "length")
            if reply.finish_reason == "tool_calls":
                names = {call.function.name for call in reply.message.tool_calls}
                assert names <= ({"time"} if isinstance(choice, dict) else {"weather", "time"})
                called |= names
    assert called
    monkeypatch.setattr(ChatTemplate, "form", property(lambda self: "harmony"))
    with pytest.raises(openai.BadRequestError, match="calls tools in its format's channels"):
        chat(client, "hello", tools=[WEATHER], tool_choice="required")


@pytest.mark.parametrize("stream", [False, True])
def test_tool_choice_none(client, server, replies_with, monkeypatch, stream):
    # tool_choice "none" as OpenAI's API has it: of harmony, the tools still declared, so that the
    # prompt is as of "auto", and the reply opened in the final channel, an answer, whatever call
    # it goes on to; of a format of no such channel, the tools left out
    chat_template, rendered, texts = server.loaded.chat, [], []
    render, tokens = chat_template.render, chat_template.tokens
    monkeypatch.setattr(chat_template, "render", lambda m, **o: rendered.append(o) or render(m))
    monkeypatch.setattr(chat_template, "tokens", lambda t, i=(): texts.append(t) or tokens(t, i))
    kwargs = {"tools": [WEATHER], "tool_choice": "none", "stream": stream}
    monkeypatch.setattr(ChatTemplate, "form", property(lambda self: "harmony"))
    replies_with("It is sunny.<|end|><|start|>assistant<|channel|>commentary to=functions.weather"
                 '<|message|>{"city": "Paris"}')  # fmt: skip
    assert complete(client, "Weather in Paris?", **kwargs)[0] == "It is sunny."
    assert rendered[-1]["tools"] == [WEATHER]
    assert texts[-1].endswith("<|channel|>final<|message|>")
    monkeypatch.setattr(ChatTemplate, "form", property(lambda self: None))
    replies_with("It is sunny.")
    assert complete(client, "Weather in Paris?", **kwargs)[0] == "It is sunny."
    assert "tools" not in rendered[-1] and not texts[-1].endswith("<|message|>")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "text, choice",
    [
        ("It is sunny.", "auto"),
        ('{"name": "unknown", "parameters": {}}', "auto"),
        ('{"name": "weather", "parameters": {}}', "none"),
    ],
)
def test_text_is_not_a_tool_call(client, replies_with, stream, text, choice):
    replies_with(text)
    kwargs = {"tools": [WEATHER], "tool_choice": choice, "stream": stream}
    assert complete(client, "Weather in Paris?", **kwargs) == (text, "stop")


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"n": 2}, "n=2 is not supported"),
        ({"frequency_penalty": 2.5}, "frequency_penalty must be a number from -2 to 2"),
        ({"logit_bias": {"a": 1}}, "logit_bias must be an object of token ids to numbers from"),
        ({"logit_bias": {"7": 101}}, "logit_bias must be an object of token ids to numbers from"),
        ({"logit_bias": {"999999": 1}}, "logit_bias names token 999999, past the vocab of"),
        ({"extra_body": {"repetition_penalty": 0}}, "repetition_penalty must be a number from 0.1"),
        ({"top_p": 1.5}, "top_p must be a number from 0 to 1"),
        ({"temperature": 3}, "temperature must be a number from 0 to 2"),
        ({"temperature": True}, "temperature must be a number from 0 to 2"),
        ({"extra_body": {"top_k": 1.5}}, "top_k must be an integer"),
        ({"max_tokens": 0}, "max_tokens must be a positive integer"),
        ({"stop": "x" * 257}, "stop must be a string or a list of 16 strings at most"),
        ({"stop": ["x"] * 17}, "stop must be a string or a list of 16 strings at most"),
        ({"tool_choice": "required"}, "tool_choice='required' needs tools to call"),
        ({"tools": [WEATHER], "tool_choice": "any"}, "tool_choice='any' is not supported"),
        (
            {"tools": [WEATHER], "tool_choice": {"type": "function", "function": {"name": "news"}}},
            "tool_choice names 'news', which is none of the tools",
        ),
        ({"messages": []}, "messages must be a non-empty list of objects"),
        ({"messages": [{"role": "user", "content": "x" * CONTEXT}]}, "the prompt has 64 tokens"),
        ({"extra_body": {"chat_template_kwargs": "x"}}, "chat_template_kwargs must be an object"),
        ({"extra_body": {"reasoning_effort": "max"}}, "reasoning_effort must be one of none,"),
        # what a message or tool holds, of the shapes templates read
        ({"messages": [{"role": "user", "content": [None]}]}, "text, image or image_url"),
        ({"messages": [{"role": "user", "content": [{"type": "text"}]}]}, "text must be a string"),
        ({"messages": [{"role": "assistant", "tool_calls": [{"id": "x"}]}]}, "each with its"),
        ({"tools": [{"function": "weather"}]}, "tools must be a list of objects, each"),
        ({"tools": [{"function": {"name": []}}]}, "tools must be a list of objects, each"),
        (
            {"messages": [{"role": "assistant", "tool_calls": [{"function": {"name": "f"}}]}]},
            "each with its function's name and arguments",
        ),
        # log-probabilities of a chat asked for as a text completion's, or of the likeliest
        # tokens alone, or of too many
        ({"logprobs": 2}, "logprobs of a chat completion must be true or false"),
        ({"top_logprobs": 2}, "top_logprobs needs logprobs to be true"),
        ({"logprobs": True, "top_logprobs": 21}, "top_logprobs must be an integer from 0 to 20"),
        # a response format of a type or schema it does not take
        ({"response_format": {"type": "xml"}}, 'response_format must be {"type": "text"}'),
        ({"response_format": {"type": "json_schema"}}, 'response_format must be {"type": "text"}'),
        ({"response_format": schema_format({"format": "x"})}, "the schema's format is not"),
        ({"response_format": schema_format(DEEP)}, "the schema nests deeper than 64 levels"),
    ],
)
def test_bad_request(client, kwargs, error):
    request = {"model": "tiny", "messages": [{"role": "user", "content": "hi"}]} | kwargs
    with pytest.raises(openai.BadRequestError, match=error):
        client.chat.completions.create(**request)


@pytest.mark.parametrize(
    "body, error",
    [
        ({}, "a completion needs prompt"),
        ({"prompt": []}, "prompt must be a string, a list of one string, or a non-empty list"),
        ({"prompt": ["a", "b"]}, "prompt must be a string, a list of one string"),
        ({"prompt": [1, True]}, "prompt must be a string, a list of one string"),
        ({"prompt": [10**6]}, "the prompt must hold some tokens, each of the"),
        ({"prompt": "x" * CONTEXT}, "the prompt has 64 tokens, too many for 64 of context"),
        ({"prompt": "a", "suffix": 1}, "suffix must be a string"),
        ({"prompt": "a", "echo": True}, "echo=True is not supported"),
        ({"prompt": "a", "best_of": 2}, "best_of=2 is not supported"),
        ({"prompt": "a", "logprobs": True}, "logprobs of a text completion must be a count"),
        ({"prompt": "a", "logprobs": 21}, "logprobs must be true or false, or of a text"),
        ({"prompt": "a", "top_logprobs": 2}, "top_logprobs is a chat completion's"),
    ],
)
def test_bad_text_completion(client, body, error):
    with pytest.raises(openai.BadRequestError, match=error):
        client.post("/completions", cast_to=object, body={"model": "tiny"} | body)


def test_web_pages_are_refused(server):
    # a page may not make the server generate or load models, even one whose name was made this
    # address's, as DNS rebinding does; a client of no browser may
    url = f"http://127.0.0.1:{server.server_port}"
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}).encode()
    rebound = {"Origin": "http://evil.example:8080", "Host": "evil.example:8080"}
    for path, data in (("/v1/chat/completions", body), ("/v1/models/load", b'{"model": "tiny"}')):
        for headers in ({"Origin": "https://evil.example"}, rebound):
            with pytest.raises(urllib.error.HTTPError, match="403"):
                urllib.request.urlopen(urllib.request.Request(url + path, data, headers))
    with urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", body)) as r:
        assert r.status == 200


def test_unknown_route(client, server):
    with pytest.raises(openai.NotFoundError):
        client.moderations.create(input="hi")
    url = f"http://127.0.0.1:{server.server_port}"
    with pytest.raises(urllib.error.HTTPError, match="404"):
        urllib.request.urlopen(f"{url}/v1/nothing")
    with pytest.raises(urllib.error.HTTPError, match="400"):  # not JSON
        urllib.request.urlopen(urllib.request.Request(f"{url}/v1/chat/completions", b"{"))
    with urllib.request.urlopen(f"{url}/v1/models?x=1") as response:  # the path, its query aside
        assert response.status == 200
    # a body's length that is no length, answered rather than read till the client hangs up
    head = "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: -1\r\n\r\n{}"
    with socket.create_connection(("127.0.0.1", server.server_port), timeout=10) as connection:
        connection.sendall(head.encode())
        assert connection.recv(64).startswith(b"HTTP/1.0 400")


def metrics(server: Server, key: str | None = None) -> dict[str, float]:
    """/metrics' samples, by their names and labels as written."""
    url, headers = f"http://127.0.0.1:{server.server_port}/metrics", {}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(url, None, headers)
    with urllib.request.urlopen(request) as response:
        assert response.headers["Content-Type"].startswith("text/plain; version=0.0.4")
        lines = response.read().decode().splitlines()
    return {k: float(v) for k, v in (line.rsplit(" ", 1) for line in lines if line[0] != "#")}


def test_metrics(client, server):
    # each completion's tokens and time counted, its requests by path and status, and the slots
    # and model of now; with no keys, of no key's
    before = metrics(server)
    _, reason = complete(client, "hello", max_tokens=3, temperature=0)
    after = metrics(server)
    grown = lambda name: after.get(name, 0) - before.get(name, 0)  # noqa: E731
    assert grown('leatengine_completion_tokens_total{key=""}') == 3
    assert grown('leatengine_completions_total{finish="length",key=""}') == 1 and reason == "length"
    assert grown('leatengine_prompt_tokens_total{key=""}') > 0
    assert grown('leatengine_decode_seconds_total{key=""}') > 0
    ok = 'leatengine_requests_total{key="",path="/v1/chat/completions",status="200"}'
    assert grown(ok) == 1
    assert grown('leatengine_requests_total{key="",path="/metrics",status="200"}') == 1
    assert after["leatengine_slots"] == 2 and after["leatengine_slots_busy"] == 0
    assert after["leatengine_completions_waiting"] == 0
    assert after['leatengine_model_loaded{model="tiny"}'] == CONTEXT


def test_prefix_store(tiny_model, monkeypatch):
    # a chat's next turn, after another took the one slot, comes back from the prefix store: its
    # tokens counted as cached and as stored, and the store's bytes. Tiles of 8 positions, as the
    # tiny model's context is 64.
    for module in ("leatengine.model", "leatengine.engine"):
        monkeypatch.setattr(f"{module}.CACHE_TILE", 8)
    options = {"max_context": CONTEXT, "prefill_chunk": 8, "slots": 1, "prefix_store": 1 << 30}
    with serving(tiny_model[0], **options) as server:
        server.load("tiny")
        client, first = connect(server), "a first question, long enough"
        for content in (first, "another one, as long as it"):
            chat(client, content, max_tokens=2, temperature=0)
        reply = chat(client, first + " and more", max_tokens=2, temperature=0)
        counted = metrics(server)
        stored = counted['leatengine_stored_tokens_total{key=""}']
        assert stored == reply.usage.prompt_tokens_details.cached_tokens >= len(first)
        assert counted["leatengine_prefix_store_bytes"] > 0
        server.shutdown()


def test_keys(tiny_model, tmp_path):
    # given keys, every request must hold one, as OpenAI's clients send it or Anthropic's, and
    # each key's use is counted apart; a key removed is refused from the next request on
    keys = Keys(tmp_path / "keys.json")
    alkin, agent = keys.add("alkin"), keys.add("agent")
    with serving(tiny_model[0], max_context=CONTEXT, prefill_chunk=8, keys=keys) as server:
        server.load("tiny")
        url = f"http://127.0.0.1:{server.server_port}"
        for headers in (
            {},
            {"Authorization": "Bearer leatengine-wrong"},
            {"x-api-key": alkin[:-1]},
        ):
            for path in ("/v1/models", "/metrics"):
                with pytest.raises(urllib.error.HTTPError) as refused:
                    urllib.request.urlopen(urllib.request.Request(url + path, None, headers))
                assert refused.value.code == 401
                assert refused.value.headers["WWW-Authenticate"].startswith("Bearer")
                error = json.loads(refused.value.read())["error"]
                assert error["code"] == "invalid_api_key"
        with pytest.raises(openai.AuthenticationError):
            complete(connect(server), "hello", max_tokens=1)
        client = openai.OpenAI(base_url=f"{url}/v1", api_key=alkin, max_retries=0)
        assert complete(client, "hello", max_tokens=2, temperature=0)[1] == "length"
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1})
        anthropic = {"x-api-key": agent, "Content-Type": "application/json"}
        request = urllib.request.Request(f"{url}/v1/chat/completions", body.encode(), anthropic)
        with urllib.request.urlopen(request) as response:
            assert response.status == 200
        # an empty x-api-key, as a client may send by default, leaves the bearer key to count
        both = {"x-api-key": "", "Authorization": f"Bearer {alkin}"}
        with urllib.request.urlopen(urllib.request.Request(url + "/v1/models", None, both)) as r:
            assert r.status == 200
        counted = metrics(server, agent)
        assert counted['leatengine_completion_tokens_total{key="alkin"}'] == 2
        assert counted['leatengine_completion_tokens_total{key="agent"}'] == 1
        assert counted['leatengine_requests_total{key="",path="/v1/models",status="401"}'] == 3
        keys.remove("alkin")
        with pytest.raises(openai.AuthenticationError):
            client.models.list()
        server.shutdown()


def test_keys_share_their_own_prefixes(tiny_model, tmp_path):
    # a key's prompts start from the cached prefixes of its own alone: another key's that shares
    # them is prefilled whole, so that it cannot tell from its time to the first token, or its
    # cached tokens, what the first asked
    keys = Keys(tmp_path / "keys.json")
    alkin, agent = keys.add("alkin"), keys.add("agent")
    options = {"max_context": CONTEXT, "prefill_chunk": 8, "slots": 2, "keys": keys}
    with serving(tiny_model[0], **options) as server:
        server.load("tiny")
        url = f"http://127.0.0.1:{server.server_port}/v1"
        first, second = (openai.OpenAI(base_url=url, api_key=key, max_retries=0)
                         for key in (alkin, agent))  # fmt: skip

        def cached(client: openai.OpenAI, content: str) -> int:
            return chat(client, content, max_tokens=2).usage.prompt_tokens_details.cached_tokens

        assert cached(first, "a shared start, then one end") == 0
        assert cached(second, "a shared start, then another") == 0
        assert cached(first, "a shared start, then another") == len("a shared start, then ")
        assert cached(second, "a shared start, then more") == len("a shared start, then ")
        server.shutdown()


@pytest.mark.parametrize("keyed", [True, False])
@pytest.mark.parametrize("field", ["user", "safety_identifier"])
def test_users_share_their_own_prefixes(tiny_model, tmp_path, keyed, field):
    # an end user a request names, as an app serving several people does, shares cached prefixes
    # with that user's prompts alone, under one key or without keys; prompts that name no one
    # share with each other
    keys = Keys(tmp_path / "keys.json") if keyed else None
    key = keys.add("agent") if keys else "unused"
    options = {"max_context": CONTEXT, "prefill_chunk": 8, "slots": 3, "keys": keys}
    with serving(tiny_model[0], **options) as server:
        server.load("tiny")
        url = f"http://127.0.0.1:{server.server_port}/v1"
        client = openai.OpenAI(base_url=url, api_key=key, max_retries=0)

        def cached(content: str, user: str | None = None) -> int:
            named = {"extra_body": {field: user}} if user else {}
            response = chat(client, content, max_tokens=2, **named)
            return response.usage.prompt_tokens_details.cached_tokens

        shared = len("a shared start, then ")
        assert cached("a shared start, then one end", "person-1") == 0
        assert cached("a shared start, then another", "person-2") == 0
        assert cached("a shared start, then more") == 0
        assert cached("a shared start, then yet more", "person-1") == shared
        assert cached("a shared start, then the last", "person-2") == shared
        assert cached("a shared start, then no one's") == shared
        with pytest.raises(openai.BadRequestError, match=f"{field} must be a string"):
            chat(client, "hi", max_tokens=1, extra_body={field: 1})
        server.shutdown()


def test_concurrent_requests(client, server, engine, expected, monkeypatch):
    # more requests at once than the engine's 2 slots, whole and streamed: each gets the reply it
    # would alone, two in batched steps, the third once a slot is free
    batches, decode_step, step = [], engine._decode_step, engine.step
    monkeypatch.setattr(engine, "_decode_step", lambda s: batches.append(len(s)) or decode_step(s))
    contents, replies, first = ["hello", "a b c", "the third one"], {}, threading.Event()

    def step_once_queued() -> list:
        # the first request's first step lets the others in and waits until they are queued, so
        # that they come before it finishes however fast the engine is: taking them off the
        # queue waits for them, and they go back in order
        if not first.is_set():
            first.set()
            queued = [server.requests.get() for _ in contents[1:]]
            for request in queued:
                server.requests.put(request)
        return step()

    monkeypatch.setattr(engine, "step", step_once_queued)

    def ask(content: str, stream: bool) -> None:
        if content != contents[0]:
            first.wait()
        replies[content] = complete(client, content, max_tokens=12, temperature=0, stream=stream)

    threads = [threading.Thread(target=ask, args=(c, i > 0)) for i, c in enumerate(contents)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert replies == {c: (expected(c, 12), "length") for c in contents}
    assert max(batches) == 2


def test_hang_up_frees_the_slot(client, engine):
    # a client that hangs up mid-reply ends its sequence, while another's carries on
    other = chat(client, "a long reply", max_tokens=30, stream=True)
    gone = chat(client, "another long reply", max_tokens=30, stream=True)
    next(iter(gone))
    gone.close()
    choices = [chunk.choices[0] for chunk in other if chunk.choices]
    assert choices[-1].finish_reason == "length"
    assert not engine.active


def test_hang_up_before_the_reply_frees_the_slot(server, engine, monkeypatch):
    # a client of a whole reply that hangs up while it waits, which no write would find, ends its
    # sequence too, steps of 50 ms long before its 40 tokens
    step, steps = engine.step, []
    monkeypatch.setattr(engine, "step", lambda: steps.append(time.sleep(0.05)) or step())
    request = {"model": "tiny", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 40}
    body = json.dumps(request).encode()
    head = f"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {len(body)}\r\n\r\n"
    with socket.create_connection(("127.0.0.1", server.server_port)) as connection:
        connection.sendall(head.encode() + body)
        while not engine.active:
            time.sleep(0.01)
    while engine.active:
        time.sleep(0.01)
    assert len(steps) < 20


def test_no_model_loaded(server):
    # of a load that failed: the server's state, a 503, as /health says it
    with pytest.raises(server_module._Unloaded, match="no model is loaded"):
        server_module._completion({"messages": [{"role": "user", "content": "hi"}]}, server, None)
    assert issubclass(server_module._Unloaded, ValueError)


def test_text_completion_max_completion_tokens(client):
    # as chat completions take it, over max_tokens
    reply = client.completions.create(model="tiny", prompt="hello", max_tokens=8,
                                      extra_body={"max_completion_tokens": 3})  # fmt: skip
    assert reply.usage.completion_tokens == 3


def test_engine_error(client, engine, monkeypatch):
    # a failing step fails every running completion, and the server serves the next
    def fail():
        monkeypatch.undo()
        raise RuntimeError("out of memory")

    monkeypatch.setattr(engine, "step", fail)
    with pytest.raises(openai.InternalServerError, match="out of memory"):
        chat(client, "hello", max_tokens=4)
    # what the step wrote of the cache is not known: no prefix is shared on
    assert not any(engine._cached)
    assert complete(client, "hello", max_tokens=2)[1] == "length"


def test_unexpected_error(client, server, monkeypatch):
    # a request that fails in a way the handler did not expect is answered, not dropped
    def fail(*args):
        raise KeyError("a bug")

    monkeypatch.setattr("leatengine.server._completion", fail)
    with pytest.raises(openai.InternalServerError, match="failed to answer"):
        chat(client, "hello", max_tokens=1)
    monkeypatch.undo()
    # a body nested too deeply to parse, or to render: sent as text, as Python 3.12 can neither
    # encode nor decode it
    nested = "[" * 20000 + "]" * 20000
    body = f'{{"messages": [{{"role": "user", "content": "hi", "x": {nested}}}], "max_tokens": 1}}'
    url = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
    with pytest.raises(urllib.error.HTTPError, match="400") as refused:
        urllib.request.urlopen(urllib.request.Request(url, body.encode()))
    assert "nested too deeply" in json.loads(refused.value.read())["error"]["message"]
    # a body too long to read is refused before it is read
    head = "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 4611686018427387904\r\n\r\n"
    with socket.create_connection(("127.0.0.1", server.server_port), timeout=10) as connection:
        connection.sendall(head.encode())
        assert connection.recv(64).startswith(b"HTTP/1.0 413")


def test_completion_rendered_as_a_load_begins(tiny_model, tmp_path, monkeypatch):
    # a completion rendered by one model's template while a load of another begins is rendered
    # again, by the other's, rather than failing once its model is gone
    (other := tmp_path / "other.gguf").symlink_to(tiny_model[0])
    completion, loading = server_module._completion, []

    def rendered(body, server, loaded):
        asked = completion(body, server, loaded)
        if not loading:  # the first time, a load begins meanwhile
            loading.append(threading.Thread(target=server.load, args=("other",)))
            loading[0].start()
            while server.ready.is_set():
                time.sleep(0.001)
        return asked

    monkeypatch.setattr(server_module, "_completion", rendered)
    with serving(tiny_model[0], other, max_context=CONTEXT) as server:
        server.load("tiny")
        assert chat(connect(server), "hello", max_tokens=2).model == "other"
        loading[0].join()


def test_worker_error(client, monkeypatch):
    # a failure past the engine's step, a bug's, fails the completions rather than the worker
    def fail(self, token, score):
        monkeypatch.undo()
        raise KeyError("a bug")

    monkeypatch.setattr(_Writer, "take", fail)
    with pytest.raises(openai.InternalServerError, match="a bug"):
        chat(client, "hello", max_tokens=4)
    assert complete(client, "hello", max_tokens=2)[1] == "length"


def test_worker_error_ends_a_waiting_load(server):
    # completions waiting for a load the error ended go on, rather than waiting forever
    server.ready.clear()
    server._loads += 1  # as load() asks for one
    load = _Load("tiny")
    server._fail(KeyError("a bug"), collections.deque([load]), {})
    assert isinstance(load.done.get(), KeyError) and server.ready.is_set()


def test_loading_the_loaded_model_waits_for_nothing(client, server, monkeypatch):
    # not even for the completions running, which a load of another model waits out. The worker
    # holds the completion unstepped, taking requests meanwhile, rather than waiting in a step,
    # which would hold the load back as well
    going, step = threading.Event(), Server._step
    monkeypatch.setattr(Server, "_step",
                        lambda self, running: step(self, running) if going.is_set()
                        else time.sleep(0.001))  # fmt: skip
    reply = threading.Thread(target=chat, args=(client, "hello"), kwargs={"max_tokens": 2})
    reply.start()
    deadline = time.monotonic() + 5
    while server.busy == 0 and time.monotonic() < deadline:  # the completion runs
        time.sleep(0.01)
    assert server.busy == 1
    loading = threading.Thread(target=server.load, args=("tiny",))
    loading.start()
    loading.join(2)
    assert not loading.is_alive()
    going.set()
    reply.join()


@pytest.fixture(scope="module")
def served(model_path) -> Iterator[openai.OpenAI]:
    with serving(model_path, max_context=4096, slots=4) as server:
        server.load(model_path.stem)
        yield connect(server)


@pytest.mark.gpu
@pytest.mark.model
def test_calls_tools(served, model_path):
    messages = [{"role": "user", "content": "What is the weather in Paris right now?"}]
    metadata = GGUF.open(model_path).metadata
    template = ChatTemplate(metadata, Tokenizer(metadata))
    if template.render(messages, tools=[WEATHER]) == template.render(messages):
        # as Phi-4-mini's, which shows tools a system message holds, not the request's
        pytest.skip("the model's chat template shows no tools")
    response = served.chat.completions.create(
        model="real", messages=messages, tools=[WEATHER], temperature=0
    )
    (call,) = response.choices[0].message.tool_calls
    assert call.function.name == "weather"
    assert json.loads(call.function.arguments) == {"city": "Paris"}


@pytest.mark.gpu
@pytest.mark.model
def test_responds_in_a_loop(served, model_path):
    # an agent's loop through the Responses API, stateless: the model calls a tool, the client
    # sends back the conversation whole, the call's output too, and the model answers. Whether it
    # answers with the output is the model's: Qwen3 8B says it snows, Llama 3.1 8B writes out its
    # tool's definition, of the same prompt as a chat completion's.
    metadata = GGUF.open(model_path).metadata
    template = ChatTemplate(metadata, Tokenizer(metadata))
    question = [{"role": "user", "content": "What is the weather in Paris right now?"}]
    if template.render(question, tools=[WEATHER]) == template.render(question):
        pytest.skip("the model's chat template shows no tools")
    tools: list = [{"type": "function", **WEATHER["function"]}]
    items: list = list(question)
    first = served.responses.create(model="real", input=items, tools=tools, temperature=0)
    calls = [item for item in first.output if item.type == "function_call"]
    assert calls and calls[0].name == "weather"
    items += [item.model_dump(exclude_none=True) for item in first.output]
    items += [{"type": "function_call_output", "call_id": call.call_id,
               "output": "In Paris it is snowing heavily, at -5 C."} for call in calls]  # fmt: skip
    second = served.responses.create(model="real", input=items, tools=tools, temperature=0)
    assert second.status == "completed" and second.output_text
    assert second.usage.input_tokens > first.usage.input_tokens + first.usage.output_tokens


@pytest.mark.gpu
@pytest.mark.model
def test_calls_then_answers_in_json(served, model_path):
    # as an agent SDK's structured output asks: tools and a schema at once, the model calling a
    # tool in place of the JSON, then, given the call's output, answering in the JSON
    metadata = GGUF.open(model_path).metadata
    template = ChatTemplate(metadata, Tokenizer(metadata))
    messages: list = [{"role": "user", "content": "What is the weather in Paris right now?"}]
    if template.render(messages, tools=[WEATHER]) == template.render(messages):
        pytest.skip("the model's chat template shows no tools")
    schema = {"type": "object", "properties": {"city": {"type": "string"},
              "weather": {"type": "string"}}, "required": ["city", "weather"],
              "additionalProperties": False}  # fmt: skip
    asked = {"model": "real", "tools": [WEATHER], "temperature": 0,
             "response_format": schema_format(schema)}  # fmt: skip
    first = served.chat.completions.create(messages=messages, **asked).choices[0]
    (call,) = first.message.tool_calls
    assert first.finish_reason == "tool_calls" and call.function.name == "weather"
    messages += [first.message.model_dump(exclude_none=True),
                 {"role": "tool", "tool_call_id": call.id,
                  "content": "In Paris it is snowing heavily, at -5 C."}]  # fmt: skip
    # of the output, the answer, as Qwen3 8B gives, or another call, as Llama 3.1 8B may: of
    # either, never text of no JSON
    second = served.chat.completions.create(messages=messages, **asked).choices[0]
    if second.finish_reason == "tool_calls":
        assert second.message.tool_calls[0].function.name == "weather"
    else:
        answer = json.loads(second.message.content)
        assert second.finish_reason == "stop" and answer["city"] == "Paris"


@pytest.mark.gpu
@pytest.mark.model
def test_answers_in_json(served):
    # a reply held to a schema, after the model's reasoning, if it reasons
    schema = {"type": "object", "properties": {"city": {"type": "string"},
              "country": {"type": "string"}}, "required": ["city", "country"],
              "additionalProperties": False}  # fmt: skip
    messages = [{"role": "user", "content": "Name the capital of France, as JSON."}]
    response = served.chat.completions.create(
        model="real", messages=messages, temperature=0, response_format=schema_format(schema)
    )
    choice = response.choices[0]
    assert choice.finish_reason == "stop"
    assert json.loads(choice.message.content) == {"city": "Paris", "country": "France"}


@pytest.mark.gpu
@pytest.mark.model
def test_shared_system_prompt(served, model_path):
    # a system prompt that another conversation cached cuts the time to the first token by 10x or
    # more: from 449 to 22 ms for these 2141 tokens on the 3090
    if "system" not in chat_template(model_path):
        pytest.skip("the model's chat template takes no system prompt")

    def system(name: str) -> str:
        rule = "{} {}: answer plainly, cite source {}, and keep replies under {} words."
        return " ".join(rule.format(name, i, i, i + 50) for i in range(100))

    def first_token(system: str, question: str) -> float:
        # the time to a reply of one token: text, or for a model that thinks first, reasoning
        messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
        start = time.perf_counter()
        served.chat.completions.create(model="real", messages=messages, max_tokens=1, temperature=0)
        return time.perf_counter() - start

    for question in ("Hi", "Hello"):  # captures the graphs, the copy's too
        first_token(system("Warm"), question)
    cold, warm = [], []
    for name in ("Rule", "Law", "Step"):
        cold.append(first_token(system(name), "What is rule 3?"))
        warm.append(first_token(system(name), "And rule 7?"))
    assert statistics.median(cold) > 10 * statistics.median(warm)


@pytest.mark.parametrize("stream", [False, True])
def test_cut_off_call_is_no_text(client, replies_with, stream):
    # whole or streamed, a reply that ends for its length in a call it had begun says the text
    # before the call alone
    replies_with('Checking.\n<tool_call>\n{"name": "weather", "argu', cut_off=True)
    kwargs = {"tools": [WEATHER], "stream": stream}
    assert complete(client, "Weather in Paris?", **kwargs) == ("Checking.", "length")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("form, reply", [
    ("harmony", "<|channel|>commentary to=functions.weather <|constrain|>json<|message|>"
                '{"city": "Pa'),
    ("atem", " to=weather<|message|>city: Pa"),
])  # fmt: skip
def test_cut_off_harmony_call(client, replies_with, monkeypatch, stream, form, reply):
    # a call of a format whose calls end its reply, cut off by max_tokens: no call, as of text
    monkeypatch.setattr(ChatTemplate, "form", property(lambda self: form))
    replies_with(reply, cut_off=True)
    response = chat(client, "Weather in Paris?", tools=[WEATHER], stream=stream)
    if stream:
        choices = [chunk.choices[0] for chunk in response if chunk.choices]
        calls, reason = [c.delta.tool_calls for c in choices if c.delta.tool_calls], choices[-1]
        reason = reason.finish_reason
    else:
        calls, reason = response.choices[0].message.tool_calls, response.choices[0].finish_reason
    assert not calls and reason == "length"


@pytest.mark.parametrize("stream", [False, True])
def test_cut_off_after_a_call(client, replies_with, stream):
    # a reply cut off after a whole call, in another it had begun, ends for its length: the
    # client learns that calls may be missing, rather than running the first alone
    whole = '<tool_call>\n{"name": "weather", "arguments": {"city": "Paris"}}\n</tool_call>\n'
    replies_with(whole + '<tool_call>\n{"name": "weather", "argu', cut_off=True)
    response = chat(client, "Weather in Paris and Rome?", tools=[WEATHER], stream=stream)
    if stream:
        choices = [chunk.choices[0] for chunk in response]
        (call,), reason = choices[-2].delta.tool_calls, choices[-1].finish_reason
        content = "".join(c.delta.content or "" for c in choices)
    else:
        (call,), reason = response.choices[0].message.tool_calls, response.choices[0].finish_reason
        content = response.choices[0].message.content or ""
    assert content == "" and reason == "length"
    assert json.loads(call.function.arguments) == {"city": "Paris"}


def test_reasons():
    # whether a request reasons, for its sampling, as its template is told
    chat = SimpleNamespace(default_effort="medium")
    assert _reasons(chat, {}) and not _reasons(chat, {"reasoning_effort": "none"})
    assert not _reasons(SimpleNamespace(default_effort="none"), {})
    assert _reasons(SimpleNamespace(default_effort="none"), {"reasoning_effort": "low"})
    assert not _reasons(chat, {"enable_thinking": False})


def test_cut_off_call():
    # a call the context cut off is no text: the reply's text ends where it began
    reply = Reply(content="Here it is.\n<tool_call>\n<function=weather>\n<parameter=city>\nPar")
    assert _calls(reply, [WEATHER], "length") == ("Here it is.", [])
    assert _calls(reply, [WEATHER], "stop")[0] == reply.content


@pytest.mark.usefixtures("reference_ops")
def test_embeddings(tiny_model, tiny):
    # an embedding model beside the model: each text's embedding, as OpenAI's API gives them, of
    # its tokens and the end-of-text token its tokenizer adds; texts refused that are none
    embedder, _ = tiny("qwen3")
    with serving(tiny_model[0], max_context=CONTEXT, embed=embedder) as server:
        client = connect(server)
        said = client.embeddings.create(model="any", input=["the rent", "a deposit"])
        engine = Engine(embedder, max_context=CONTEXT, prefill_chunk=8)
        tokens = engine.tokenizer.encode("the rent")
        if GGUF.open(embedder).metadata.get("tokenizer.ggml.add_eos_token"):
            tokens.append(engine.tokenizer.eos_id)
        assert [d.index for d in said.data] == [0, 1] and said.model == embedder.stem
        assert said.data[0].embedding == pytest.approx(engine.embed(tokens), abs=1e-5)
        assert sum(x * x for x in said.data[1].embedding) == pytest.approx(1, abs=1e-4)
        assert said.usage.prompt_tokens == len(tokens) + len(engine.tokenizer.encode("a deposit"))
        for wrong in ([], [""], [1], "x" * 0):
            with pytest.raises(openai.BadRequestError):
                client.embeddings.create(model="any", input=wrong)
    with (
        serving(tiny_model[0], max_context=CONTEXT) as server,
        pytest.raises(openai.NotFoundError, match="--embed"),
    ):
        connect(server).embeddings.create(model="any", input="x")


def test_embeddings_end_as_the_client_hangs_up(tiny_model, tiny, monkeypatch):
    # a long list of texts stops being embedded once its client has gone
    with serving(tiny_model[0], max_context=CONTEXT, embed=tiny("qwen3")[0]) as server:
        while server.embedder is None:
            time.sleep(0.01)
        embedded, embed = [], server.embedder.embed
        monkeypatch.setattr(server.embedder, "embed", lambda tokens: embedded.append(tokens)
                            or time.sleep(0.02) or embed(tokens))  # fmt: skip
        body = json.dumps({"input": ["the rent"] * 200}).encode()
        head = f"POST /v1/embeddings HTTP/1.1\r\nContent-Length: {len(body)}\r\n\r\n"
        with socket.create_connection(("127.0.0.1", server.server_port), timeout=10) as connection:
            connection.sendall(head.encode() + body)
            while not embedded:
                time.sleep(0.01)
        time.sleep(0.6)  # for the handler to find the client gone, and the worker to drop it
        gone = len(embedded)
        time.sleep(0.6)
        assert len(embedded) <= gone + 1 < 100


@pytest.mark.usefixtures("reference_ops")
def test_embeddings_of_an_encoder(tiny_model, tiny):
    # an encoder's embedding, EmbeddingGemma 2's, of a text's start, as many tokens as its context
    # holds but one, run in one chunk, as its tokens see each other
    embedder, weights = tiny("gemma-embedding2")
    text = "the tenant pays the rent by the third working day of each month, in advance"
    with serving(tiny_model[0], max_context=CONTEXT, embed=embedder) as server:
        said = connect(server).embeddings.create(model="any", input=text)
    tokens = Tokenizer(GGUF.open(embedder).metadata).encode(text)
    assert len(tokens) >= CONTEXT
    expected = reference_embedding(weights, tokens[: CONTEXT - 1], "gemma-embedding2")
    assert said.data[0].embedding == pytest.approx(expected, abs=2e-3)


def test_background_waits(monkeypatch):
    # completions someone waits on start before background ones that came before them, each in
    # turn; background ones never take the last slot free; a load lets nothing after it go first
    def completion(background: bool) -> _Completion:
        return _Completion(
            [1], "m", 8, None, None, [], None, False, False, background=background)  # fmt: skip

    engine = SimpleNamespace(active=[], slots=3)
    started: list = []

    def start(request, running):
        if isinstance(request, _Load):
            return not engine.active
        if len(engine.active) == engine.slots:
            return False
        engine.active.append(request)
        started.append(request)
        return True

    fake = SimpleNamespace(loaded=SimpleNamespace(engine=engine), _start=start)
    fake._spare = lambda: Server._spare(fake)
    readers = [completion(True) for _ in range(4)]
    chat, other = completion(False), completion(False)
    waiting = collections.deque([*readers, chat])
    Server._admit(fake, waiting, {})
    assert started == [chat, *readers[:1]]  # the chat first; one reader, the last slot kept
    assert list(waiting) == readers[1:]
    waiting.append(other)
    Server._admit(fake, waiting, {})
    assert started[-1] is other and len(engine.active) == 3  # the slot kept, the chat's
    engine.active.clear()
    load = _Load("n")
    waiting = collections.deque([readers[1], load, chat])
    started.clear()
    Server._admit(fake, waiting, {})
    assert started == [readers[1]] and list(waiting) == [load, chat]  # nothing past the load
    one = SimpleNamespace(active=[], slots=1)  # of one slot, nothing kept
    fake.loaded = SimpleNamespace(engine=one)
    engine = one
    waiting = collections.deque([readers[2]])
    Server._admit(fake, waiting, {})
    assert not waiting
