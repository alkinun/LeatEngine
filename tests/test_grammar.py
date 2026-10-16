import json
import random
import re

import pytest

from leatengine import grammar
from leatengine.grammar import STACKS, Constraint, Grammar, Matcher, Tokens

# a schema of the kinds OpenAI's strict structured outputs take, as its SDK writes a Pydantic
# model's: optional fields as anyOf with null, an enum, nested objects by $ref, and arrays
PERSON = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "title": "Name"},
        "age": {"type": "integer"},
        "height": {"type": "number"},
        "role": {"enum": ["admin", "user"]},
        "nick": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        "pets": {"type": "array", "items": {"$ref": "#/$defs/pet"}, "maxItems": 2},
    },
    "required": ["name", "age", "height", "role", "nick", "pets"],
    "additionalProperties": False,
    "$defs": {
        "pet": {
            "type": "object",
            "properties": {"kind": {"type": "string"}, "legs": {"type": ["integer", "null"]}},
            "required": ["kind", "legs"],
            "additionalProperties": False,
        }
    },
}


def takes(matcher: Matcher, text: str) -> bool:
    # whether the matcher takes the whole reply, its JSON whole at its end
    state = matcher.feed(matcher.start, text.encode())
    return bool(state) and matcher.done(state)


@pytest.mark.parametrize("text", [
    '{}', '{"a": 1}', '{"a":-0.5e+10,"b":[true,false,null,"x\\"y\\u00e9"]}', ' {"k": {}}',
    '{\n  "a": [\n    1,\n    {}\n  ]\n}', '{"é": "ü😀"}',
])  # fmt: skip
def test_json(text):
    assert takes(Matcher(Grammar(None)), text)


@pytest.mark.parametrize("text", [
    '[]', '"x"', '{"a": 01}', '{"a" 1}', '{"a": 1,}', '{"a": tru}', '{"a": "x\ny"}',
    '{"a": 1.}', '{"a": .5}', '{"a": "\\x"}', '{"a": "\\u12g4"}', "{'a': 1}", '{"a": 1}}',
    '{  "a": 1}', '{\n' + ' ' * 21 + '"a": 1}', '{"a": ' + '9' * 17 + '}', '{"a": 1}\n',
])  # fmt: skip
def test_not_json(text):
    # neither JSON, nor whitespace as llama.cpp's JSON grammar takes it, nor a number's digits
    # past DIGITS, nor anything after the JSON
    assert not takes(Matcher(Grammar(None)), text)


def test_json_as_python_writes_it():
    # random values, as json.dumps writes them compact or indented: each taken whole, and each cut
    # short at a random byte not, but where what remains is an object too
    rng = random.Random(0)

    def value(depth: int):
        kinds = ["string", "int", "float", "bool", "null"] + (["object", "array"] if depth else [])
        kind = rng.choice(kinds)
        if kind == "object":
            return {f"k{rng.randrange(99)}{chr(rng.randrange(32, 900))}": value(depth - 1)
                    for _ in range(rng.randrange(4))}  # fmt: skip
        if kind == "array":
            return [value(depth - 1) for _ in range(rng.randrange(4))]
        return {"string": lambda: "".join(chr(rng.randrange(1, 300)) for _ in range(5)),
                "int": lambda: rng.randrange(-10**9, 10**9),
                "float": lambda: rng.uniform(-1e5, 1e5), "bool": lambda: rng.random() < 0.5,
                "null": lambda: None}[kind]()  # fmt: skip

    matcher = Matcher(Grammar(None))
    for i in range(200):
        obj = {f"key{i}": value(3)}
        text = json.dumps(obj, ensure_ascii=i % 2 == 0, indent=2 if i % 3 == 0 else None)
        assert takes(matcher, text), text
        cut = text[: rng.randrange(1, len(text))]
        assert not takes(matcher, cut), cut


@pytest.mark.parametrize("text", [
    '{"name": "Ada", "age": 36, "height": 1.65, "role": "admin", "nick": null, "pets": []}',
    '{"name":"","age":-1,"height":2,"role":"user","nick":"a","pets":[{"kind":"cat","legs":4},'
    '{"kind":"fish","legs":null}]}',
])  # fmt: skip
def test_schema(text):
    assert takes(Matcher(Grammar(PERSON)), text)


@pytest.mark.parametrize("text", [
    # properties out of the schema's order, one missing, one more, of another type, an enum's
    # value it does not list, an integer with a fraction, and more items than maxItems
    '{"age": 36, "name": "Ada", "height": 1, "role": "admin", "nick": null, "pets": []}',
    '{"name": "Ada", "age": 36, "height": 1, "role": "admin", "nick": null}',
    '{"name": "Ada", "age": 36, "height": 1, "role": "admin", "nick": null, "pets": [], "x": 1}',
    '{"name": 1, "age": 36, "height": 1, "role": "admin", "nick": null, "pets": []}',
    '{"name": "Ada", "age": 36, "height": 1, "role": "root", "nick": null, "pets": []}',
    '{"name": "Ada", "age": 36.5, "height": 1, "role": "admin", "nick": null, "pets": []}',
    '{"name": "Ada", "age": 3, "height": 1, "role": "user", "nick": null, "pets": '
    '[{"kind": "a", "legs": 1}, {"kind": "b", "legs": 2}, {"kind": "c", "legs": 3}]}',
])  # fmt: skip
def test_not_schema(text):
    assert json.loads(text) and not takes(Matcher(Grammar(PERSON)), text)


def test_optional_and_more_properties():
    # optional properties may be left out, in order still, and no others, which an object of its
    # properties alone needs not; an object of none takes others, as additionalProperties allows
    properties = {"a": {"type": "integer"}, "b": {"type": "string"}, "c": {"type": "boolean"}}
    schema = {"type": "object", "properties": properties, "required": ["c"],
              "additionalProperties": True}  # fmt: skip
    matcher = Matcher(Grammar(schema))
    for text in ('{"c": true}', '{"a": 1, "c": false}', '{"a": 1, "b": "x", "c": true}'):
        assert takes(matcher, text), text
    for text in ('{"a": 1}', '{"c": true, "z": 1}', '{"b": "x", "a": 1, "c": true}'):
        assert not takes(matcher, text), text
    counts = Matcher(Grammar({"type": "object", "additionalProperties": {"type": "integer"}}))
    assert takes(counts, '{"x": 1, "y": 2}') and not takes(counts, '{"x": "1"}')
    assert takes(Matcher(Grammar({"type": "object"})), '{"x": [1, {"y": null}]}')
    assert not takes(
        Matcher(Grammar({"type": "object", "additionalProperties": False})), '{"x": 1}'
    )


def test_enum_of_its_type():
    # an enum or const beside its type, as Pydantic writes a Literal's, of values of that type
    matcher = Matcher(Grammar({"type": "string", "enum": ["a", "b"]}))
    assert takes(matcher, '"a"') and not takes(matcher, '"c"')
    assert takes(Matcher(Grammar({"type": ["integer", "null"], "const": None})), "null")


def test_values_at_the_root():
    # a schema of any value at its root, of a list of types, an array's least items, and a
    # recursive one
    assert takes(Matcher(Grammar({"type": ["number", "null"]})), "-1.5e3")
    assert takes(Matcher(Grammar({"const": [1, {"a": True}]})), '[1, {"a": true}]')
    least = Matcher(Grammar({"type": "array", "items": {"type": "integer"}, "minItems": 2}))
    assert takes(least, "[1, 2]") and not takes(least, "[1]")
    tree = {"type": "object", "properties": {"kids": {"type": "array", "items": {"$ref": "#"}}},
            "required": ["kids"], "additionalProperties": False}  # fmt: skip
    assert takes(Matcher(Grammar(tree)), '{"kids": [{"kids": []}, {"kids": [{"kids": []}]}]}')


DEEP: dict = {"type": "null"}
for _ in range(64):
    DEEP = {"type": "array", "items": DEEP}


@pytest.mark.parametrize("schema, said", [
    ({"type": "string", "pattern": "a+"}, "the schema's pattern is not supported"),
    ({"type": "string", "format": "date"}, "the schema's format is not supported"),
    ({"type": "integer", "minimum": 0}, "the schema's minimum is not supported"),
    ({"allOf": [{"type": "string"}]}, "the schema's allOf is not supported"),
    ({"type": "string", "items": {}}, "the schema's items does not go with type string"),
    ({"type": "object", "required": ["a"]}, "the schema requires a, which it does not define"),
    ({"$ref": "#/$defs/x"}, "the schema's #/$defs/x refers to no definition"),
    ({"$ref": "other.json"}, "the schema's $ref other.json is not supported"),
    ({"type": "text"}, "a schema's type is one of object"),
    ({"enum": []}, "a schema's enum lists its values"),
    ({"$ref": "#"}, "the schema refers to itself before a value of its own"),
    ({"$ref": 1}, "a schema's $ref is a string"),
    ({"type": "array", "minItems": 2, "maxItems": 1}, "minItems and maxItems are whole numbers"),
    ({"type": "array", "minItems": "1"}, "minItems and maxItems are whole numbers"),
    ({"type": "object", "required": "a"}, "a schema's required lists the names"),
    ({"type": "object", "properties": []}, "a schema's properties are an object of schemas"),
    (DEEP, "the schema nests deeper than 64 levels"),
    ({"type": []}, "a schema's list of types names one at least"),
    ({"$ref": "#/$defs/a", "maxItems": 1, "$defs": {"a": {"type": "array"}}},
     "a schema of $ref takes no maxItems beside it"),
    ({"type": "integer", "enum": [1, "x"]}, 'the schema\'s value "x" is not of its type'),
    ({"enum": [1], "const": 1}, "a schema gives const or enum, not both"),
    ({"enum": [1], "items": {}}, "a schema of enum takes no items beside it"),
    ({"type": "object", "properties": {"a": {"$ref": "#"}}, "required": ["a"]},
     "the schema has a value that never ends"),
    ({"type": "array", "items": {"$ref": "#"}, "minItems": 1},
     "the schema has a value that never ends"),
    ({"$defs": [], "$ref": "#/$defs/a"}, "a schema's $defs is an object of schemas"),
    ({"$defs": {"a": {"anyOf": [{"type": "null"}, {"$ref": "#/$defs/b"}]},
                "b": {"$ref": "#/$defs/a"}}, "$ref": "#/$defs/a"},
     "the schema refers to itself before a value of its own"),
])  # fmt: skip
def test_refused(schema, said):
    # a keyword that would hold more of a value than the grammar holds is refused, not ignored
    with pytest.raises(ValueError, match=re.escape(said)):
        Grammar(schema)


def test_reasoning_first():
    # a block of reasoning, opened by its marker, before the JSON, or none; or text up to the
    # marker that opens the answer, as harmony's
    think = Matcher(Grammar(None), ("<think>", "</think>"))
    assert takes(think, '<think>{"not": "this"}</think>\n\n{"a": 1}') and takes(think, '{"a": 1}')
    assert not takes(think, 'Sure: {"a": 1}') and not takes(think, '<think>{"a": 1}')
    harmony = Matcher(Grammar(None), "<|channel|>final<|message|>")
    reply = "<|channel|>analysis<|message|>hm<|end|><|start|>assistant<|channel|>final<|message|>"
    assert takes(harmony, reply + '{"a": 2}') and not takes(harmony, '{"a": 2}')


# a vocab of every single byte of JSON's and some longer pieces, a format's markers and an end
VOCAB = [bytes([b]) for b in b'{}[]:," \n\tabcdeflnrstu0123456789-.+eE\\<>/'] + [
    b'{"', b'":', b'", "', b'"}', b"true", b"null", b"12", b"0.5", b'"a', b"<think>",
    b"</think>", b"<tool_call>", b"ab\\n", b'x"}', b"\n  ", b" " * 3,
]  # fmt: skip
MARKUP, END = {len(VOCAB) - 7, len(VOCAB) - 6, len(VOCAB) - 5}, len(VOCAB)


@pytest.mark.parametrize("preamble", [None, ("<think>", "</think>")])
def test_masks(preamble):
    # at every state of replies cut short, a token is allowed exactly where its bytes go on from
    # the state, the end exactly where the JSON is whole; markup only where reasoning takes it
    matcher, tokens = Matcher(Grammar(PERSON), preamble), Tokens([*VOCAB, b""], MARKUP, [END])
    replies = ['{"name": "a\\nb", "age": 12, "height": 0.5, "role": "user", "nick": null, '
               '"pets": [{"kind": "x", "legs": null}]}']  # fmt: skip
    if preamble:
        replies.append('<think>{"a"} ok</think>\n{"name": "", "age": -1, "height": 1e5, '
                       '"role": "admin", "nick": "ab", "pets": []}')  # fmt: skip
    for reply in replies:
        for cut in range(len(reply) + 1):
            state = matcher.feed(matcher.start, reply[:cut].encode())
            mask = tokens.allowed(matcher, state)
            if mask is None:  # inside the reasoning
                assert any(frame[0] == "until" for stack in state for frame in stack)
                continue
            pending = any(frame[0] == "until" for stack in state for frame in stack)
            for i, token in enumerate(VOCAB):
                goes = bool(matcher.feed(state, token)) and (i not in MARKUP or pending)
                assert bool(mask >> i & 1) == goes, (reply[:cut], token)
            assert bool(mask >> END & 1) == matcher.done(state)


def test_constraint():
    # a sequence's state as it takes tokens, its masks cached by state and shared, a string's the
    # same as it goes on; an end token takes nothing
    matcher, tokens = Matcher(Grammar(None)), Tokens([*VOCAB, b""], set(), [END])
    cache: dict = {}
    first, second = Constraint(matcher, tokens, cache), Constraint(matcher, tokens, cache)
    for c in (first, second):
        for token in (b'{"', b"a", b'":', b" ", b"1", b"}"):
            assert c.mask() >> VOCAB.index(token) & 1
            c.advance(VOCAB.index(token), token)
        assert c.mask() >> END & 1
        c.advance(END, b"")
        assert matcher.done(c.state)
    assert first.state == second.state and len(cache) == 6


def test_states_bounded():
    # of a schema of alike options nested, the ways a reply may go on would double with each
    # array opened: STACKS at most are kept, the reply still matched
    twice = {"anyOf": [{"type": "array", "items": {"$ref": "#/$defs/a"}}] * 2}
    matcher = Matcher(Grammar({"$defs": {"a": twice}, "$ref": "#/$defs/a"}))
    state = matcher.feed(matcher.start, b"[" * 12)
    assert 0 < len(state) <= STACKS and takes(matcher, "[" * 12 + "]" * 12)
    # an array's items counted as far as minItems and maxItems ask, so that one's states repeat
    plain = Matcher(Grammar(None))
    assert plain.feed(plain.start, b'{"a": [1, ') == plain.feed(plain.start, b'{"a": [1, 1, 1, ')
    least = Matcher(Grammar({"type": "array", "items": {"type": "integer"}, "minItems": 2}))
    assert least.feed(least.start, b"[1, ") != least.feed(least.start, b"[1, 1, ")
    assert least.feed(least.start, b"[1, 1, ") == least.feed(least.start, b"[1, 1, 1, ")


def test_masks_bounded(monkeypatch):
    # a grammar's masks, of MASKS states at most, the last asked for kept
    monkeypatch.setattr(grammar, "MASKS", 3)
    matcher, tokens = Matcher(Grammar(None)), Tokens([*VOCAB, b""], set(), [END])
    cache: dict = {}
    c = Constraint(matcher, tokens, cache)
    for token in (b'{"', b"a", b'":', b" ", b"1"):
        c.mask()
        c.advance(VOCAB.index(token), token)
    c.mask()
    assert len(cache) == 3 and list(cache)[-1] == c.state
