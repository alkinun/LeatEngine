"""Constrained output: a reply that is JSON, as a schema has it, and the tokens it allows next.

A schema compiles to a Grammar, a table of nodes, each what a JSON value may be: OpenAI's strict
structured outputs' subset, objects whose properties come in the schema's order, required or not,
and of no properties, others where additionalProperties allows them; arrays, of as many items as
minItems and maxItems allow; strings, numbers, integers, booleans and null; enum and const values;
anyOf and lists of types; and $ref to $defs, recursive too. A schema's keyword that would hold
more than this holds is refused rather than ignored, so that what is generated always matches.

A Matcher takes a reply a byte at a time. Its state is a set of stacks of frames, each stack one
way the bytes so far may go on: of a value, an object, an array, a string, a number, or a choice
of literals, as of keys and enum values. Whitespace goes where llama.cpp's JSON grammar has it:
none, a space, or a newline and up to 20 spaces or tabs, so that a reply cannot run on in it.
Before the JSON, a reply may reason, in a block of its format's markers, as Qwen3's <think>, or
until the marker that opens its answer, as harmony's final channel.

Tokens says which tokens a state allows: of each first byte the state takes, those whose bytes it
takes, walking the vocab's tokens sorted by their bytes; inside a string, the tokens of string bytes
alone at once, as they leave the state as it is, and those of a quote or backslash one by one. A
Constraint is a sequence's: its state, and its mask of the tokens allowed next, cached by state.
"""

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

# whitespace between a value's parts: none, a space, or a newline and up to INDENT spaces or tabs
INDENT = 20
DIGITS = 16  # most digits of a number's integer part, and of its fraction
# the keywords a schema may hold that hold nothing more of a value, which are ignored
_NOTES = {"title", "description", "examples", "default", "$schema", "$id", "$comment",
          "deprecated", "readOnly", "writeOnly", "$defs", "definitions"}  # fmt: skip
_KEYWORDS = _NOTES | {"type", "properties", "required", "additionalProperties", "items", "enum",
                      "const", "anyOf", "oneOf", "$ref", "minItems", "maxItems"}  # fmt: skip
_TYPES = ("object", "array", "string", "number", "integer", "boolean", "null")
_SPACE = b" \t"
_HEX = frozenset(b"0123456789abcdefABCDEF")
_ESCAPED = frozenset(b'"\\/bfnrt')
_DIGIT = frozenset(b"0123456789")
# a number's phases, as JSON's grammar: before its digits, after a minus, after a leading 0, in
# its integer part, before its fraction's digits, in them, before its exponent's sign or digits,
# after its sign, and in its digits; those it may end in
_START, _MINUS, _ZERO, _INT, _POINT, _FRACTION, _E, _SIGN, _EXPONENT = range(9)
_ENDS = {_ZERO, _INT, _FRACTION, _EXPONENT}


@dataclass(frozen=True)
class Node:
    """What a JSON value may be: of `kind` object (`properties`, each (key, node, required), and
    `extra`, the node of others, if any), array (`items`, and its `least` and `most`), string,
    number (`integer` or not), literals (`choices`, the JSON of each), any (any value), or anyOf
    (`options`). Nodes are a Grammar's by index."""

    kind: str
    properties: tuple[tuple[str, int, bool], ...] = ()
    extra: int | None = None
    items: int = 0
    least: int = 0
    most: int | None = None
    integer: bool = False
    choices: tuple[bytes, ...] = ()
    options: tuple[int, ...] = ()


class Grammar:
    """A schema compiled: its nodes, the first its root."""

    def __init__(self, schema: dict[str, Any] | None):
        # of no schema, any object, as OpenAI's json_object has a reply be
        self.nodes: list[Node] = []
        self._refs: dict[str, int] = {}
        self._root = schema
        if schema is None:
            self._add(Node("object"))
            self.nodes[0] = Node("object", extra=self._any())
        else:
            self._compile(schema)

    def _add(self, node: Node) -> int:
        self.nodes.append(node)
        return len(self.nodes) - 1

    def _any(self) -> int:
        # any value: an object or array of any values, or a string, number, boolean or null
        at = self._add(Node("any"))
        options = (
            self._add(Node("object", extra=at)), self._add(Node("array", items=at)),
            self._add(Node("string")), self._add(Node("number")),
            self._add(Node("literals", choices=(b"true", b"false", b"null"))),
        )  # fmt: skip
        self.nodes[at] = Node("anyOf", options=options)
        return at

    def _compile(self, schema: Any) -> int:
        if not isinstance(schema, dict):
            raise ValueError(f"a schema is an object, not {json.dumps(schema)}")
        if unknown := sorted(set(schema) - _KEYWORDS):
            raise ValueError(f"the schema's {unknown[0]} is not supported")
        if "$ref" in schema:
            return self._ref(schema["$ref"])
        if "const" in schema or "enum" in schema:
            values = [schema["const"]] if "const" in schema else schema["enum"]
            if not isinstance(values, list) or not values:
                raise ValueError("a schema's enum lists its values")
            choices = tuple(json.dumps(v, ensure_ascii=False).encode() for v in values)
            return self._add(Node("literals", choices=tuple(dict.fromkeys(choices))))
        alternatives = schema.get("anyOf", schema.get("oneOf"))
        if alternatives is not None:
            if not isinstance(alternatives, list) or not alternatives:
                raise ValueError("a schema's anyOf lists its schemas")
            rest = {k: v for k, v in schema.items() if k not in ("anyOf", "oneOf")}
            if set(rest) - _NOTES:
                raise ValueError("a schema of anyOf takes no other keywords but notes")
            at = self._add(Node("anyOf"))
            self.nodes[at] = Node("anyOf", options=tuple(self._compile(s) for s in alternatives))
            return at
        kinds = schema.get("type")
        if kinds is None:
            return self._any()
        if isinstance(kinds, list):
            at = self._add(Node("anyOf"))
            options = tuple(self._compile(schema | {"type": kind}) for kind in kinds)
            self.nodes[at] = Node("anyOf", options=options)
            return at
        if kinds not in _TYPES:
            raise ValueError(f"a schema's type is one of {', '.join(_TYPES)}, not {kinds!r}")
        held = set(schema) - _NOTES - {"type"}
        allowed = {"object": {"properties", "required", "additionalProperties"},
                   "array": {"items", "minItems", "maxItems"}}.get(kinds, set())  # fmt: skip
        if held - allowed:
            raise ValueError(f"the schema's {sorted(held - allowed)[0]} does not go with type "
                             f"{kinds}")  # fmt: skip
        if kinds == "object":
            return self._object(schema)
        if kinds == "array":
            at = self._add(Node("array"))
            items = self._compile(schema["items"]) if "items" in schema else self._any()
            least, most = schema.get("minItems", 0), schema.get("maxItems")
            self.nodes[at] = Node("array", items=items, least=least, most=most)
            return at
        if kinds in ("boolean", "null"):
            choices = (b"true", b"false") if kinds == "boolean" else (b"null",)
            return self._add(Node("literals", choices=choices))
        return self._add(Node(kinds if kinds == "string" else "number", integer=kinds == "integer"))

    def _object(self, schema: dict[str, Any]) -> int:
        at = self._add(Node("object"))
        required = set(schema.get("required", []))
        properties = tuple(
            (key, self._compile(value), key in required)
            for key, value in schema.get("properties", {}).items()
        )
        if unknown := required - {key for key, _, _ in properties}:
            raise ValueError(f"the schema requires {sorted(unknown)[0]}, which it does not define")
        # others, where additionalProperties allows them, only of an object that defines none: an
        # object of its defined properties alone matches the schema either way, and an object of
        # no properties takes any it allows, by default any
        more = schema.get("additionalProperties", not properties)
        if properties or more is False:
            extra = None
        else:
            extra = self._any() if more is True else self._compile(more)
        self.nodes[at] = Node("object", properties=properties, extra=extra)
        return at

    def _ref(self, ref: str) -> int:
        if ref in self._refs:
            return self._refs[ref]
        if ref == "#":
            schema: Any = self._root
        elif m := re.fullmatch(r"#/(\$defs|definitions)/([^/]+)", ref):
            schema = (self._root or {}).get(m.group(1), {}).get(m.group(2))
            if schema is None:
                raise ValueError(f"the schema's {ref} refers to no definition")
        else:
            raise ValueError(f"the schema's $ref {ref} is not supported: refer to #/$defs/NAME")
        # a placeholder first, filled in after, as the definition may refer to itself
        self._refs[ref] = at = self._add(Node("anyOf"))
        self.nodes[at] = Node("anyOf", options=(self._compile(schema),))
        return at


Frame = tuple[Any, ...]
Stack = tuple[Frame, ...]
State = frozenset[Stack]


class Matcher:
    """A grammar's reply, a byte at a time. A frame is one of:

    ("value", node, ws): a value of a node is next, after ws, the whitespace so far: 0 for none,
        1 a space, 2 + n a newline and n spaces or tabs, and before the reply's JSON -1 - n, n
        whitespace bytes of any kind, as a blank line after reasoning;
    ("object", node, next, phase, ws, child): inside an object, the index of the next property
        it may hold, and its phase: "first" after "{", a key or "}"; "key" after ",", a key; ":"
        after a key, its colon, the value then of node `child`; "value" while the value runs;
        "after" after it, "," or "}";
    ("array", node, phase, ws, count): inside an array: "first" after "[", an item or "]"; "item"
        after ","; "value" while an item runs; "after" after it, "," or "]";
    ("string", escape, then): inside a string: escape 0 in its text, -1 after a backslash, n of
        \\uXXXX's hex digits still to come; then the frame its end goes on with, if not the end
        of a value, as of a key;
    ("number", integer, phase, digits): inside a number;
    ("literals", choices): of choices, each (bytes still to come, the frame its end goes on with
        or None);
    ("until", marker, matched): text up to and through a marker, as reasoning before the JSON;
    ("done", ws): the reply's JSON whole, and whitespace after it.
    """

    def __init__(self, grammar: Grammar, preamble: tuple[str, str] | str | None = None):
        # preamble: an optional block of reasoning, (opens, closes), before the JSON; or text up
        # to the marker that opens the answer; or none
        self.grammar = grammar
        root: Stack = (("value", 0, -1),)
        if isinstance(preamble, str):
            self.start: State = frozenset({(*root, ("until", preamble.encode(), 0))})
        elif preamble is not None:
            opens, closes = (m.encode() for m in preamble)
            block = (*root, ("until", closes, 0), ("literals", ((opens, None),)))
            self.start = frozenset({root, block})
        else:
            self.start = frozenset({root})

    def feed(self, state: State, data: bytes) -> State:
        """The state after bytes, empty where they cannot go on."""
        for byte in data:
            state = frozenset(s for stack in state for s in self._step(stack, byte))
            if not state:
                break
        return state

    def done(self, state: State) -> bool:
        """Whether the reply may end here: its JSON whole, as a number may be at its end."""
        return any(
            stack[-1][0] == "done"
            or (len(stack) == 1 and stack[0][0] == "number" and stack[0][2] in _ENDS)
            for stack in state
        )

    def _step(self, stack: Stack, b: int) -> list[Stack]:
        top, rest = stack[-1], stack[:-1]
        kind = top[0]
        if kind == "string":
            return self._string(top, rest, b)
        if kind == "value":
            if (ws := _spaced(top[2], b)) is not None:
                return [(*rest, ("value", top[1], ws))]
            return self._value(top[1], rest, b)
        if kind == "object":
            return self._object(top, rest, b)
        if kind == "array":
            return self._array(top, rest, b)
        if kind == "number":
            return self._number(top, rest, b)
        if kind == "literals":
            return self._literals(top, rest, b)
        if kind == "until":
            return self._until(top, rest, b)
        if (ws := _spaced(top[1], b)) is not None:  # done: whitespace after the JSON
            return [(("done", ws),)]
        return []

    def _value(self, at: int, rest: Stack, b: int) -> list[Stack]:
        # the first byte of a value of node `at`
        node = self.grammar.nodes[at]
        if node.kind == "anyOf":
            return [s for option in node.options for s in self._value(option, rest, b)]
        if node.kind == "object" and b == ord("{"):
            return [(*rest, ("object", at, 0, "first", 0, None))]
        if node.kind == "array" and b == ord("["):
            return [(*rest, ("array", at, "first", 0, 0))]
        if node.kind == "string" and b == ord('"'):
            return [(*rest, ("string", 0, None))]
        if node.kind == "number":
            return self._number(("number", node.integer, _START, 0), rest, b)
        if node.kind == "literals":
            choices = tuple((c[1:], None) for c in node.choices if c[0] == b)
            return self._finished(rest, choices)
        return []

    def _finished(
        self, rest: Stack, choices: tuple[tuple[bytes, Frame | None], ...]
    ) -> list[Stack]:
        # literals with these choices left: those ended go on, the others wait for more
        out = [s for left, then in choices if not left for s in self._ended(rest, then)]
        if waiting := tuple(c for c in choices if c[0]):
            out.append((*rest, ("literals", waiting)))
        return out

    def _ended(self, rest: Stack, then: Frame | None) -> list[Stack]:
        # a value, or a key, has ended: `then` goes on where given, else the frame below
        if then is not None:
            return [(*rest, then)]
        if not rest:
            return [(("done", 0),)]
        parent, below = rest[-1], rest[:-1]
        if parent[0] == "object":
            return [(*below, ("object", parent[1], parent[2], "after", 0, None))]
        if parent[0] == "array":
            return [(*below, ("array", parent[1], "after", 0, parent[4] + 1))]
        # the end of reasoning's block, or of the text before the answer: the JSON's value next
        return [rest]

    def _object(self, top: Frame, rest: Stack, b: int) -> list[Stack]:
        _, at, next_, phase, ws, child = top
        if phase == "value":  # the value's frame is above: nothing reaches here
            return []
        if (spaced := _spaced(ws, b)) is not None:
            return [(*rest, ("object", at, next_, phase, spaced, child))]
        node = self.grammar.nodes[at]
        properties = node.properties
        # the keys it may take next: the defined from `next` on, up to the first required one
        keys = []
        for i in range(next_, len(properties)):
            keys.append(i)
            if properties[i][2]:
                break
        if b == ord("}") and phase in ("first", "after"):
            closes = not any(required for _, _, required in properties[next_:])
            return self._ended(rest, None) if closes else []
        if phase == "after":
            if b != ord(",") or not keys and node.extra is None:
                return []
            return [(*rest, ("object", at, next_, "key", 0, None))]
        if phase == ":":
            if b != ord(":"):
                return []
            return [(*rest, ("object", at, next_, "value", 0, None), ("value", child, 0))]
        # a key, opened by its quote: one of those defined it may take next, or any other
        if b != ord('"'):
            return []
        out = []
        if keys:
            choices = tuple(
                (json.dumps(properties[i][0], ensure_ascii=False).encode()[1:],
                 ("object", at, i + 1, ":", 0, properties[i][1]))
                for i in keys
            )  # fmt: skip
            out.append((*rest, ("literals", choices)))
        if node.extra is not None:
            then = ("object", at, next_, ":", 0, node.extra)
            out.append((*rest, ("string", 0, then)))
        return out

    def _array(self, top: Frame, rest: Stack, b: int) -> list[Stack]:
        _, at, phase, ws, count = top
        if phase == "value":
            return []
        if (spaced := _spaced(ws, b)) is not None:
            return [(*rest, ("array", at, phase, spaced, count))]
        node = self.grammar.nodes[at]
        if b == ord("]") and phase in ("first", "after"):
            return self._ended(rest, None) if count >= node.least else []
        if phase == "after":
            more = node.most is None or count < node.most
            return [(*rest, ("array", at, "item", 0, count))] if b == ord(",") and more else []
        if node.most == 0:
            return []
        return self._value(node.items, (*rest, ("array", at, "value", 0, count)), b)

    def _string(self, top: Frame, rest: Stack, b: int) -> list[Stack]:
        _, escape, then = top
        if escape == 0:
            if b == ord('"'):
                return self._ended(rest, then)
            if b == ord("\\"):
                return [(*rest, ("string", -1, then))]
            return [] if b < 0x20 else [(*rest, top)]
        if escape == -1:
            if b == ord("u"):
                return [(*rest, ("string", 4, then))]
            return [(*rest, ("string", 0, then))] if b in _ESCAPED else []
        return [(*rest, ("string", escape - 1 if escape > 1 else 0, then))] if b in _HEX else []

    def _number(self, top: Frame, rest: Stack, b: int) -> list[Stack]:
        _, integer, phase, digits = top
        digit = b in _DIGIT

        def to(next_phase: int, counted: int = 0) -> list[Stack]:
            return [(*rest, ("number", integer, next_phase, counted))]

        if phase in (_START, _MINUS) and b == ord("0"):
            return to(_ZERO)
        if phase in (_START, _MINUS) and digit:
            return to(_INT, 1)
        if phase == _START and b == ord("-"):
            return to(_MINUS)
        if phase in (_INT, _FRACTION) and digit and digits < DIGITS:
            return to(phase, digits + 1)
        if phase in (_ZERO, _INT) and b == ord(".") and not integer:
            return to(_POINT)
        if phase == _POINT and digit:
            return to(_FRACTION, 1)
        if phase in (_ZERO, _INT, _FRACTION) and b in b"eE" and not integer:
            return to(_E)
        if phase == _E and b in b"+-":
            return to(_SIGN)
        if phase in (_E, _SIGN) and digit:
            return to(_EXPONENT, 1)
        if phase == _EXPONENT and digit and digits < 3:
            return to(_EXPONENT, digits + 1)
        if phase in _ENDS:  # the number ended before this byte, which goes on from the frame below
            return [s for stack in self._ended(rest, None) for s in self._step(stack, b)]
        return []

    def _literals(self, top: Frame, rest: Stack, b: int) -> list[Stack]:
        choices = tuple((left[1:], then) for left, then in top[1] if left[0] == b)
        return self._finished(rest, choices)

    def _until(self, top: Frame, rest: Stack, b: int) -> list[Stack]:
        # any text, up to and through the marker: of its bytes, as many as the text ends in
        _, marker, matched = top
        text = marker[:matched] + bytes([b])
        matched = next(n for n in range(min(len(text), len(marker)), -1, -1)
                       if text.endswith(marker[:n]))  # fmt: skip
        if matched == len(marker):
            return self._ended(rest, None)
        return [(*rest, ("until", marker, matched))]


def _spaced(ws: int, b: int) -> int | None:
    # whitespace after `ws` so far, if b goes on with it: a space or a newline first, then after a
    # newline up to INDENT spaces or tabs; before the reply's JSON, up to INDENT of any
    if ws < 0:
        return ws - 1 if b in b" \t\n\r" and -ws <= INDENT else None
    if ws == 0:
        return 1 if b == ord(" ") else 2 if b == ord("\n") else None
    if ws >= 2 and b in _SPACE and ws - 2 < INDENT:
        return ws + 1
    return None


class Tokens:
    """A vocab's tokens as a grammar takes them: their bytes, sorted; those of a format's markup,
    which a reply's reasoning takes, and its markers, but its JSON never; and those that end it."""

    def __init__(self, spelled: list[bytes], markup: Iterable[int], ends: Iterable[int]):
        self.ends, self.count = frozenset(ends), len(spelled)
        self._markup = [(spelled[i], i) for i in sorted(set(markup) - self.ends) if spelled[i]]
        held = {i for _, i in self._markup} | self.ends
        tokens = sorted((b, i) for i, b in enumerate(spelled) if b and i not in held)
        self._sorted = tokens
        # where each first byte's tokens begin and end in the sorted list, and how many bytes
        # each token shares with the one before
        self._spans: dict[int, tuple[int, int]] = {}
        for at, (b, _) in enumerate(tokens):
            first, _ = self._spans.get(b[0], (at, at))
            self._spans[b[0]] = (first, at + 1)
        self._shared = [0] + [
            _common(tokens[i - 1][0], tokens[i][0]) for i in range(1, len(tokens))
        ]
        # the tokens of string bytes alone, which leave a string as it is, as a mask, and those
        # of a quote, backslash or control byte
        self._plain, self._special = 0, list[tuple[bytes, int]]()
        for b, i in tokens:
            if b'"' in b or b"\\" in b or any(c < 0x20 for c in b):
                self._special.append((b, i))
            else:
                self._plain |= 1 << i

    def allowed(self, matcher: Matcher, state: State) -> int | None:
        """The tokens a state allows next, as a mask of a bit per token: those whose bytes it
        takes, and of a state that may end the reply, those that end one. None where any token
        goes, as in reasoning before the JSON."""
        if any(stack[-1][0] == "until" for stack in state):
            return None
        mask = sum(1 << i for i in self.ends) if matcher.done(state) else 0
        if any(frame[0] == "until" for stack in state for frame in stack):  # a marker's opening
            mask |= sum(1 << i for b, i in self._markup if matcher.feed(state, b))
        if all(stack[-1][0] == "string" and stack[-1][1] == 0 for stack in state):
            # string bytes leave the state as it is: those tokens at once
            mask |= self._plain
            for b, i in self._special:
                if matcher.feed(state, b):
                    mask |= 1 << i
            return mask
        for first, (start, end) in self._spans.items():
            if after := matcher.feed(state, bytes([first])):
                mask |= self._walk(matcher, after, start, end)
        return mask

    def _walk(self, matcher: Matcher, state: State, start: int, end: int) -> int:
        # the tokens of one first byte, sorted, from the state after it: each from the state of
        # the bytes it shares with the one before, skipping those past a byte the state refused
        mask, states, at = 0, [state], start  # states[n]: after n + 1 bytes of the token at
        while at < end:
            token, i = self._sorted[at]
            del states[max(self._shared[at], 1) if at > start else 1 :]
            for n in range(len(states), len(token)):
                if not (after := matcher.feed(states[-1], token[n : n + 1])):
                    # every later token that shares these n + 1 bytes is refused too
                    at += 1
                    while at < end and self._shared[at] > n:
                        at += 1
                    break
                states.append(after)
            else:
                mask |= 1 << i
                at += 1
        return mask


def _common(a: bytes, b: bytes) -> int:
    n = 0
    while n < len(a) and n < len(b) and a[n] == b[n]:
        n += 1
    return n


class Constraint:
    """A sequence's constrained reply: its matcher's state, and which tokens it allows next."""

    def __init__(
        self, matcher: Matcher, tokens: Tokens, cache: dict[State, int | None] | None = None
    ):
        self.matcher, self.tokens = matcher, tokens
        self.state = matcher.start
        self._cache = {} if cache is None else cache  # masks by state, which constraints share

    def mask(self) -> int | None:
        """The tokens allowed next, a bit each; None where any token goes."""
        if self.state not in self._cache:
            self._cache[self.state] = self.tokens.allowed(self.matcher, self.state)
        return self._cache[self.state]

    def advance(self, token: int, spelled: bytes) -> None:
        """Takes in a token generated, of these bytes."""
        if token not in self.tokens.ends:
            self.state = self.matcher.feed(self.state, spelled)
