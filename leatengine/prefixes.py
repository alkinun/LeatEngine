"""Prefixes of the KV cache kept beyond its slots, in host memory: what a slot held when it was
taken for another prompt, so that a later prompt that starts as it did takes it back rather than
prefilling it again, as agents' sessions do, more of them than there are slots."""

from dataclasses import dataclass
from typing import Any


@dataclass(eq=False)
class Prefix:
    """What a slot held, as Engine keeps it of a slot: its tokens, those before its kept recurrent
    state, whose they are, and how far its runs wrote its rings; and its data, `tiles` of its
    positions' keys and values and the `rest` of what it held, of `nbytes` bytes in all."""

    owner: str
    cached: list[int]
    kept: list[int]
    written: int
    tiles: list[Any]
    rest: Any
    nbytes: int


class PrefixStore:
    """Prefixes within `limit` bytes, the least recently used let go first to make room."""

    def __init__(self, limit: int):
        self.limit, self.used = limit, 0
        self._prefixes: list[Prefix] = []  # the latest used last

    def __len__(self) -> int:
        return len(self._prefixes)

    def of(self, owner: str) -> list[Prefix]:
        """An owner's prefixes, the latest used first."""
        return [p for p in reversed(self._prefixes) if p.owner == owner]

    def holds(self, owner: str, cached: list[int]) -> bool:
        """Whether an owner's prefix holds all of these tokens, and maybe more."""
        return any(p.cached[: len(cached)] == cached for p in self.of(owner))

    def add(self, prefix: Prefix) -> None:
        """Keeps a prefix, rather than those of its owner whose tokens it holds all of, as the
        latest used; and lets go of the least recently used, as many as it needs the room of. A
        prefix larger than the limit is not kept."""
        if prefix.nbytes > self.limit:
            return
        held = [p for p in self.of(prefix.owner) if prefix.cached[: len(p.cached)] == p.cached]
        for p in held:
            self._drop(p)
        while self.used + prefix.nbytes > self.limit:
            self._drop(self._prefixes[0])
        self._prefixes.append(prefix)
        self.used += prefix.nbytes

    def take(self, prefix: Prefix) -> None:
        """Marks a prefix used now, last to be let go."""
        self._prefixes.remove(prefix)
        self._prefixes.append(prefix)

    def clear(self) -> None:
        self._prefixes, self.used = [], 0

    def _drop(self, prefix: Prefix) -> None:
        self._prefixes.remove(prefix)
        self.used -= prefix.nbytes
