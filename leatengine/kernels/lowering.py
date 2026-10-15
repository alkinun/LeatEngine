"""How tinygrad schedules and lowers kernels, made faster for a model's graphs.

A graph calls the same few kernels layer after layer, and tinygrad canonicalizes each call's body
anew: `canonicalize_once` does each body once. tinygrad also schedules and lowers every kernel at
every start, keeping only the compiled binaries on disk: `keep_programs` and `keep_schedules` keep
the lowered programs and the schedules there too. On the 3090, Llama 3.1 8B's warm-up takes 26 s,
19 with each body canonicalized once, and at a later start 12 with its programs kept and 9 with its
schedules kept too.

tinygrad's GPU queues run host programs, which work out the launches of kernels of symbolic sizes:
`host_renderer` has them rendered as before tinygrad took LLVM's renderer for them.
"""

import functools
import hashlib
import multiprocessing
import os
import pathlib
import pickle
import re
import weakref
import zlib
from collections.abc import Callable
from typing import Any

import tinygrad
from tinygrad import UOp
from tinygrad.helpers import CCACHE, diskcache_get, diskcache_put

from leatengine.kernels import clang

# of tinygrad's disk cache: each row a program's or a schedule's pickle, compressed, and its
# SHA-256 before it
PROGRAMS, SCHEDULES = "leatengine_checked_programs", "leatengine_checked_schedules"
# the environment variables tinygrad's source reads, by name
_READ = re.compile(rb"""(?:getenv|ContextVar)\(\s*["']([A-Z0-9_]+)["']""")

# each kernel body tinygrad has canonicalized, and its canonical form: None where it is its own,
# as a value that is its key would keep the key alive
_CANONICAL: weakref.WeakKeyDictionary[UOp, UOp | None] = weakref.WeakKeyDictionary()


def install() -> None:
    host_renderer()
    canonicalize_once()
    if not multiprocessing.current_process().daemon:  # tinygrad's compile workers lower what
        keep_programs()  # their parent found in neither cache, and schedule nothing
        keep_schedules()


def host_renderer() -> None:
    """Makes tinygrad render the host programs that drive its GPU queues with the CPU's first
    renderer, clang's where clang is installed, as it did until its commit 741ec2853 made it
    LLVM's. Such a program works out a kernel's symbolic sizes as the GPU's renderer lowered them,
    with subtractions and negations, which LLVM's renderer lacks: the kernels of a ring cache
    failed to lower. An HCQ_RUNTIME_DEV of one's own stands."""
    from tinygrad.helpers import HCQ_RUNTIME_DEV

    if "HCQ_RUNTIME_DEV" not in os.environ:
        HCQ_RUNTIME_DEV.value = "CPU"


def canonicalize_once() -> None:
    """Makes tinygrad canonicalize each kernel's body once rather than at each call of it: Llama
    3.1 8B's warm-up canonicalizes 89 bodies 3565 times, 6 of its 26 s. The canonical body, its
    local buffers numbered for the schedule cache, is the body's alone."""
    import tinygrad.schedule as schedule

    canonicalize = schedule.canonicalize_call_body
    if hasattr(canonicalize, "__wrapped__"):  # already
        return

    @functools.wraps(canonicalize)
    def once(call: UOp) -> UOp:
        if (body := call.body) not in _CANONICAL:
            canonical = canonicalize(call).body
            _CANONICAL[body] = None if canonical is body else canonical
        canonical = _CANONICAL[body]
        return call if canonical is None else call.replace(src=(canonical, *call.src[1:]))

    schedule.canonicalize_call_body = once  # type: ignore[assignment]


def keep_programs() -> None:
    """Makes tinygrad keep the programs it lowers in its disk cache, as it keeps their binaries, and
    read them there before it lowers one: Llama 3.1 8B's 160 take 1.2 MB, compressed, and loaded, a
    later start's warm-up takes 12 s rather than 18. A program's row is by the key of tinygrad's own
    cache of them and what else makes a program: the source of tinygrad and of the AMD renderer,
    clang's flags, and the environment variables tinygrad reads, so that a change to any lowers
    anew. CACHELEVEL=0 or CCACHE=0 turn it off, as they do tinygrad's."""
    import tinygrad.codegen as codegen
    import tinygrad.engine.realize as realize

    if not isinstance(codegen.to_program_cache, _Kept):
        programs = _Kept(codegen.to_program_cache, PROGRAMS, _program)
        codegen.to_program_cache = realize.to_program_cache = programs


def keep_schedules() -> None:
    """Makes tinygrad keep the schedules it makes in its disk cache, as keep_programs keeps the
    programs, and read them there before it schedules a graph: Llama 3.1 8B's 84 take 0.26 MB,
    compressed, and loaded, a later start's warm-up takes 9 s rather than 12. A schedule's row is
    by the key of tinygrad's own cache of them, the graph's without its buffers or its variables'
    values, and by tinygrad's source and the environment variables it reads, so that a change to
    any schedules anew. CACHELEVEL=0 or SCACHE=0 turn it off, as they do tinygrad's."""
    import tinygrad.schedule as schedule

    if not isinstance(schedule.schedule_cache, _Kept):
        schedule.schedule_cache = _Kept(schedule.schedule_cache, SCHEDULES, _schedule)


class _Kept(dict[Any, UOp]):
    """What tinygrad keeps in memory by a key of its own, its lowered programs or its schedules,
    read from its disk cache where none is in memory, and written there as tinygrad makes them:
    each in `table`, in the row that `row` names of its key, or nowhere for None."""

    def __init__(self, kept: dict, table: str, row: Callable[[Any], str | None]):
        super().__init__(kept)
        self.table, self.row = table, row

    def __contains__(self, key: object) -> bool:
        return dict.__contains__(self, key) or self._load(key)

    def get(self, key: Any, default: Any = None) -> Any:
        return dict.get(self, key, default) if key in self else default  # `in` reads the disk

    def __setitem__(self, key: Any, made: UOp) -> None:
        if dict.get(self, key) is not made and (name := self.row(key)) is not None:
            data = zlib.compress(pickle.dumps(made), 1)
            diskcache_put(self.table, name, (hashlib.sha256(data).digest(), data))
        dict.__setitem__(self, key, made)

    def _load(self, key: Any) -> bool:
        # a row damaged on its way to the disk or back, which its digest tells, or that no longer
        # unpickles, is made anew and written over: it would be served at every start
        if (name := self.row(key)) is None:
            return False
        try:
            if (row := diskcache_get(self.table, name)) is None:
                return False
            digest, data = row
            if hashlib.sha256(data).digest() != digest:
                return False
            made = pickle.loads(zlib.decompress(data))
        except Exception:
            return False
        dict.__setitem__(self, key, made)
        return True


def _program(key: tuple) -> str | None:
    # a program's row, of to_program_key's AST key, renderer type, target, an architecture's, and
    # configuration, and of clang's flags and what else lowers it; None where tinygrad keeps no
    # binaries
    if not CCACHE:
        return None
    ast, renderer, target, *config = key
    parts = (ast.hex(), f"{renderer.__module__}.{renderer.__qualname__}", repr(target),
             repr(config), repr(clang.FLAGS), _fingerprint())  # fmt: skip
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _schedule(key: bytes) -> str:
    # a schedule's row, of the key of tinygrad's schedule cache and what else schedules it
    return hashlib.sha256(f"{key.hex()}\n{_fingerprint()}".encode()).hexdigest()


@functools.cache
def _fingerprint() -> str:
    # a digest of what schedules, lowers and compiles a kernel: tinygrad's source and the AMD
    # renderer's, and the environment variables tinygrad reads, as they switch parts of any, as set
    # when this process first lowers, as tinygrad's getenv caches them
    digest, root, read = hashlib.sha256(), pathlib.Path(tinygrad.__file__).parent, set[bytes]()
    for path in sorted(root.rglob("*.py")):
        source = path.read_bytes()
        digest.update(str(path.relative_to(root)).encode() + source)
        read.update(_READ.findall(source))
    digest.update(pathlib.Path(clang.__file__).read_bytes())
    names = sorted(name.decode() for name in read)
    digest.update(repr([(name, os.environ[name]) for name in names if name in os.environ]).encode())
    return digest.hexdigest()
