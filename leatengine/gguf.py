"""GGUF reader: metadata, tensor index and tensor data, of one file or of the parts of a split one,
as llama.cpp's gguf-split writes them.

Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

import io
import math
import mmap
import os
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tinygrad import Device, Tensor, dtypes
from tinygrad.device import Buffer

from leatengine.files import PART
from leatengine.quant import BLOCK, NATIVE, GGMLType, QTensor

# metadata value type -> struct format; 8 is a string, 9 an array
_SCALAR = dict(enumerate("BbHhIif?")) | {10: "Q", 11: "q", 12: "d"}
_STRING, _ARRAY = 8, 9
CHUNK = 1 << 30  # bytes of adjacent tensors `load` reads into one buffer; a larger one has its own


@dataclass(frozen=True)
class TensorInfo:
    name: str
    type: GGMLType
    shape: tuple[int, ...]  # row-major; GGUF itself lists dimensions innermost first
    offset: int  # absolute byte offset in the file
    nbytes: int
    part: int = 0  # the part of a split GGUF it is in, from 0


class _Reader:
    def __init__(self, buf: memoryview):
        self.buf, self.pos = buf, 0

    def scalar(self, fmt: str) -> Any:
        (value,) = struct.unpack_from("<" + fmt, self.buf, self.pos)
        self.pos += struct.calcsize(fmt)
        return value

    def string(self) -> str:
        n = self.scalar("Q")
        self.pos += n
        # as bytes in llama.cpp: a token may not be UTF-8 of its own
        return str(self.buf[self.pos - n : self.pos], "utf-8", errors="replace")

    def value(self, vtype: int) -> Any:
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            etype, count = self.scalar("I"), self.scalar("Q")
            if etype in _SCALAR:
                fmt = f"<{count}{_SCALAR[etype]}"
                values = list(struct.unpack_from(fmt, self.buf, self.pos))
                self.pos += struct.calcsize(fmt)
                return values
            return [self.value(etype) for _ in range(count)]
        if vtype not in _SCALAR:
            raise ValueError(f"unknown GGUF metadata value type {vtype}")
        return self.scalar(_SCALAR[vtype])


@dataclass(frozen=True)
class GGUF:
    """An opened GGUF: its metadata and tensor index, of a split one its first part's metadata
    and every part's tensors. Tensor data is read by `load`."""

    path: Path  # the file, of a split GGUF its first part
    metadata: dict[str, Any]
    tensors: dict[str, TensorInfo]
    parts: tuple[Path, ...] = ()  # of a split GGUF, every part's file, the first `path`

    @staticmethod
    def open(path: str | Path) -> "GGUF":
        path = Path(path)
        metadata, tensors = _opened(path, 0)
        if (count := metadata.get("split.count", 1)) <= 1:
            return GGUF(path, metadata, tensors)
        named = PART.fullmatch(path.name)
        if named is None or int(named[3]) != count:
            raise ValueError(f"{path}: a part of a split GGUF of {count}, not named as one")
        if (number := metadata.get("split.no", 0)) != 0:
            raise ValueError(
                f"{path}: part {number + 1} of a split GGUF of {count}: open the first"
            )
        parts = [
            path.with_name(f"{named[1]}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1)
        ]
        for i, part in enumerate(parts[1:], 1):
            if not part.exists():
                raise ValueError(
                    f"{path}: a split GGUF of {count} parts, but {part.name} is not there"
                )
            held, more = _opened(part, i)
            if held.get("split.no") != i:
                raise ValueError(f"{part}: not part {i + 1} of {path.name}'s split GGUF")
            tensors |= more
        if len(tensors) != (expected := metadata.get("split.tensors.count", len(tensors))):
            raise ValueError(f"{path}: its parts hold {len(tensors)} tensors of {expected}")
        return GGUF(path, metadata, tensors, tuple(parts))

    def load(self, names: Iterable[str] | None = None) -> dict[str, QTensor]:
        """Copies the data of the tensors named, or of all, to the device and returns a view per
        tensor. Tensors adjacent in the file share a buffer of up to CHUNK bytes, read through
        one host buffer, from a file that may be read-only, which the page cache then drops: a
        load takes little host memory beside the model's."""
        tensors = list(self.tensors.values()) if names is None else [self.tensors[n] for n in names]
        if not tensors:
            return {}
        device = Device.DEFAULT
        alignment = self.metadata.get("general.alignment", 32)
        files = self.parts or (self.path,)
        runs = {i: _runs([t for t in tensors if t.part == i], alignment) for i in range(len(files))}
        # tinygrad's own memory, which a GPU maps to copy from: once, as the buffer lives on
        largest = max(_end(run[-1]) - run[0].offset for part in runs.values() for run in part)
        host = Tensor.empty(largest, dtype=dtypes.uint8, device="CPU").realize()
        assert isinstance(buffer := host.uop.buffer, Buffer)
        memory = buffer.ensure_allocated().as_memoryview(allow_zero_copy=True)
        out: dict[str, QTensor] = {}
        loaded = 0
        for path, part in zip(files, runs.values(), strict=True):
            loaded = self._load(path, part, device, host, memory, out, loaded)
        return out

    def _load(
        self, path: Path, runs: list[list[TensorInfo]], device: str, host: Tensor,
        memory: memoryview, out: dict[str, QTensor], loaded: int,
    ) -> int:  # fmt: skip
        # the runs of one file, through the host buffer, into `out`; the bytes loaded then
        if not runs:
            return loaded
        with open(path, "rb", buffering=0) as f:
            for run in runs:
                start, size = run[0].offset, _end(run[-1]) - run[0].offset
                _read(f, start, memory[:size])
                try:
                    data = host[:size].clone(device).realize()
                    Device[device].synchronize()  # before the next run overwrites the host buffer
                except MemoryError as e:
                    raise MemoryError(
                        f"{path.name}: {device} has no room for {size / 2**30:.2f} GiB more, "
                        f"of {run[0].name} to {run[-1].name}, with {loaded / 2**30:.2f} GiB "
                        "loaded; see the Strix Halo guide's notes on GPU memory"
                    ) from e
                os.posix_fadvise(f.fileno(), start, size, os.POSIX_FADV_DONTNEED)
                for t in run:
                    out[t.name] = _view(data[t.offset - start : t.offset - start + t.nbytes], t)
                loaded += size
        return loaded


def _opened(path: Path, part: int) -> tuple[dict[str, Any], dict[str, TensorInfo]]:
    # a file's metadata and tensors, each of `part`, checked to end within it
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        with memoryview(mm) as buf:
            try:
                metadata, tensors = _parse(buf, part)
            except struct.error:  # its header or index cut short
                raise ValueError(f"{path}: truncated, its header ends early") from None
        size = len(mm)
    for t in tensors.values():
        if t.offset + t.nbytes > size:
            raise ValueError(f"{path}: truncated, tensor {t.name!r} ends past the end of file")
    return metadata, tensors


def _end(t: TensorInfo) -> int:
    return t.offset + t.nbytes


def _runs(tensors: list[TensorInfo], alignment: int) -> list[list[TensorInfo]]:
    # the tensors by offset, in runs of those adjacent in the file but for alignment padding, of
    # CHUNK bytes at most but for a larger tensor alone
    runs: list[list[TensorInfo]] = []
    for t in sorted(tensors, key=lambda t: t.offset):
        run = runs[-1] if runs else []
        if run and t.offset - _end(run[-1]) < alignment and _end(t) - run[0].offset <= CHUNK:
            run.append(t)
        else:
            runs.append([t])
    return runs


def _read(f: io.FileIO, at: int, into: memoryview) -> None:
    f.seek(at)
    end = at + len(into)
    while into:
        if not (n := f.readinto(into)):
            raise ValueError(f"{f.name}: ends before byte {end}")
        into = into[n:]


def _view(raw: Tensor, t: TensorInfo) -> QTensor:
    # a tensor's bytes in its storage format
    if t.type in NATIVE:
        return QTensor(raw.bitcast(NATIVE[t.type]), t.type, t.shape)
    return QTensor(raw.reshape(-1, BLOCK[t.type][1]), t.type, t.shape)


def _parse(buf: memoryview, part: int = 0) -> tuple[dict[str, Any], dict[str, TensorInfo]]:
    r = _Reader(buf)
    if bytes(buf[:4]) != b"GGUF":
        raise ValueError("not a GGUF file")
    r.pos = 4
    if (version := r.scalar("I")) not in (2, 3):
        raise ValueError(f"unsupported GGUF version {version}")
    n_tensors, n_kv = r.scalar("Q"), r.scalar("Q")
    metadata = {}
    for _ in range(n_kv):
        key = r.string()
        metadata[key] = r.value(r.scalar("I"))

    infos = []
    for _ in range(n_tensors):
        name = r.string()
        dims = [r.scalar("Q") for _ in range(r.scalar("I"))]
        type_id, offset = r.scalar("I"), r.scalar("Q")
        infos.append((name, dims, type_id, offset))

    alignment = metadata.get("general.alignment", 32)
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError(f"alignment {alignment} is not a power of 2")
    data_start = -(-r.pos // alignment) * alignment
    tensors = {}
    for name, dims, type_id, offset in infos:
        try:
            ggml_type = GGMLType(type_id)
        except ValueError:
            raise ValueError(f"tensor {name!r} has unknown ggml type {type_id}") from None
        elements, block_bytes = BLOCK[ggml_type]
        if dims and dims[0] % elements:  # blocks lie along rows
            raise ValueError(f"tensor {name!r} has rows of {dims[0]}, not a multiple of {elements}")
        nbytes = math.prod(dims) // elements * block_bytes
        shape = tuple(reversed(dims))
        tensors[name] = TensorInfo(name, ggml_type, shape, data_start + offset, nbytes, part)
    return metadata, tensors
