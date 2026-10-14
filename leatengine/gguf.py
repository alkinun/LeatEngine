"""GGUF reader: metadata, tensor index and tensor data.

Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

import ctypes
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
    """An opened GGUF file: its metadata and tensor index. Tensor data is read by `load`."""

    path: Path
    metadata: dict[str, Any]
    tensors: dict[str, TensorInfo]

    @staticmethod
    def open(path: str | Path) -> "GGUF":
        path = Path(path)
        with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            with memoryview(mm) as buf:
                metadata, tensors = _parse(buf)
            size = len(mm)
        for t in tensors.values():
            if t.offset + t.nbytes > size:
                raise ValueError(f"{path}: truncated, tensor {t.name!r} ends past the end of file")
        return GGUF(path, metadata, tensors)

    def load(
        self, device: str | None = None, names: Iterable[str] | None = None
    ) -> dict[str, QTensor]:
        """Copies the data of the tensors named, or of all, to `device` and returns a view per
        tensor. Tensors adjacent in the file share a buffer of up to CHUNK bytes, read through
        one host buffer, from a file that may be read-only, which the page cache then drops: a
        load takes little host memory beside the model's."""
        tensors = list(self.tensors.values()) if names is None else [self.tensors[n] for n in names]
        if not tensors:
            return {}
        device = Device.canonicalize(device)
        runs = _runs(tensors, self.metadata.get("general.alignment", 32))
        host = bytearray(max(_end(run[-1]) - run[0].offset for run in runs))
        address, out, loaded = ctypes.addressof(ctypes.c_char.from_buffer(host)), {}, 0
        with open(self.path, "rb", buffering=0) as f:
            for run in runs:
                start, size = run[0].offset, _end(run[-1]) - run[0].offset
                _read(f, start, memoryview(host)[:size])
                staged = Tensor.from_blob(address, (size,), dtype=dtypes.uint8, device="CPU")
                try:
                    data = staged.clone(device).realize()
                    Device[device].synchronize()  # before the next run overwrites the host buffer
                except MemoryError as e:
                    raise MemoryError(
                        f"{self.path.name}: {device} has no room for {size / 2**30:.2f} GiB more, "
                        f"of {run[0].name} to {run[-1].name}, with {loaded / 2**30:.2f} GiB "
                        "loaded; see the Strix Halo guide's notes on GPU memory"
                    ) from e
                os.posix_fadvise(f.fileno(), start, size, os.POSIX_FADV_DONTNEED)
                for t in run:
                    out[t.name] = _view(data[t.offset - start : t.offset - start + t.nbytes], t)
                loaded += size
        return out


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
    while into:
        if not (n := f.readinto(into)):
            raise ValueError(f"{f.name}: ends before byte {at + len(into)}")
        into = into[n:]


def _view(raw: Tensor, t: TensorInfo) -> QTensor:
    # a tensor's bytes in its storage format
    if t.type in NATIVE:
        return QTensor(raw.bitcast(NATIVE[t.type]), t.type, t.shape)
    return QTensor(raw.reshape(-1, BLOCK[t.type][1]), t.type, t.shape)


def _parse(buf: memoryview) -> tuple[dict[str, Any], dict[str, TensorInfo]]:
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
        tensors[name] = TensorInfo(name, ggml_type, shape, data_start + offset, nbytes)
    return metadata, tensors
