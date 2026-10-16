"""The machine a model runs on: its AMD GPUs, as KFD lists them, an NVIDIA GPU, as nvidia-smi
reports it, and the memory free for a model's buffers, by which the engine sizes a context and
`leatengine doctor` reports it.

Everything here reads what Linux and the drivers show any user, under a `root` that tests set.
"""

import contextlib
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# the RDNA 3 and 3.5 APUs, whose GPU memory is the machine's own: Phoenix's, Strix Point's,
# Strix Halo's and Krackan Point's
_APUS = ("gfx1103", "gfx1150", "gfx1151", "gfx1152")
_NODES = "sys/devices/virtual/kfd/kfd/topology/nodes"


@dataclass
class GPU:
    """An AMD GPU, as KFD's topology lists it."""

    arch: str  # as gfx1151
    cus: int
    render: int  # its render node's minor, of /dev/dri/renderD<render>
    firmware: str  # its compute and SDMA engines' firmware versions
    apu: bool  # whether its memory is the machine's own


@dataclass(frozen=True)
class Memory:
    """The memory a device's buffers come from: its pool, as GTT, and the bytes of it there are and
    that are free now."""

    pool: str
    total: int
    free: int


def gpus(root: Path = Path("/")) -> list[GPU]:
    """KFD's GPUs, in tinygrad's order: its nodes with a GPU id, by number."""
    nodes = root / _NODES
    if not nodes.is_dir():
        return []
    found = []
    for node in sorted((n for n in nodes.iterdir() if n.name.isdigit()), key=lambda n: int(n.name)):
        if (read(node / "gpu_id") or "0") == "0":
            continue
        props = {}
        for line in (read(node / "properties") or "").splitlines():
            key, _, value = line.partition(" ")
            with contextlib.suppress(ValueError):
                props[key] = int(value)
        v = props.get("gfx_target_version", 0)
        arch = f"gfx{v // 10000}{v // 100 % 100:x}{v % 100:x}"  # as tinygrad names it
        cus = props.get("simd_count", 0) // max(props.get("simd_per_cu", 1), 1)
        firmware = f"MEC {props.get('fw_version', '?')}, SDMA {props.get('sdma_fw_version', '?')}"
        apu = arch in _APUS or props.get("cpu_cores_count", 0) > 0
        found.append(GPU(arch, cus, props.get("drm_render_minor", 128), firmware, apu))
    return found


def memory(device: str, root: Path = Path("/")) -> Memory | None:
    """The memory tinygrad's `device`, as DEV names it, takes buffers from, now: the machine's for
    the CPU; of an AMD GPU, the pool amdgpu serves it from, VRAM, or of an APU the larger of GTT
    and its carve-out, and no more than Linux has available, as its pool is the machine's memory;
    of an NVIDIA GPU, its memory, as nvidia-smi reports it. None where they do not say, as of
    tinygrad's emulated GPU."""
    backend = device.split(":")[0]
    if backend == "CPU":
        return _ram(root)
    if backend == "AMD" and (found := gpus(root)):
        return _amd(root, found[0])
    if backend in ("NV", "CUDA"):
        return _nvidia()
    return None


def ram(root: Path = Path("/")) -> int | None:
    """The memory Linux has, in bytes, which leaves out an APU's carve-out."""
    return _meminfo(root, "MemTotal")


def used(render: int, root: Path = Path("/")) -> dict[str, int]:
    """How much of GTT and of VRAM, an APU's carve-out, amdgpu reports in use, by pool."""
    return _pools(render, root, "used")


def sizes(render: int, root: Path = Path("/")) -> dict[str, int]:
    """The size of GTT and of VRAM, an APU's carve-out, amdgpu reports, by pool."""
    return _pools(render, root, "total")


def pool(gpu: GPU, total: dict[str, int]) -> tuple[str, str]:
    """The pool amdgpu serves a GPU's buffers from, of `total`'s, and what to call it: tinygrad
    asks KFD for VRAM, which amdgpu serves from GTT on an APU whose carve-out is less than GTT."""
    if gpu.apu and total["VRAM"] < total["GTT"]:
        return "GTT", "GTT"
    return "VRAM", "the carve-out" if gpu.apu else "VRAM"


def read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def number(path: Path) -> int | None:
    text = read(path)
    return int(text) if text is not None and text.isdigit() else None


def _ram(root: Path) -> Memory | None:
    total, available = _meminfo(root, "MemTotal"), _meminfo(root, "MemAvailable")
    return None if total is None or available is None else Memory("RAM", total, available)


def _pools(render: int, root: Path, kind: str) -> dict[str, int]:
    device = root / f"sys/class/drm/renderD{render}/device"
    found = {p.upper(): number(device / f"mem_info_{p}_{kind}") for p in ("gtt", "vram")}
    return {p: n for p, n in found.items() if n is not None}


def _amd(root: Path, gpu: GPU) -> Memory | None:
    total, taken = sizes(gpu.render, root), used(gpu.render, root)
    if len(total) < 2 or len(taken) < 2:
        return None
    served, called = pool(gpu, total)
    free = total[served] - taken[served]
    if gpu.apu and (available := _meminfo(root, "MemAvailable")) is not None:
        free = min(free, available)
    return Memory(called, total[served], free)


def _nvidia() -> Memory | None:
    # the first GPU's, as tinygrad's NV:0
    if (said := _nvidia_smi("memory.total,memory.used", "nounits")) is None:
        return None
    try:
        total, taken = (int(n) << 20 for n in said)
    except ValueError:
        return None
    return Memory("the GPU's memory", total, total - taken)


def nvidia_gpu() -> str | None:
    """The first NVIDIA GPU's name and driver, as nvidia-smi gives them, if it is there."""
    said = _nvidia_smi("name,driver_version")
    return None if said is None else f"{said[0]}, driver {said[1]}"


def _nvidia_smi(fields: str, *options: str) -> list[str] | None:
    # the first GPU's fields, as nvidia-smi gives them, if it is there and answers
    if shutil.which("nvidia-smi") is None:
        return None
    query = [
        "nvidia-smi",
        f"--query-gpu={fields}",
        f"--format=csv,noheader{''.join(',' + o for o in options)}",
    ]
    try:
        done = subprocess.run(query, capture_output=True, text=True, timeout=30, check=True)
        said = [x.strip() for x in done.stdout.splitlines()[0].split(",")]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None
    return said if len(said) == fields.count(",") + 1 else None


def _meminfo(root: Path, field: str) -> int | None:
    # a field of /proc/meminfo, in bytes
    found = re.search(rf"{field}:\s*(\d+)\s*kB", read(root / "proc/meminfo") or "")
    return int(found.group(1)) * 1024 if found else None
