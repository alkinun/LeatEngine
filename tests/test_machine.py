import subprocess

import pytest

from leatengine import machine
from leatengine.machine import Memory
from tests.helpers import GIB, GPU_NODE, properties, write

NODES = "sys/devices/virtual/kfd/kfd/topology/nodes"
DEVICE = "sys/class/drm/renderD128/device"


@pytest.fixture
def root(tmp_path):
    # a machine of 124 GiB, 100 GiB of them available, and a GPU, which each test makes
    write(tmp_path, {
        "proc/meminfo": "MemTotal:       130023424 kB\nMemAvailable:   104857600 kB",
        f"{NODES}/0/gpu_id": "0",
        f"{NODES}/0/properties": properties({"cpu_cores_count": 32, "simd_count": 0}),
        f"{NODES}/1/gpu_id": "41234",
    })  # fmt: skip
    return tmp_path


def gpu(root, node: dict, vram: tuple[int, int], gtt: tuple[int, int]) -> None:
    # a GPU of KFD's node properties, and its pools' sizes and use, in GiB
    write(root, {f"{NODES}/1/properties": properties(node)})
    for pool, (total, used) in (("vram", vram), ("gtt", gtt)):
        write(root, {f"{DEVICE}/mem_info_{pool}_total": total * GIB,
                     f"{DEVICE}/mem_info_{pool}_used": used * GIB})  # fmt: skip


def test_cpu(root):
    assert machine.memory("CPU", root) == Memory("RAM", 124 * GIB, 100 * GIB)


def test_strix_halo(root):
    # an APU's buffers come from GTT where its carve-out is smaller, no more of it than Linux has
    # available; and from the carve-out where that is the larger
    gpu(root, GPU_NODE, (1, 0), (120, 10))
    assert machine.memory("AMD", root) == Memory("GTT", 120 * GIB, 100 * GIB)
    gpu(root, GPU_NODE, (1, 0), (120, 30))
    assert machine.memory("AMD", root) == Memory("GTT", 120 * GIB, 90 * GIB)
    gpu(root, GPU_NODE, (96, 6), (16, 0))
    assert machine.memory("AMD:0", root) == Memory("the carve-out", 96 * GIB, 90 * GIB)


def test_discrete(root):
    # a GPU of its own memory, VRAM, of which Linux's says nothing
    gpu(root, GPU_NODE | {"gfx_target_version": 110000}, (24, 4), (62, 0))
    assert machine.memory("AMD", root) == Memory("VRAM", 24 * GIB, 20 * GIB)


def test_unknown(root):
    # a GPU whose driver does not say, none, or tinygrad's emulated one
    write(root, {f"{NODES}/1/properties": properties(GPU_NODE)})
    assert machine.memory("AMD", root) is None
    (root / f"{NODES}/1/gpu_id").write_text("0\n")
    assert machine.memory("AMD", root) is None and machine.memory("MOCK+AMD", root) is None


def test_nvidia(monkeypatch):
    # nvidia-smi's first GPU, in MiB, where it is installed and answers
    monkeypatch.setattr(machine.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    answers = iter(["24576, 4096\n24576, 0\n", subprocess.CalledProcessError(9, "nvidia-smi")])

    def run(args, **kwargs):
        if isinstance(answer := next(answers), Exception):
            raise answer
        return subprocess.CompletedProcess(args, 0, answer, "")

    monkeypatch.setattr(machine.subprocess, "run", run)
    assert machine.memory("NV") == Memory("the GPU's memory", 24 * GIB, 20 * GIB)
    assert machine.memory("NV") is None
    monkeypatch.setattr(machine.shutil, "which", lambda name: None)
    assert machine.memory("CUDA") is None
