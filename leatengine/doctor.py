"""What `leatengine doctor` checks: whether this machine is ready for LeatEngine on Strix Halo.

Each check reads what the kernel, the driver and the toolchain show any user, and says what it
found, whether that will do, and what to change where it will not: the kernel, access to the GPU,
the GPU, where its buffers come from and how much of them there can be, and clang. Two checks run
something: clang compiles a kernel for the GPU, and a process of its own runs one there and
allocates a buffer, as a GPU that hangs would otherwise hang the report.

A report names the machine's model and firmware, and nothing of its owner's, so that it can be
pasted into an issue as it is.
"""

import contextlib
import grp
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import textwrap
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

TARGET = "gfx1151"  # Strix Halo's GPU, the one LeatEngine is made for
# the first Linux with KFD's fixes for gfx1151, as AMD's Strix Halo guide gives it for most
# distributions; Ubuntu 24.04's HWE and OEM kernels have them from 6.17.0-19 and 6.14.0-1018 on
KERNEL = (6, 18, 4)
GUIDE = "https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html"
# the share of the machine's memory that the GPU's buffers may take, under which the largest
# models do not fit: GTT's default is about half
ENOUGH = 0.6
RUN_SECONDS = 120  # for the GPU's run, compiles included, before it counts as hung
# the RDNA 3 and 3.5 APUs, whose GPU memory is the machine's own: Phoenix's, Strix Point's,
# Strix Halo's and Krackan Point's
_APUS = ("gfx1103", "gfx1150", "gfx1151", "gfx1152")
_NODES = "sys/devices/virtual/kfd/kfd/topology/nodes"
# a kernel for clang, as tinygrad renders one for AMD's GPUs
_KERNEL = 'extern "C" __attribute__((global)) void doctor(float* x) { x[0] = 1.0f; }'


@dataclass
class Check:
    """A line of the report: what was checked, how it stands (ok, warn, fail or info), what was
    found, and what to do where it is not ok."""

    name: str
    status: str
    found: str
    fix: str = ""


@dataclass
class GPU:
    """An AMD GPU, as KFD's topology lists it."""

    arch: str  # as gfx1151
    cus: int
    render: int  # its render node's minor, of /dev/dri/renderD<render>
    firmware: str  # its compute and SDMA engines' firmware versions
    apu: bool  # whether its memory is the machine's own


def checks(root: Path = Path("/"), run: bool = True) -> list[Check]:
    """The checks of the system whose files are under `root`; `run` runs a kernel on the GPU when
    nothing before it failed."""
    ram = _meminfo(root)
    found = [_machine(root), _cpu(root, ram), _system(root), _kernel(root), *_environment()]
    if (profile := _read(root / "sys/firmware/acpi/platform_profile")) is not None:
        found.append(Check("power mode", "info", profile))
    gpus = _gpus(root)
    gpu = gpus[0] if gpus else None  # tinygrad's AMD:0
    found += [_device_file("/dev/kfd", root / "dev/kfd"), _gpu(gpus)]
    if gpu is not None:
        render = f"dev/dri/renderD{gpu.render}"
        found.append(_device_file(f"/{render}", root / render))
        found += _memory(root, gpu, ram)
    found.append(clang := _clang())
    if clang.status == "ok":
        found.append(_compile(gpu.arch if gpu is not None else TARGET))
    if run and gpu is not None and not any(c.status == "fail" for c in found):
        found.append(gpu_run(gpu.render))
    return found


def report(found: list[Check]) -> str:
    """The checks as text, a line each, with what to do under those that need it."""
    width = max(len(c.name) for c in found)
    lines = [f"LeatEngine {_leatengine()}, tinygrad {_tinygrad()}", ""]
    for c in found:
        lines.append(f"  {c.status:<4}  {c.name:<{width}}  {c.found}")
        if c.fix:
            pad = " " * (width + 10)
            lines += textwrap.wrap(c.fix, 100, initial_indent=pad + "fix: ",
                                   subsequent_indent=pad + "     ", break_long_words=False,
                                   break_on_hyphens=False)  # fmt: skip
    problems = sum(c.status == "fail" for c in found)
    warnings = sum(c.status == "warn" for c in found)
    if problems or warnings:
        said = [f"{n} {word}{'s' if n > 1 else ''}"
                for n, word in ((problems, "problem"), (warnings, "warning")) if n]  # fmt: skip
        lines += ["", f"{' and '.join(said)}."]
    else:
        lines += ["", "Ready."]
    lines.append("The report names this machine's model, not its owner: paste it into an issue as "
                 "it is.")  # fmt: skip
    return "\n".join(lines)


def as_json(found: list[Check]) -> dict[str, Any]:
    """The checks as one JSON object, with the versions they were made with."""
    checked = [asdict(c) for c in found]
    return {"leatengine": _leatengine(), "tinygrad": _tinygrad(), "checks": checked}


def gpu_run(render: int | None, dev: str = "AMD") -> Check:
    """Runs a kernel on the AMD GPU that DEV=`dev` gives tinygrad, in a process of its own, so
    that a GPU that hangs cannot hang the report; and allocates 1 GiB there to see where it
    lands, if the GPU's render node has minor `render`."""
    code = f"from leatengine.doctor import probe; probe({render!r})"
    env = {**os.environ, "DEV": dev}
    # as isolated from PYTHONPATH and the like as this process, as the release tarball's launcher
    # runs it
    isolated = ["-I"] if sys.flags.isolated else []
    try:
        done = subprocess.run([sys.executable, *isolated, "-c", code], env=env,
                              capture_output=True, text=True, timeout=RUN_SECONDS)  # fmt: skip
    except subprocess.TimeoutExpired:
        return Check("gpu run", "fail", f"no answer in {RUN_SECONDS} s",
                     "the GPU may have hung: see `dmesg | grep -i amdgpu`")  # fmt: skip
    if done.returncode:
        last = (done.stderr.strip().splitlines() or [f"exit status {done.returncode}"])[-1]
        return Check("gpu run", "fail", last)
    result = json.loads(done.stdout.strip().splitlines()[-1])
    if result["sum"] != 4096:
        return Check("gpu run", "fail", f"a sum of 4096 ones came to {result['sum']}")
    landed = f"; 1 GiB landed in {result['pool']}" if result["pool"] else ""
    return Check("gpu run", "ok", f"a kernel ran on {result['arch']}{landed}")


def probe(render: int | None) -> None:
    """gpu_run's process: sums 4096 ones on tinygrad's device, then, where the driver reports its
    pools' use, allocates 1 GiB and sees which pool grew; prints what happened as JSON."""
    from tinygrad import Device, Tensor
    from tinygrad.device import Buffer

    import leatengine.kernels  # noqa: F401, as AMD devices compile with clang from here on

    device = Device.DEFAULT
    dev = Device[device]
    total = Tensor.ones(4096, device=device).sum().item()
    pool = None
    if render is not None and (before := _used(render)):
        buffer = Buffer(device, 1 << 30, preallocate=True)
        dev.synchronize()
        grew = {name: used - before.get(name, used) for name, used in _used(render).items()}
        buffer.deallocate()
        pool = max(grew, key=lambda name: grew[name]) if max(grew.values()) >= 1 << 29 else None
    print(json.dumps({"arch": getattr(dev, "arch", device), "sum": total, "pool": pool}))


def _used(render: int) -> dict[str, int]:
    # how much of GTT and of VRAM, the carve-out of an APU, the driver reports in use
    device = Path(f"/sys/class/drm/renderD{render}/device")
    used = {pool.upper(): _number(device / f"mem_info_{pool}_used") for pool in ("gtt", "vram")}
    return {pool: n for pool, n in used.items() if n is not None}


def _machine(root: Path) -> Check:
    dmi = root / "sys/class/dmi/id"
    vendor, product = _read(dmi / "sys_vendor"), _read(dmi / "product_name")
    bios, date = _read(dmi / "bios_version"), _read(dmi / "bios_date")
    found = " ".join(x for x in (vendor, product) if x) or "unknown"
    if bios:
        found += f", BIOS {bios}" + (f" of {date}" if date else "")
    return Check("machine", "info", found)


def _cpu(root: Path, ram: int | None) -> Check:
    model = re.search(r"model name\s*:\s*(.*)", _read(root / "proc/cpuinfo") or "")
    found = model.group(1).strip() if model else "unknown"
    return Check("cpu", "info", found + (f", {_gib(ram)} of memory" if ram is not None else ""))


def _system(root: Path) -> Check:
    release = _read(root / "etc/os-release") or _read(root / "usr/lib/os-release") or ""
    name = re.search(r'^PRETTY_NAME="?([^"\n]*)"?', release, re.M)
    return Check("system", "info", f"{name.group(1) if name else 'Linux'}, Python "
                                   f"{platform.python_version()}")  # fmt: skip


def _kernel(root: Path) -> Check:
    release = _read(root / "proc/sys/kernel/osrelease") or platform.release()
    version = tuple(int(n) for n in re.findall(r"\d+", release.split("-")[0])[:3])
    if version >= KERNEL:
        return Check("kernel", "ok", release)
    return Check("kernel", "warn", release,
                 f"Linux {'.'.join(map(str, KERNEL))} or newer has KFD's fixes for gfx1151, as do "
                 f"Ubuntu 24.04's HWE kernel from 6.17.0-19 and OEM kernel from 6.14.0-1018 "
                 f"({GUIDE})")  # fmt: skip


def _environment() -> list[Check]:
    # DEV, where it chooses a backend other than AMD's, which LeatEngine would then run on
    dev = os.environ.get("DEV")
    if dev is None or dev.split(":")[0].endswith("AMD"):
        return []
    return [Check("DEV", "warn", f"{dev}: LeatEngine runs there rather than on the AMD GPU",
                  "unset DEV, or set DEV=AMD")]  # fmt: skip


def _gpus(root: Path) -> list[GPU]:
    # KFD's GPUs, in tinygrad's order: its nodes with a GPU id, by number
    nodes = root / _NODES
    if not nodes.is_dir():
        return []
    gpus = []
    for node in sorted((n for n in nodes.iterdir() if n.name.isdigit()), key=lambda n: int(n.name)):
        if (_read(node / "gpu_id") or "0") == "0":
            continue
        props = {}
        for line in (_read(node / "properties") or "").splitlines():
            key, _, value = line.partition(" ")
            with contextlib.suppress(ValueError):
                props[key] = int(value)
        v = props.get("gfx_target_version", 0)
        arch = f"gfx{v // 10000}{v // 100 % 100:x}{v % 100:x}"  # as tinygrad names it
        cus = props.get("simd_count", 0) // max(props.get("simd_per_cu", 1), 1)
        firmware = f"MEC {props.get('fw_version', '?')}, SDMA {props.get('sdma_fw_version', '?')}"
        apu = arch in _APUS or props.get("cpu_cores_count", 0) > 0
        gpus.append(GPU(arch, cus, props.get("drm_render_minor", 128), firmware, apu))
    return gpus


def _gpu(gpus: list[GPU]) -> Check:
    if not gpus:
        return Check("gpu", "fail", "none in KFD's topology",
                     "LeatEngine reaches the GPU through amdgpu's KFD: see whether amdgpu found "
                     "the GPU, `dmesg | grep -i amdgpu`")  # fmt: skip
    gpu = gpus[0]
    found = f"{gpu.arch}, {gpu.cus} CUs, firmware {gpu.firmware}"
    found += f" (the first of {len(gpus)})" if len(gpus) > 1 else ""
    major = int(gpu.arch[3:-2] or 0)
    if gpu.arch == TARGET:
        return Check("gpu", "ok", found)
    if major in (11, 12):
        return Check("gpu", "warn", found, f"LeatEngine is made for Strix Halo's {TARGET}: on "
                     f"{gpu.arch} its RDNA {major - 8} kernels run untested")  # fmt: skip
    return Check("gpu", "fail", found, f"LeatEngine's AMD kernels need RDNA 3 or 4, as Strix "
                 f"Halo's {TARGET}")  # fmt: skip


def _device_file(name: str, path: Path) -> Check:
    # a device file the GPU is reached through, which this user must read and write
    if not path.exists():
        return Check(name, "fail", "missing", "amdgpu makes it: see whether it is loaded, "
                     "`lsmod | grep amdgpu`")  # fmt: skip
    if os.access(path, os.R_OK | os.W_OK):
        return Check(name, "ok", "readable and writable")
    group = _group(path)
    return Check(name, "fail", f"not readable and writable by this user (its group is {group})",
                 f"`sudo usermod -aG {group} $USER`, then log in again")  # fmt: skip


def _memory(root: Path, gpu: GPU, ram: int | None) -> list[Check]:
    # where the GPU's buffers come from, and so how much of them there can be. tinygrad asks KFD
    # for VRAM, which amdgpu serves from GTT on an APU whose carve-out is less than its GTT
    device = root / f"sys/class/drm/renderD{gpu.render}/device"
    vram, gtt = (_number(device / f"mem_info_{pool}_total") for pool in ("vram", "gtt"))
    limit = []
    if (pages := _number(root / "sys/module/ttm/parameters/pages_limit")) is not None:
        size = pages * os.sysconf("SC_PAGE_SIZE")
        limit.append(Check("ttm limit", "info", f"{pages} pages, {_gib(size)}"))
    if vram is None or gtt is None:
        return [Check("gpu memory", "warn", "amdgpu does not report its pools' sizes"), *limit]
    if not gpu.apu:
        return [Check("gpu memory", "info", f"VRAM {_gib(vram)}, GTT {_gib(gtt)}"), *limit]
    pool, most = ("GTT", gtt) if vram < gtt else ("the carve-out", vram)
    said = f"carve-out {_gib(vram)}, GTT {_gib(gtt)}: buffers come from {pool}, {_gib(most)}"
    if ram is None or most >= ENOUGH * (ram + vram):  # MemTotal leaves the carve-out out
        return [Check("gpu memory", "ok", said + " at most"), *limit]
    fix = (f"raise GTT's limit, TTM's pages_limit, as AMD's guide has it ({GUIDE}): "
           "`pipx install amd-debug-tools`, `amd-ttm --set GB`, then reboot")  # fmt: skip
    if pool != "GTT":
        fix = "set the firmware's carve-out small, 0.5 GB, and " + fix
    said += " at most, too little for the largest models"
    return [Check("gpu memory", "warn", said, fix), *limit]


def _clang() -> Check:
    # LeatEngine compiles its AMD kernels with the system's clang, given its AMDGPU target
    fix = "install clang with its AMDGPU target, as most distributions' clang is built"
    if shutil.which("clang") is None:
        return Check("clang", "fail", "not found", fix)
    try:
        version = _output("clang", "--version").splitlines()[0]
        targets = _output("clang", "-print-targets")
    except (OSError, subprocess.SubprocessError, IndexError) as e:
        return Check("clang", "fail", f"does not run: {e}", fix)
    if not re.search(r"^\s*amdgcn\b", targets, re.M):
        return Check("clang", "fail", f"{version}, without the AMDGPU target", fix)
    return Check("clang", "ok", f"{version}, with the AMDGPU target")


def _compile(arch: str) -> Check:
    from leatengine.kernels.clang import ClangCompiler

    try:
        ClangCompiler(arch).compile(_KERNEL)
    except Exception as e:  # clang's errors, or its assembler's
        last = (str(e).strip().splitlines() or [type(e).__name__])[-1]
        return Check("compile", "fail", f"a kernel for {arch}: {last}")
    return Check("compile", "ok", f"a kernel for {arch}")


def _leatengine() -> str:
    # the installed version, and the commit of a checkout it runs from
    version = metadata.version("leatengine")
    checkout = Path(__file__).resolve().parent.parent
    if (checkout / ".git").exists():
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            commit = _output("git", "-C", str(checkout), "describe", "--always", "--dirty")
            version += f" at {commit}"
    return version


def _tinygrad() -> str:
    # the pinned commit tinygrad was installed from, or its version
    with contextlib.suppress(KeyError, TypeError, ValueError):
        url = json.loads(metadata.distribution("tinygrad").read_text("direct_url.json") or "{}")
        return url["vcs_info"]["commit_id"][:9]
    return metadata.version("tinygrad")


def _output(*args: str) -> str:
    done = subprocess.run(args, capture_output=True, text=True, timeout=30, check=True)
    return done.stdout.strip()


def _meminfo(root: Path) -> int | None:
    # the memory Linux has, in bytes, which leaves out the GPU's carve-out
    total = re.search(r"MemTotal:\s*(\d+)\s*kB", _read(root / "proc/meminfo") or "")
    return int(total.group(1)) * 1024 if total else None


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _number(path: Path) -> int | None:
    text = _read(path)
    return int(text) if text is not None and text.isdigit() else None


def _group(path: Path) -> str:
    gid = path.stat().st_gid
    try:
        return grp.getgrgid(gid).gr_name
    except KeyError:
        return str(gid)


def _gib(n: int) -> str:
    return f"{n / 2**30:.1f} GiB"
