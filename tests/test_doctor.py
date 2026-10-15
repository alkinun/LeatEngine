import json
import os
import shutil
import subprocess

import pytest

from leatengine import doctor
from leatengine.cli import main
from leatengine.doctor import Check

GIB = 2**30
GPU_NODE = {"cpu_cores_count": 0, "simd_count": 80, "simd_per_cu": 2, "gfx_target_version": 110501,
            "drm_render_minor": 128, "fw_version": 37, "sdma_fw_version": 14}  # fmt: skip


def write(root, files: dict) -> None:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{text}\n")


def properties(props: dict) -> str:
    return "\n".join(f"{key} {value}" for key, value in props.items())


@pytest.fixture
def strix(tmp_path, monkeypatch):
    # the files of a Strix Halo machine set up as AMD's guide has it: a carve-out of 0.5 GiB and a
    # GTT of 100 GiB, of 124 GiB the system has; clang left out, as the tests of it are below
    monkeypatch.delenv("DEV", raising=False)
    monkeypatch.setattr(doctor, "_clang", lambda: Check("clang", "ok", "clang version 22"))
    monkeypatch.setattr(doctor, "_compile", lambda arch: Check("compile", "ok", arch))
    nodes = "sys/devices/virtual/kfd/kfd/topology/nodes"
    write(tmp_path, {
        "proc/sys/kernel/osrelease": "6.18.4-arch1-1",
        "proc/meminfo": "MemTotal:       130023424 kB\nMemFree:        120000000 kB",
        "proc/cpuinfo": "processor\t: 0\nmodel name\t: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S",
        "etc/os-release": 'NAME="Fedora Linux"\nPRETTY_NAME="Fedora Linux 43 (KDE Plasma)"',
        "sys/class/dmi/id/sys_vendor": "Framework",
        "sys/class/dmi/id/product_name": "Desktop (AMD Ryzen AI Max 300 Series)",
        "sys/class/dmi/id/bios_version": "03.03",
        "sys/class/dmi/id/bios_date": "06/12/2026",
        "sys/firmware/acpi/platform_profile": "performance",
        f"{nodes}/0/gpu_id": "0",
        f"{nodes}/0/properties": properties({"cpu_cores_count": 32, "simd_count": 0}),
        f"{nodes}/1/gpu_id": "41234",
        f"{nodes}/1/properties": properties(GPU_NODE),
        "dev/kfd": "",
        "dev/dri/renderD128": "",
        "sys/class/drm/renderD128/device/mem_info_vram_total": GIB // 2,
        "sys/class/drm/renderD128/device/mem_info_gtt_total": 100 * GIB,
        "sys/module/ttm/parameters/pages_limit": 26214400,
    })  # fmt: skip
    return tmp_path


def by_name(found: list[Check]) -> dict[str, Check]:
    return {c.name: c for c in found}


def test_ready(strix):
    found = by_name(doctor.checks(strix, run=False))
    assert {c.status for c in found.values()} == {"ok", "info"}
    assert found["gpu"].found == "gfx1151, 40 CUs, firmware MEC 37, SDMA 14"
    assert found["gpu memory"].found == (
        "carve-out 0.5 GiB, GTT 100.0 GiB: buffers come from GTT, 100.0 GiB at most"
    )
    assert found["machine"].found == (
        "Framework Desktop (AMD Ryzen AI Max 300 Series), BIOS 03.03 of 06/12/2026"
    )
    assert found["cpu"].found == "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S, 124.0 GiB of memory"
    assert found["system"].found.startswith("Fedora Linux 43 (KDE Plasma), Python 3.")
    assert found["power mode"].found == "performance"
    assert found["ttm limit"].found.startswith("26214400 pages")
    assert [n for n in found if n.startswith("/dev")] == ["/dev/kfd", "/dev/dri/renderD128"]
    text = doctor.report(list(found.values()))
    assert text.splitlines()[0].startswith("LeatEngine ") and "\nReady.\n" in text
    assert "  ok    kernel               6.18.4-arch1-1" in text


@pytest.mark.parametrize("files, name, status, said", [
    # a kernel before the fixes; the fix names Ubuntu's, whose releases do not show that they
    # have them
    ({"proc/sys/kernel/osrelease": "6.17.0-14-generic"}, "kernel", "warn", "HWE kernel"),
    # GTT's default, half of memory: too little for the largest models
    ({"sys/class/drm/renderD128/device/mem_info_gtt_total": 62 * GIB}, "gpu memory", "warn",
     "amd-ttm"),
    # a large carve-out and GTT's default: buffers come from the carve-out, which is enough
    ({"sys/class/drm/renderD128/device/mem_info_vram_total": 96 * GIB,
      "sys/class/drm/renderD128/device/mem_info_gtt_total": 16 * GIB,
      "proc/meminfo": "MemTotal:       33000000 kB"}, "gpu memory", "ok", "from the carve-out"),
    # a carve-out larger than a small GTT: both to change
    ({"sys/class/drm/renderD128/device/mem_info_vram_total": 8 * GIB,
      "sys/class/drm/renderD128/device/mem_info_gtt_total": 4 * GIB}, "gpu memory", "warn",
     "carve-out small"),
    # another RDNA 3 GPU than Strix Halo's runs the kernels untested
    ({"sys/devices/virtual/kfd/kfd/topology/nodes/1/properties":
      properties(GPU_NODE | {"gfx_target_version": 110000})}, "gpu", "warn", "RDNA 3 kernels"),
    ({"sys/devices/virtual/kfd/kfd/topology/nodes/1/gpu_id": "0"}, "gpu", "fail", "dmesg"),
])  # fmt: skip
def test_problems(strix, files, name, status, said):
    write(strix, files)
    check = by_name(doctor.checks(strix, run=False))[name]
    assert check.status == status
    assert said in check.found + check.fix


def test_not_rdna(strix):
    # a GPU of no RDNA generation, named as tinygrad names it, takes none of the kernels
    write(strix, {"sys/devices/virtual/kfd/kfd/topology/nodes/1/properties":
                  properties(GPU_NODE | {"gfx_target_version": 90010})})  # fmt: skip
    check = by_name(doctor.checks(strix, run=False))["gpu"]
    assert check.status == "fail" and check.found.startswith("gfx90a, 40 CUs")
    assert "RDNA 3 or 4" in check.fix


def test_device_files(strix):
    # the device files the GPU is reached through: missing, or not this user's to use
    (strix / "dev/kfd").unlink()
    found = by_name(doctor.checks(strix, run=False))
    assert (found["/dev/kfd"].status, found["/dev/kfd"].found) == ("fail", "missing")
    if os.geteuid() == 0:
        return  # root may use any file
    (strix / "dev/dri/renderD128").chmod(0)
    check = by_name(doctor.checks(strix, run=False))["/dev/dri/renderD128"]
    assert check.status == "fail" and check.fix.startswith("`sudo usermod -aG ")


def test_dev(strix, monkeypatch):
    # DEV choosing another backend than AMD's is said; DEV=AMD, or the emulator's, is not
    monkeypatch.setenv("DEV", "CPU")
    assert by_name(doctor.checks(strix, run=False))["DEV"].status == "warn"
    for dev in ("AMD", "MOCK+AMD", "AMD:1"):
        monkeypatch.setenv("DEV", dev)
        assert "DEV" not in by_name(doctor.checks(strix, run=False))


def test_run(strix, monkeypatch):
    # the GPU runs a kernel once nothing before has failed, its render node said
    runs = []
    monkeypatch.setattr(doctor, "gpu_run", lambda render: runs.append(render) or Check(
        "gpu run", "ok", "ran"))  # fmt: skip
    assert doctor.checks(strix)[-1].name == "gpu run" and runs == [128]
    (strix / "dev/kfd").unlink()
    assert "gpu run" not in by_name(doctor.checks(strix)) and runs == [128]


@pytest.mark.parametrize("outcome, status, said", [
    ("timeout", "fail", "no answer"),
    ((1, "", "Traceback (most recent call last):\nRuntimeError: No device found for 0"), "fail",
     "RuntimeError: No device found for 0"),
    ((0, json.dumps({"arch": "gfx1151", "sum": 0.0, "pool": None}), ""), "fail", "came to 0.0"),
    ((0, json.dumps({"arch": "gfx1151", "sum": 4096.0, "pool": "GTT"}), ""), "ok",
     "a kernel ran on gfx1151; 1 GiB landed in GTT"),
    ((0, json.dumps({"arch": "gfx1100", "sum": 4096.0, "pool": None}), ""), "ok",
     "a kernel ran on gfx1100"),
])  # fmt: skip
def test_gpu_run(monkeypatch, outcome, status, said):
    # what the GPU's process came to, as it is read
    def run(args, **kwargs):
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        code, out, err = outcome
        return subprocess.CompletedProcess(args, code, out, err)

    monkeypatch.setattr(doctor.subprocess, "run", run)
    check = doctor.gpu_run(128)
    assert check.status == status and said in check.found


def test_clang(monkeypatch):
    # the system's clang, given its AMDGPU target, compiles a kernel for Strix Halo's GPU
    if shutil.which("clang") is None:
        pytest.skip("needs clang")
    check = doctor._clang()
    if check.status != "ok":
        pytest.skip(f"clang here: {check.found}")
    assert check.found.endswith("with the AMDGPU target")
    assert doctor._compile("gfx1151") == Check("compile", "ok", "a kernel for gfx1151")
    assert doctor._compile("gfx0").status == "fail"
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    assert doctor._clang().status == "fail"


def exit_status(*args: str) -> int:
    try:
        main(list(args))
    except SystemExit as e:
        return int(e.code or 0)
    return 0


@pytest.mark.parametrize("failed", [False, True])
def test_cli(monkeypatch, capsys, failed):
    # leatengine doctor prints the report, or its JSON, and exits 1 when a check failed
    found = [Check("kernel", "ok", "7.1.3"), Check("gpu", "fail" if failed else "ok", "gfx1151")]
    monkeypatch.setattr(doctor, "checks", lambda: found)
    assert exit_status("doctor", "--json") == int(failed)
    report = json.loads(capsys.readouterr().out)
    assert report["checks"][1]["status"] == ("fail" if failed else "ok")
    assert report["tinygrad"] and report["leatengine"]
    assert exit_status("doctor") == int(failed)
    text = capsys.readouterr().out
    assert ("1 problem." in text) == failed and ("Ready." in text) != failed


@pytest.mark.gpu
def test_gpu_run_on_the_gpu():
    # a kernel runs on an AMD GPU, or on tinygrad's emulated one, in the probe's own process
    dev = os.environ["DEV"]
    if not dev.split(":")[0].endswith("AMD"):
        pytest.skip("needs an AMD GPU, or DEV=MOCK+AMD")
    check = doctor.gpu_run(None, dev)
    assert check.status == "ok" and check.found.startswith("a kernel ran on gfx11"), check
