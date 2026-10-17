"""Day one on a new Strix Halo machine as one command: the bring-up's steps in order, each checked
before the next, into one Markdown report in reports/, or --out, printed too. It stops at the
first step that fails, and says what to do.

    uv run python scripts/bring_up.py SMALL.gguf [MODEL.gguf ...] --llama-cpp BIN [BIN ...] \\
        --wikitext wiki.test.raw [--ceiling MODEL.gguf]

1. `leatengine doctor`: the host is ready;
2. the device is gfx1151;
3. the GPU tests: first the attention groups', of heads up to GLM's 576 wide, then the rest, the
   small model's against the first llama.cpp build among them;
4. the memory ceiling: the largest model, or --ceiling, served at the context it fits to memory
   and answering a chat, while GTT and VRAM are watched;
5. agreement and speed, as validate_hardware.py measures them, of the small model and then of
   each other, the large ones; and the server answering a chat of the small model.

DEV picks the device, AMD by default. Elsewhere, on NV or the emulator's MOCK+AMD, doctor's
verdict and the arch are recorded but not checked, and the ceiling, an APU's, is skipped; without
models, so are 4 and 5.
"""

import argparse
import contextlib
import json
import os
import re
import sys
import threading
from pathlib import Path

from validate_hardware import (
    ROOT,
    Report,
    _ggufs,
    _is_drafter,
    _last_line,
    _machine,
    _model,
    _run,
    _server,
    _tests,
)

TARGET = "gfx1151"
# what step 5 runs of each llama.cpp build, as scripts/build_llama_cpp.py builds them
TOOLS = ("llama-bench", "llama-batched-bench", "llama-perplexity")


class Stop(Exception):
    """A step failed: what to do about it."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "models", nargs="*", type=Path,
        help="GGUF files: a small model that fits easily first, then large ones",
    )  # fmt: skip
    parser.add_argument("--llama-cpp", type=Path, nargs="+", default=[], help="llama.cpp builds")
    parser.add_argument("--wikitext", type=Path, help="wikitext-2's wiki.test.raw")
    parser.add_argument("--ceiling", type=Path, help="the model to load near the memory's limit")
    parser.add_argument("--out", type=Path, default=ROOT / "reports", help="report directory")
    args = parser.parse_args()
    if args.models and not (args.llama_cpp and args.wikitext):
        parser.error("models need --llama-cpp and --wikitext, to measure agreement")
    # what the last step needs, checked before the first, which takes many minutes
    for build in args.llama_cpp:
        if missing := [tool for tool in TOOLS if not (build / tool).is_file()]:
            said = ", ".join(missing)
            parser.error(f"{build} has no {said}: scripts/build_llama_cpp.py builds them")
    if args.wikitext and not args.wikitext.is_file():
        parser.error(f"there is no {args.wikitext}")
    # as validate_hardware.py's arguments: every measure, the drafters apart
    found, args.quick = _ggufs(args.models), False
    args.models = [m for m in found if not _is_drafter(m)]
    args.drafters = [m for m in found if _is_drafter(m)]
    env = os.environ | {"DEV": os.environ.get("DEV", "AMD")}
    report = Report(env["DEV"], "bring-up")
    steps = (
        ("doctor", _doctor), ("arch", _arch), ("tests", _gpu_tests),
        ("ceiling", _ceiling), ("models", _models),
    )  # fmt: skip
    try:
        for name, step in steps:
            print(f"== {name}", file=sys.stderr, flush=True)
            step(env, args, report)
    except Stop as stop:
        report.section("stopped", f"## Stopped at {name}\n\n{stop}", str(stop))
    path = report.write(args.out)
    print(report.text())
    print(f"\nwritten to {path}", file=sys.stderr)
    sys.exit(1 if "stopped" in report.sections else 0)


def _doctor(env: dict, args: argparse.Namespace, report: Report) -> None:
    # the host's checks, of an AMD GPU's: elsewhere recorded only
    done = _run([sys.executable, "-m", "leatengine", "doctor", "--json"], env, 600, cwd=ROOT)
    try:
        found = json.loads(done["out"].strip().splitlines()[-1])
    except (IndexError, ValueError):
        raise Stop(f"`leatengine doctor --json` gave no report: `{_last_line(done)}`") from None
    failed = [c for c in found["checks"] if c["status"] == "fail"]
    lines = [f"- {c['status']}: {c['name']}: {c['found']}" for c in found["checks"]]
    checked = env["DEV"] == "AMD"
    verdict = "FAILED" if failed else "ready"
    if not checked:
        verdict = f"recorded, not checked on {env['DEV']}"
    report.section("doctor", f"## Doctor: {verdict}\n\n" + "\n".join(lines), found)
    if checked and failed:
        fixes = [f"- {c['name']}: {c['fix'] or c['found']}" for c in failed]
        raise Stop("Do what doctor says, then run this again:\n\n" + "\n".join(fixes))


def _arch(env: dict, args: argparse.Namespace, report: Report) -> None:
    # the device tinygrad opens, which on AMD must be Strix Halo's
    info = _machine(env)
    report.machine(info)
    if env["DEV"] == "AMD" and info["arch"] != TARGET:
        raise Stop(
            f"The GPU is {info['arch']}, not {TARGET}: on another APU or GPU the kernels' paths "
            "and tiles are not the ones measured. Check DEV and the GPU doctor found."
        )


def _gpu_tests(env: dict, args: argparse.Namespace, report: Report) -> None:
    # the attention groups' first, then the rest, with the small model's against llama.cpp
    first = _tests(env, "-k", "attention_groups")
    models = {"LEATENGINE_MODEL": str(args.models[0]), "LLAMA_CPP": str(args.llama_cpp[0]),
              "WIKITEXT": str(args.wikitext)} if args.models else {}  # fmt: skip
    rest = _tests(env | models, "-k", "not attention_groups")
    combined = {
        "ok": first["ok"] and rest["ok"],
        "summary": f"attention groups: {first['summary']}; the rest: {rest['summary']}",
        "failures": first["failures"] + rest["failures"],
        "seconds": first["seconds"] + rest["seconds"],
    }
    report.tests(combined)
    if not combined["ok"]:
        raise Stop(
            "Run each failing test alone, `uv run pytest -m gpu -x TEST`, with DEV as here; "
            "docs/development.md has how the GPU tests run."
        )


def _ceiling(env: dict, args: argparse.Namespace, report: Report) -> None:
    # the largest model, or the one given, served at the context that fills the memory its
    # weights leave, while an AMD GPU's pools are watched
    if env["DEV"] != "AMD" or not args.models:
        why = "no models given" if not args.models else f"GTT is an AMD APU's, not {env['DEV']}'s"
        report.section("ceiling", f"## Memory ceiling: skipped, {why}", None)
        return
    model = args.ceiling or max(args.models, key=_size)
    pools = _pools()
    peaks = dict.fromkeys(pools, 0)
    done = threading.Event()

    def watch() -> None:
        while not done.wait(0.5):
            for name, used in pools.items():
                with contextlib.suppress(OSError, ValueError):
                    peaks[name] = max(peaks[name], int(used.read_text()))

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    result = _server(env, model, args.out)
    done.set()
    watcher.join()
    log = (args.out / ".server.log").read_text(errors="replace").splitlines()
    result["context"] = next((line for line in log if ": a context of " in line), "not said")
    totals = {name: _read(used.with_name(used.name.replace("_used", "_total")))
              for name, used in pools.items()}  # fmt: skip
    seen = ", ".join(f"{name} {peaks[name] / 2**30:.1f} of {totals[name] / 2**30:.1f} GiB"
                     for name in pools)  # fmt: skip
    size = f"{model.name}, {_size(model) / 2**30:.1f} GiB"
    raw = {"model": model.name, "peaks": peaks, "totals": totals, "server": result}
    text = f"{size}; {result['context']}; the most in use, {seen}"
    if not result["ok"]:
        text += f"\n\n`{result.get('error', 'no reply')}`"
        report.section("ceiling", f"## Memory ceiling: FAILED\n\n{text}", raw)
        raise Stop(
            f"The model did not load and answer: the server's log is {args.out / '.server.log'}. "
            "If an allocation failed while GTT had room, look for KFD's own limits in `dmesg`; "
            "if GTT was full, raise TTM's limit as AMD's Strix Halo guide says "
            "(docs/strix-halo.md), or choose a smaller --ceiling."
        )
    report.section("ceiling", f"## Memory ceiling: answered\n\n{text}", raw)


def _models(env: dict, args: argparse.Namespace, report: Report) -> None:
    # agreement and speed of the small model first, then of each other; then the server's chat
    if not args.models:
        return
    for model in args.models:
        print(f"== {model.name}", file=sys.stderr, flush=True)
        report.model(row := _model(env, model, args, args.drafters))
        if failed := report.failures(row):
            raise Stop(
                f"{model.name}'s runs failed: {', '.join(failed)}. The errors are above: run each "
                "alone, as validate_hardware.py does, to see it whole."
            )
    report.server(server := _server(env, args.models[0], args.out))
    if not server["ok"]:
        raise Stop(f"The server did not answer: its log is {args.out / '.server.log'}.")


def _pools() -> dict[str, Path]:
    # the first AMD GPU's memory pools' use, as amdgpu reports them
    for device in sorted(Path("/sys/class/drm").glob("renderD*/device")):
        if (device / "mem_info_gtt_used").exists():
            return {pool: device / f"mem_info_{pool}_used" for pool in ("gtt", "vram")}
    return {}


def _read(path: Path) -> int:
    with contextlib.suppress(OSError, ValueError):
        return int(path.read_text())
    return 0


def _size(model: Path) -> int:
    # a model's bytes, of all its parts if it is split
    if (split := re.match(r"(.*)-00001-of-(\d+)\.gguf$", model.name)) is None:
        return model.stat().st_size
    return sum(p.stat().st_size for p in model.parent.glob(f"{split.group(1)}-*-of-*.gguf"))


if __name__ == "__main__":
    main()
