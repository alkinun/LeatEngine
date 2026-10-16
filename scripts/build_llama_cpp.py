"""Builds llama.cpp at one commit for each AMD backend LeatEngine is measured against on Strix Halo:
Vulkan, which Mesa's RADV runs, and HIP, of ROCm with rocWMMA's flash attention, for gfx1151.

    uv run python scripts/build_llama_cpp.py [--commit 1b70350] [--backends vulkan hip]

Each build's tools go to DIR/build-BACKEND/bin, which scripts/validate_hardware.py's --llama-cpp
takes, several at once, reporting the better per model and measure. The source is a clone of
llama.cpp in DIR/src, fetched at the commit, or --source, a checkout of one's own, as it is.
--dry-run says the commands without running them.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

REPOSITORY = "https://github.com/ggml-org/llama.cpp"
COMMIT = "1b70350"  # docs/benchmarks.md's, the first tested build that runs Muse Glimmer
TARGETS = ("llama-bench", "llama-batched-bench", "llama-perplexity", "llama-cli")
# each backend's CMake options: Vulkan's, and HIP's of the GPU's target, with rocWMMA's FA
OPTIONS = {
    "vulkan": ["-DGGML_VULKAN=ON"],
    "hip": ["-DGGML_HIP=ON", "-DGGML_HIP_ROCWMMA_FATTN=ON"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--commit", default=COMMIT, help=f"the commit to build, {COMMIT} by default"
    )
    parser.add_argument("--backends", nargs="+", choices=OPTIONS, default=list(OPTIONS))
    parser.add_argument("--target", default="gfx1151", help="HIP's GPU target, Strix Halo's")
    parser.add_argument("--dir", type=Path, default=Path.home() / ".cache/leatengine/llama.cpp")
    parser.add_argument("--source", type=Path, help="a llama.cpp checkout to build as it is")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--dry-run", action="store_true", help="say the commands, run none")
    args = parser.parse_args()
    run = _said if args.dry_run else _run
    source = args.source or _fetched(args.dir / "src", args.commit, run)
    built = {}
    for backend in args.backends:
        build = args.dir / f"build-{backend}"
        run(commands(backend, source, build, args.target, args.jobs), _environment(backend))
        built[backend] = str(build / "bin")
    if not args.dry_run:
        (args.dir / "builds.json").write_text(json.dumps({"commit": args.commit} | built, indent=1))
    print("\n".join(built.values()))


def commands(backend: str, source: Path, build: Path, target: str, jobs: int) -> list[list[str]]:
    """The CMake commands that configure and build a backend's tools of `source` in `build`."""
    options = OPTIONS[backend] + ([f"-DAMDGPU_TARGETS={target}"] if backend == "hip" else [])
    configure = ["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                 "-DLLAMA_CURL=OFF", *options]  # fmt: skip
    return [configure, ["cmake", "--build", str(build), "-j", str(jobs), "--target", *TARGETS]]


def _environment(backend: str) -> dict[str, str]:
    # HIP's compiler and ROCm's path, as llama.cpp's build docs have them, from hipconfig
    env = os.environ.copy()
    if backend == "hip" and shutil.which("hipconfig"):
        for key, flag in (("HIP_PATH", "-R"), ("HIPCXX", "-l")):
            out = subprocess.run(["hipconfig", flag], capture_output=True, text=True).stdout
            env.setdefault(key, out.strip() + ("/clang" if key == "HIPCXX" else ""))
    return env


def _fetched(src: Path, commit: str, run: Callable[..., None]) -> Path:
    # llama.cpp's source at the commit, cloned once and fetched since
    if not (src / ".git").exists():
        run([["git", "clone", REPOSITORY, str(src)]], os.environ.copy())
    run([["git", "-C", str(src), "fetch", "origin"], ["git", "-C", str(src), "checkout",
         "--detach", commit]], os.environ.copy())  # fmt: skip
    return src


def _run(commands: list[list[str]], env: dict[str, str]) -> None:
    for command in commands:
        print("+ " + " ".join(command), file=sys.stderr, flush=True)
        subprocess.run(command, env=env, check=True)


def _said(commands: list[list[str]], env: dict[str, str]) -> None:
    for command in commands:
        print(" ".join(command))


if __name__ == "__main__":
    main()
