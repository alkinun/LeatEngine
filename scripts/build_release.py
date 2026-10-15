"""Build LeatEngine for Linux x86_64 as one tarball that runs wherever it is unpacked.

    uv run python scripts/build_release.py

The tarball, dist/leatengine-VERSION-linux-x86_64.tar.gz, holds python-build-standalone's CPython
with LeatEngine and its locked dependencies installed into it, and bin/leatengine, which runs it:
it needs no network and no system Python. clang with its AMDGPU target stays the system's. A
commit builds the same bytes each time. The tarball is then unpacked elsewhere and checked: its
launcher, the package's isolation, and a streamed completion from generated CPU weights, served
without a network where a network namespace can be made.
"""

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
NAME = f"leatengine-{VERSION}-linux-x86_64"

# python-build-standalone's CPython, the build of it uv installs; a bump takes the SHA-256 from
# the release's SHA256SUMS
PYTHON = "3.14.6+20260728"
PYTHON_SHA256 = "d6254462d5fe066810a472d286861f6a18f66400008bfa9688e202a5672e88cb"
PYTHON_FILE = f"cpython-{PYTHON}-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
PYTHON_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/"
    f"{PYTHON.split('+')[1]}/{PYTHON_FILE.replace('+', '%2B')}"
)
LIB = "lib/python3.14"

# of python/, what serving has no use for: Tk and IDLE, pip, and what builds or embeds Python,
# as its headers and libpython, which the interpreter has linked in
TRIM = (
    "include", "share/man", "lib/libpython3*", "lib/pkgconfig",
    "lib/itcl*", "lib/libtcl*", "lib/tcl*", "lib/thread*", "lib/tk*",
    f"{LIB}/config-*", f"{LIB}/ensurepip", f"{LIB}/idlelib", f"{LIB}/lib-dynload/_tkinter*",
    f"{LIB}/site-packages/pip*", f"{LIB}/tkinter", f"{LIB}/turtle.py", f"{LIB}/turtledemo",
)  # fmt: skip

# runs the Python beside it, wherever the tree is unpacked and through any links to it, isolated
# from the environment's: no PYTHONPATH, user site-packages or working directory
LAUNCHER = """\
#!/bin/sh
root=$(dirname -- "$(dirname -- "$(realpath -- "$0")")")
exec "$root/python/bin/python3" -I -m leatengine "$@"
"""

# as a supervisor, such as Lemonade, would run it: started by its launcher, ready at /health
SERVE = """
import json
import signal
import socket
import subprocess
import sys
import time
import urllib.request

with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
server = subprocess.Popen([sys.argv[1], "serve", sys.argv[2], "--port", str(port),
                           "--max-context", "32", "--slots", "1"])
url = f"http://127.0.0.1:{port}"
try:
    deadline = time.monotonic() + 300
    while True:
        assert server.poll() is None, f"the server exited with {server.returncode}"
        assert time.monotonic() < deadline, "the server was not ready in time"
        try:
            with urllib.request.urlopen(url + "/health", timeout=10) as response:
                break
        except OSError:  # refused until it listens, then 503 until the model is warmed up
            time.sleep(0.5)
    body = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 4,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    request = urllib.request.Request(url + "/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        events = [line[6:].strip() for line in response if line.startswith(b"data: ")]
    assert events[-1] == b"[DONE]", events
    chunks = [json.loads(event) for event in events[:-1]]
    assert not any("error" in chunk for chunk in chunks), chunks
    assert any(chunk.get("usage", {}).get("completion_tokens", 0) > 0 for chunk in chunks), chunks
finally:
    server.send_signal(signal.SIGINT)
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server.kill()
        raise
assert server.returncode == 0, f"the server exited with {server.returncode} on SIGINT"
print("Launcher: serve answered /health, streamed a completion, and quit on SIGINT")
"""


def run(args: list[str], cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(args, cwd=cwd, env=env, check=True, timeout=600)


def output(args: list[str], cwd: Path, env: dict[str, str]) -> str:
    done = subprocess.run(args, cwd=cwd, env=env, check=True, timeout=600, capture_output=True)
    return done.stdout.decode()


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def environment() -> dict[str, str]:
    # without the development environment's Python
    drop = {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}
    return {k: v for k, v in os.environ.items() if k not in drop}


def python_archive() -> Path:
    # downloaded once, and checked at each build
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "leatengine-release"
    path = cache / PYTHON_FILE
    if not path.exists() or sha256(path) != PYTHON_SHA256:
        print(f"Downloading {PYTHON_URL}", flush=True)
        cache.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        with urllib.request.urlopen(PYTHON_URL, timeout=60) as response, partial.open("wb") as f:
            shutil.copyfileobj(response, f)
        if (found := sha256(partial)) != PYTHON_SHA256:
            partial.unlink()
            raise SystemExit(f"{PYTHON_URL} has SHA-256 {found}, not {PYTHON_SHA256}")
        partial.replace(path)
    return path


def build(out: Path) -> Path:
    """Builds the tarball in `out`, returning its path."""
    env = environment()
    commit = output(["git", "rev-parse", "HEAD"], ROOT, env).strip()
    packaged = ("leatengine", "pyproject.toml", "uv.lock", "LICENSE")
    if output(["git", "status", "--porcelain", "--", *packaged], ROOT, env):
        print(f"Building from uncommitted changes to {commit[:9]}", flush=True)
        commit += "-dirty"
    # files' times are the commit's
    log = ["git", "log", "-1", "--format=%ct"]
    epoch = int(env.get("SOURCE_DATE_EPOCH") or output(log, ROOT, env))
    env["SOURCE_DATE_EPOCH"] = str(epoch)
    # the build backends as they were at the commit, so that later releases change nothing
    newest = datetime.fromtimestamp(epoch, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".build-", dir=out) as temporary:
        work = Path(temporary)
        tree = work / NAME
        with tarfile.open(python_archive()) as archive:
            archive.extractall(tree, filter="data")
        python = tree / "python/bin/python3"

        print(f"Installing LeatEngine {VERSION} ({commit[:9]}) into CPython {PYTHON}", flush=True)
        requirements = work / "requirements.txt"
        requirements.write_text(output(
            ["uv", "export", "--locked", "--no-dev", "--no-emit-project", "--no-header",
             "--format", "requirements.txt"], ROOT, env,
        ))  # fmt: skip
        run(["uv", "build", "--wheel", "--exclude-newer", newest, "-o", str(work)], ROOT, env)
        (wheel,) = work.glob(f"leatengine-{VERSION}-*.whl")
        # nothing from uv's cache, which may hold tinygrad's commit as a newer setuptools built it,
        # but downloaded and built as on a clean machine; wheels for glibc 2.28, as Pillow's need,
        # whatever the building machine's
        run(["uv", "pip", "install", "--python", str(python), "--no-deps", "--no-cache",
             "--link-mode", "copy", "--exclude-newer", newest,
             "--python-platform", "x86_64-manylinux_2_28",
             "-r", str(requirements), str(wheel)], work, env)  # fmt: skip

        for pattern in TRIM:
            for path in (tree / "python").glob(pattern):
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        for path in (tree / "python/bin").iterdir():  # the interpreter, not scripts, whose
            if path.name not in ("python", "python3", "python3.14"):  # first lines hold a path
                path.unlink()
        # what uv records of this build alone: the wheel's path, and when it was cached
        site = tree / "python" / LIB / "site-packages"
        for path in [*site.glob("leatengine-*.dist-info/direct_url.json"),
                     *site.glob("*.dist-info/uv_cache.json")]:  # fmt: skip
            path.unlink()
        for record in site.glob("*.dist-info/RECORD"):  # of the files kept
            lines = record.read_text().splitlines(keepends=True)
            kept = [x for x in lines if os.path.lexists(site / x.rsplit(",", 2)[0])]
            record.write_text("".join(kept))

        # bytecode checked against its source's hash rather than its time, which an unpacking may
        # not keep, and named relative to the tree; one hash seed orders each build's sets alike.
        # -B, lest compileall's own imports be cached as Python caches them, by time and path.
        for cache in list(tree.rglob("__pycache__")):
            shutil.rmtree(cache)
        run([str(python), "-B", "-s", "-P", "-m", "compileall", "-f", "-q", "-j0",
             "--invalidation-mode", "checked-hash", "-d", f"python/{LIB}", str(site.parent)],
            work, {"PYTHONHASHSEED": "0"})  # fmt: skip

        (tree / "bin").mkdir()
        (tree / "bin/leatengine").write_text(LAUNCHER)
        (tree / "bin/leatengine").chmod(0o755)
        shutil.copy(ROOT / "LICENSE", tree / "LICENSE")
        pin = re.search(r"tinygrad\.git@([0-9a-f]{40})", (ROOT / "pyproject.toml").read_text())
        assert pin is not None
        build_info = {"leatengine": VERSION, "commit": commit, "python": PYTHON,
                      "tinygrad": pin[1]}  # fmt: skip
        (tree / "BUILD-INFO.json").write_text(json.dumps(build_info, indent=2) + "\n")

        # written by the tree's own Python, so that its zlib, not the building machine's,
        # compresses it
        tarball = out / f"{NAME}.tar.gz"
        run([str(python), "-I", "-B", __file__, "--archive", str(tree), str(tarball), str(epoch)],
            work, {})  # fmt: skip
        unpacked = sum(p.lstat().st_size for p in tree.rglob("*"))
    digest = sha256(tarball)
    (out / f"{tarball.name}.sha256").write_text(f"{digest}  {tarball.name}\n")
    size = tarball.stat().st_size
    print(f"Built {tarball}: {size / 2**20:.1f} MiB ({size} bytes), {unpacked / 2**20:.0f} MiB "
          f"unpacked, SHA-256 {digest}", flush=True)  # fmt: skip
    return tarball


def archive(tree: Path, tarball: Path, epoch: int) -> None:
    """Writes `tree` to `tarball` the same each time: in name order, of one time and owner, of
    modes 755 or 644, by gzip of no time or name."""
    partial = tarball.with_name(tarball.name + ".part")
    with (
        partial.open("wb") as raw,
        gzip.GzipFile("", "wb", 9, raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar,
    ):
        for directory, directories, files in os.walk(tree):
            directories.sort()  # walked in this order; links to directories are added, not walked
            here = Path(directory)
            for path in [here] + [here / n for n in sorted(directories + files)]:
                if path.is_dir() and not path.is_symlink() and path != here:
                    continue  # added when walked
                info = tar.gettarinfo(path, path.relative_to(tree.parent).as_posix())
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = epoch
                if not info.issym():
                    info.mode = 0o755 if info.isdir() or info.mode & 0o100 else 0o644
                if info.isreg():
                    with path.open("rb") as f:
                        tar.addfile(info, f)
                else:
                    tar.addfile(info)
    partial.replace(tarball)


def offline() -> list[str]:
    """A prefix that runs a command in a network namespace of its own, with only loopback, or
    none where one cannot be made."""
    prefix = ["unshare", "--user", "--map-root-user", "--net",
              "sh", "-c", 'ip link set lo up && exec "$@"', "sh"]  # fmt: skip
    try:
        if subprocess.run([*prefix, "true"], capture_output=True, timeout=30).returncode == 0:
            return prefix
    except FileNotFoundError:
        pass
    print("No network namespace can be made here: checking with the network", flush=True)
    return []


def check(tarball: Path) -> None:
    """Unpacks `tarball` elsewhere and runs it with neither the checkout nor a network."""
    from check_package import SMOKE

    print(f"Checking {tarball}", flush=True)
    proxies = {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}
    env = {k: v for k, v in environment().items() if k.lower() not in proxies}
    with tempfile.TemporaryDirectory(prefix="leatengine-release-") as temporary:
        work = Path(temporary)
        model = work / "tiny.gguf"
        generate = (
            "from pathlib import Path; import sys; "
            "from tests.helpers import write_tiny_model; "
            "write_tiny_model(Path(sys.argv[1]), 'llama')"
        )
        run([sys.executable, "-c", generate, str(model)], ROOT, env)
        unpacked = work / "un packed"  # a space, for the launcher's quoting
        unpacked.mkdir()
        run(["tar", "-xzf", str(tarball), "-C", str(unpacked)], work, env)
        tree = unpacked / NAME
        link = work / "bin/leatengine"  # as a user's own bin/ would link to it
        link.parent.mkdir()
        link.symlink_to(tree / "bin/leatengine")
        # the Python settings the launcher's must not read
        env |= {"DEV": "CPU", "CACHEDB": str(work / "compile.db"),
                "PYTHONHOME": str(work / "none"), "PYTHONPATH": str(ROOT)}  # fmt: skip
        for launcher in (tree / "bin/leatengine", link):
            usage = output([str(launcher), "--help"], work, env)
            assert usage.startswith("usage: leatengine") and "serve" in usage, usage
        # the machine's check reports from the tarball, of the tinygrad built into it, and exits 1
        # on a machine without an AMD GPU rather than failing to run
        done = subprocess.run([str(tree / "bin/leatengine"), "doctor", "--json"], cwd=work,
                              env=env, capture_output=True, text=True, timeout=120)  # fmt: skip
        report, built = json.loads(done.stdout), json.loads((tree / "BUILD-INFO.json").read_text())
        assert report["tinygrad"] == built["tinygrad"][:9], done.stdout + done.stderr
        assert report["leatengine"] == VERSION and report["checks"], done.stdout
        prefix = offline()
        python = str(tree / "python/bin/python3")
        run([*prefix, python, "-I", "-c", SMOKE, str(model), VERSION], work, env)
        run([*prefix, python, "-I", "-c", SERVE, str(tree / "bin/leatengine"), str(model)],
            work, env)  # fmt: skip
    print("The unpacked tarball passed its checks", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--no-check", action="store_true", help="build without checking")
    parser.add_argument(
        "--reproducible", action="store_true",
        help="build again in another directory, and require the same bytes",
    )  # fmt: skip
    parser.add_argument("--archive", nargs=3, help=argparse.SUPPRESS)  # in the tree's Python
    args = parser.parse_args()
    if args.archive:
        return archive(Path(args.archive[0]), Path(args.archive[1]), int(args.archive[2]))
    tarball = build(args.out_dir.resolve())
    if args.reproducible:
        with tempfile.TemporaryDirectory(prefix=".again-", dir=tarball.parent) as again:
            if sha256(build(Path(again))) != sha256(tarball):
                raise SystemExit("A second build differs from the first")
        print("A second build, elsewhere, gave the same bytes", flush=True)
    if not args.no_check:
        check(tarball)


if __name__ == "__main__":
    main()
