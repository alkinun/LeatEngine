"""Build and exercise wheel/sdist installs outside the checkout, with runtime dependencies only.

    uv run python scripts/check_package.py

Optionally test a published Git revision too: --git git+https://github.com/OWNER/REPO.git@SHA
"""

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]

# This code runs in a fresh installation, with no access to checkout modules through sys.path.
SMOKE = """
import importlib.metadata
import importlib.util
import json
import sys
import threading
import urllib.request
from pathlib import Path

import leatengine.cli

assert "tinygrad" not in sys.modules, "CLI import initialized the inference backend"
assert importlib.util.find_spec("leat") is None, "legacy package installed"
for name in ("agent", "box", "locales"):
    assert importlib.util.find_spec("leatengine." + name) is None, name
installed = Path(leatengine.cli.__file__).resolve()
assert installed.is_relative_to(Path(sys.prefix).resolve()), installed
assert importlib.metadata.version("leatengine") == sys.argv[2]

from leatengine.server import Server

with Server([Path(sys.argv[1])], port=0, max_context=32, prefill_chunk=8, slots=1) as server:
    server.load("tiny")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(url + "/v1/models", timeout=30) as response:
            model = json.load(response)["data"][0]
        assert model["status"] == "loaded" and model["owned_by"] == "leatengine", model
        body = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 4,
                "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
        request = urllib.request.Request(url + "/v1/chat/completions",
            json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            events = [line[6:].strip() for line in response if line.startswith(b"data: ")]
        assert events[-1] == b"[DONE]", events
        chunks = [json.loads(event) for event in events[:-1]]
        assert not any("error" in chunk for chunk in chunks), chunks
        assert any(chunk.get("usage", {}).get("completion_tokens", 0) > 0
                   for chunk in chunks), chunks
        with urllib.request.urlopen(url + "/metrics", timeout=30) as response:
            metrics = response.read().decode()
        assert "leatengine_completion_tokens_total" in metrics, metrics
    finally:
        server.shutdown()
        thread.join(timeout=10)
print("Installed runtime: package isolation, model loading, streaming and metrics passed")
"""


def run(args: list[str], cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(args, cwd=cwd, env=env, check=True, timeout=600)


def artifacts(directory: Path) -> list[Path]:
    (wheel,) = directory.glob("*.whl")
    (source,) = directory.glob("*.tar.gz")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "leatengine/cli.py" in names
        assert all(n.startswith(("leatengine/", f"leatengine-{VERSION}.dist-info/")) for n in names)
        assert all(n.endswith(".py") for n in names if n.startswith("leatengine/"))
        assert not any(
            n.startswith(("leatengine/agent/", "leatengine/box/", "leatengine/locales/"))
            for n in names
        )
    with tarfile.open(source) as archive:
        names = {
            Path(n).relative_to(f"leatengine-{VERSION}").as_posix() for n in archive.getnames()
        }
        for required in (
            "leatengine/cli.py",
            "tests/helpers.py",
            "scripts/validate_hardware.py",
            "scripts/check_package.py",
            "docs/development.md",
            "README.md",
            "LICENSE",
            "pyproject.toml",
            "uv.lock",
        ):
            assert required in names, required
        allowed = {
            "leatengine",
            "tests",
            "scripts",
            "docs",
            "README.md",
            "LICENSE",
            "pyproject.toml",
            "uv.lock",
            "PKG-INFO",
            ".gitignore",
        }
        assert all(n == "." or n.split("/")[0] in allowed for n in names), names
    return [wheel, source]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--git", help="also install and check this published Git revision")
    args = parser.parse_args()
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}
    }
    env |= {"DEV": "CPU", "PYTHONNOUSERSITE": "1"}
    with tempfile.TemporaryDirectory(prefix="leatengine-package-") as temporary:
        work = Path(temporary)
        env["CACHEDB"] = str(work / "compile.db")
        model = work / "tiny.gguf"
        # Only fixture creation runs in the development environment; installed probes use the
        # resulting data file, not tests or any source-tree code.
        generate = (
            "from pathlib import Path; import sys; "
            "from tests.helpers import write_tiny_model; "
            "write_tiny_model(Path(sys.argv[1]), 'llama')"
        )
        run([sys.executable, "-c", generate, str(model)], ROOT, env)
        dist = work / "dist"
        run(["uv", "build", "--out-dir", str(dist)], ROOT, env)
        targets = [str(p) for p in artifacts(dist)] + ([args.git] if args.git else [])
        for i, target in enumerate(targets):
            print(f"Checking {target}", flush=True)
            venv = work / f"runtime-{i}"
            run(["uv", "venv", "--python", sys.executable, str(venv)], work, env)
            python = str(venv / "bin/python")
            run(["uv", "pip", "install", "--python", python, target], work, env)
            for command in ([str(venv / "bin/leatengine")], [python, "-m", "leatengine"]):
                done = subprocess.run(
                    [*command, "--help"],
                    cwd=work,
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                assert "{run,serve,keys,bench,perplexity}" in done.stdout, done.stdout
            run([python, "-c", SMOKE, str(model), VERSION], work, env)
    print("All package checks passed", flush=True)


if __name__ == "__main__":
    main()
