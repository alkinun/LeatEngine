"""Checks a machine end to end in one command, on a new inference host: the GPU tests, agreement
with llama.cpp and speed against it on the same files, speculative decoding over chat prompts,
and the server answering a streamed chat. Writes a Markdown report to reports/, or --out, and
prints it.

    uv run python scripts/validate_hardware.py MODEL.gguf|DIR ... [--llama-cpp BIN [BIN ...]]
        [--wikitext wiki.test.raw]

Every model runs in a process of its own, so each has the GPU's memory to itself. DEV picks the
device, AMD by default for Strix Halo. A drafter is found for each model of an MTP layer
of its own, as Qwen3.5's and Qwen3.6's, in its file or an `mtp-*.gguf` beside it, and for Gemma 4,
a Gemma 4 assistant's GGUF among the files given. --quick runs fewer repetitions and skips the
batched and chat runs. --llama-cpp takes several builds' bin directories, as of Vulkan and of HIP
from scripts/build_llama_cpp.py: each runs, and the report gives the better per model and measure,
and which build it was.

Agreement, given wikitext-2's test set (`wiki.test.raw`, as llama.cpp's
scripts/get-wikitext-2.sh fetches it) and llama.cpp: each build's logits of its first 8 chunks of
512 tokens, from `llama-perplexity --kl-divergence-base`, and LeatEngine's KL divergence from them
and top-1 agreement, as prompts, one token at a time as generation runs, and with a q8_0 KV cache.
scripts/agreement.py measures it alone.
"""

import argparse
import contextlib
import datetime
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from leatengine.draft import mtp, own_drafter
from leatengine.gguf import GGUF

ROOT = Path(__file__).resolve().parent.parent

# a reply to this, streamed from the server, shows the model loads, generates and detokenizes
SERVER_PROMPT = "Name three primary colors, one per line."

# how LeatEngine scores llama.cpp's logits: the flags of `leatengine perplexity` of each way
AGREEMENT = {"prompts": (), "one at a time": ("--decode",), "q8_0 cache": ("--cache-type", "q8_0")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("models", nargs="+", type=Path, help="GGUF files, or directories of them")
    parser.add_argument(
        "--llama-cpp", type=Path, nargs="+", default=[],
        help="llama.cpp's build/bin, or several builds', to compare against the better of",
    )  # fmt: skip
    parser.add_argument(
        "--wikitext", type=Path, help="wikitext-2's wiki.test.raw: with --llama-cpp, agreement too"
    )
    parser.add_argument("--quick", action="store_true", help="one repetition, no batch or chat")
    parser.add_argument("--no-tests", action="store_true", help="skip the GPU tests")
    parser.add_argument("--out", type=Path, default=ROOT / "reports", help="report directory")
    args = parser.parse_args()
    env = os.environ | {"DEV": os.environ.get("DEV", "AMD")}
    models = [m for m in _ggufs(args.models) if not _is_drafter(m)]
    drafters = [m for m in _ggufs(args.models) if _is_drafter(m)]
    report = Report(env["DEV"])
    report.machine(_machine(env))
    if not args.no_tests:
        report.tests(_tests(env))
    for model in models:
        print(f"== {model.name}", file=sys.stderr, flush=True)
        report.model(_model(env, model, args, drafters))
    if models:
        report.server(_server(env, models[0], args.out))
    path = report.write(args.out)
    print(report.text())
    print(f"\nwritten to {path}", file=sys.stderr)


def _model(env: dict, model: Path, args: argparse.Namespace, drafters: list[Path]) -> dict:
    """A model's row: agreement given args.wikitext, speed against args.llama_cpp's builds, and
    unless args.quick, of 4 sequences at once and of chat replies with the model's drafter."""
    row: dict = {"model": model.stem}
    reps = "1" if args.quick else "3"
    if args.wikitext and args.llama_cpp:
        row["agreement"] = _agreement(env, model, args.llama_cpp, args.wikitext, args.out)
    row["leatengine"] = _leatengine(env, "bench", model, "-r", reps)
    if args.llama_cpp:
        row["llama.cpp"] = _best({_name(d): _llama_bench(d, model, reps) for d in args.llama_cpp})
    if not args.quick:
        row["leatengine 4"] = _leatengine(env, "bench", model, "-r", reps, "-s", "4", "-p", "64")
        if args.llama_cpp:
            row["llama.cpp 4"] = _best({_name(d): _llama_batched(d, model) for d in args.llama_cpp})
        draft = _drafter(model, drafters + _ggufs([model.parent]))
        if draft is not None:
            chat = ("bench", model, "--chat", "-n", "256", "-r", "2")
            row["chat"] = _leatengine(env, *chat)
            row["chat draft"] = _leatengine(env, *chat, "--draft", draft)
            row["chat draft 2"] = _leatengine(env, *chat, "--draft", draft, "-s", "2")
            row["chat 2"] = _leatengine(env, *chat, "-s", "2")
    return row


class Report:
    """The report's sections, in Markdown, and the raw results at its end."""

    # in this order, bring_up.py's steps among them
    ORDER = ("machine", "doctor", "tests", "ceiling", "agreement", "speed", "errors", "server",
             "stopped")  # fmt: skip

    def __init__(self, device: str, kind: str = "validate"):
        self.device, self.kind = device, kind
        self.sections, self.raw = dict[str, str](), dict[str, object]()
        self.rows: list[dict] = []
        self.started = datetime.datetime.now()

    def machine(self, info: dict) -> None:
        self.raw["machine"] = info
        lines = [f"- {k}: {v}" for k, v in info.items()]
        self.sections["machine"] = "## Machine\n\n" + "\n".join(lines)

    def tests(self, result: dict) -> None:
        self.raw["tests"] = result
        verdict = "passed" if result["ok"] else "FAILED"
        text = f"## GPU tests: {verdict}\n\n`{result['summary']}` in {result['seconds']:.0f} s"
        if result["failures"]:
            text += "\n\n" + "\n".join(f"- {f}" for f in result["failures"])
        self.sections["tests"] = text

    def section(self, key: str, text: str, raw: object) -> None:
        """Another script's section, as bring_up.py's of its steps."""
        self.sections[key], self.raw[key] = text, raw

    def model(self, row: dict) -> None:
        self.rows.append(row)
        self.raw.setdefault("models", []).append(row)  # type: ignore[union-attr]

    def server(self, result: dict) -> None:
        self.raw["server"] = result
        verdict = "answered" if result["ok"] else "FAILED"
        detail = ", ".join(f"{k} {v}" for k, v in result.items() if k not in ("ok", "reply"))
        reply = result.get("reply", "").strip().replace("\n", " / ")[:300]
        self.sections["server"] = f"## Server: {verdict}\n\n{detail}\n\n> {reply}"

    def text(self) -> str:
        head = f"# leatengine on {self.device}, {self.started:%Y-%m-%d %H:%M}"
        raw = "## Raw\n\n```json\n" + json.dumps(self.raw, indent=1) + "\n```\n"
        if any("agreement" in r for r in self.rows):
            self.sections["agreement"] = self._agreement()
        if any("leatengine" in r for r in self.rows):
            self.sections["speed"] = self._speed()
        if errors := self._errors():
            self.sections["errors"] = "## Errors\n\n" + "\n".join(errors)
        return "\n\n".join([head, *(self.sections[k] for k in self.ORDER if k in self.sections),
                             raw])  # fmt: skip

    def write(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        host = socket.gethostname()
        path = directory / f"{self.kind}-{host}-{self.device}-{self.started:%Y%m%d-%H%M}.md"
        path.write_text(self.text())
        return path

    def _agreement(self) -> str:
        # of each model and build of llama.cpp, both perplexities, and KL and top-1 of each way
        def way(result: dict) -> str:
            if "kl_mean" not in result:
                return ""
            return f"{result['kl_mean']:.4f}, p99 {result['kl_p99']:.4f}, {result['top1']:.1%}"

        def perplexity(result: dict) -> str:
            return f"{result['perplexity']:.3f}" if "perplexity" in result else ""

        lines = [
            "## Agreement with llama.cpp\n",
            "wikitext-2's first 8 chunks of 512 tokens: both engines' perplexity, and LeatEngine's "
            "KL divergence from each llama.cpp build's next-token distributions, mean and 99th "
            "percentile, and how often both pick the same token. Lower KL is closer.\n",
            "| | llama.cpp | llama.cpp perplexity | LeatEngine perplexity | "
            f"{' | '.join(AGREEMENT)} |",
            "|---|---|---:|---:" + "|---:" * len(AGREEMENT) + "|",
        ]
        for r in self.rows:
            for build, result in r.get("agreement", {}).items():
                ways = [result.get(name, {}) for name in AGREEMENT]
                cells = [f"{build} {result.get('commit', '')}", perplexity(result),
                         perplexity(ways[0]), *map(way, ways)]  # fmt: skip
                lines.append(f"| {r['model']} | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def _speed(self) -> str:
        # tokens per second: one sequence, four at once, and chat replies speculative or not
        def at(row: dict, key: str, field: str) -> str:
            # of several builds of llama.cpp, the better's, and its name
            value, by = (row.get(key) or {}).get(field), (row.get(key) or {}).get(f"{field} by")
            return "" if value is None else f"{value:.1f}" + (f" ({by})" if by else "")

        lines = [
            "## Speed, tokens per second\n",
            "| | llama.cpp pp512 | leatengine pp512 | llama.cpp tg128 | leatengine tg128 "
            "| llama.cpp 4 at once | leatengine 4 at once |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in self.rows:
            lines.append(
                f"| {r['model']} | {at(r, 'llama.cpp', 'prefill')} "
                f"| {at(r, 'leatengine', 'prefill')} "
                f"| {at(r, 'llama.cpp', 'decode')} | {at(r, 'leatengine', 'decode')} "
                f"| {at(r, 'llama.cpp 4', 'decode')} | {at(r, 'leatengine 4', 'decode')} |"
            )
        chats = [r for r in self.rows if r.get("chat")]
        if chats:
            lines += [
                "\nChat replies, greedy, 256 tokens each, without a drafter and with one, and the "
                "tokens a reply takes per step:\n",
                "| | plain | speculative | tokens a step | plain, 2 at once | speculative, 2 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            for r in chats:
                lines.append(
                    f"| {r['model']} | {at(r, 'chat', 'decode')} | {at(r, 'chat draft', 'decode')} "
                    f"| {at(r, 'chat draft', 'per_step')} | {at(r, 'chat 2', 'decode')} "
                    f"| {at(r, 'chat draft 2', 'decode')} |"
                )
        return "\n".join(lines)

    @staticmethod
    def failures(row: dict) -> dict[str, str]:
        """A model's runs that failed and their errors, agreement's of each build and way among
        them."""
        failed = {k: v["error"] for k, v in row.items() if isinstance(v, dict) and "error" in v}
        for build, result in row.get("agreement", {}).items():
            if "error" in result:
                failed[f"agreement, {build}"] = result["error"]
            failed |= {f"agreement, {build}, {name}": v["error"] for name, v in result.items()
                       if isinstance(v, dict) and "error" in v}  # fmt: skip
        return failed

    def _errors(self) -> list[str]:
        return [f"- {r['model']}, {k}: `{e}`" for r in self.rows
                for k, e in self.failures(r).items()]  # fmt: skip


def _ggufs(paths: list[Path]) -> list[Path]:
    # the files given, and those in the directories given; split files by their first part
    found = []
    for path in paths:
        files = sorted(path.glob("*.gguf")) if path.is_dir() else [path]
        found += [f for f in files if not re.search(r"-0000[2-9]-of-|mmproj", f.name)]
    return list(dict.fromkeys(found))


def _is_drafter(path: Path) -> bool:
    # a Gemma 4 assistant, or a model's MTP layer alone
    with contextlib.suppress(Exception):
        return mtp(path) or GGUF.open(path).metadata["general.architecture"] == "gemma4-assistant"
    return False


def _drafter(model: Path, candidates: list[Path]) -> Path | None:
    # the model's own MTP layer, in its file or beside it, or a Gemma 4 assistant for Gemma 4
    if (own := own_drafter(model)) is not None:
        return own
    if GGUF.open(model).metadata["general.architecture"] == "gemma4":
        return next((c for c in candidates if _is_drafter(c) and not mtp(c)), None)
    return None


def _machine(env: dict) -> dict:
    info = {"host": socket.gethostname(), "kernel": platform.release(), "device": env["DEV"]}
    with contextlib.suppress(Exception):
        cpu = Path("/proc/cpuinfo").read_text()
        info["cpu"] = re.search(r"model name\s*:\s*(.*)", cpu).group(1)  # type: ignore[union-attr]
        mem = re.search(r"MemTotal:\s*(\d+)", Path("/proc/meminfo").read_text())
        info["memory"] = f"{int(mem.group(1)) / 2**20:.0f} GiB"  # type: ignore[union-attr]
    probe = "from tinygrad import Device; d = Device[Device.DEFAULT]; print(getattr(d, 'arch', ''))"
    done = _run([sys.executable, "-c", probe], env, 300)
    info["arch"] = done["out"].strip().splitlines()[-1] if done["ok"] and done["out"] else "?"
    commit = _run(["git", "-C", str(ROOT), "describe", "--always", "--dirty"], env, 30)
    info["leatengine"] = commit["out"].strip()
    return info


def _tests(env: dict, *selection: str) -> dict:
    # the GPU tests, or those -k selects of them
    start = time.perf_counter()
    # one process on a real GPU, which the workers would share; many on the emulated one
    workers = ["-n", "auto"] if env["DEV"].startswith("MOCK") else []
    args = [sys.executable, "-m", "pytest", "-m", "gpu", "-q", *workers, "-p", "no:cacheprovider",
            *selection, str(ROOT / "tests")]  # fmt: skip
    done = _run(args, env, 7200, cwd=ROOT)
    lines = done["out"].strip().splitlines()
    summary = next((line for line in reversed(lines) if " in " in line), "no summary")
    failures = [line.removeprefix("FAILED ") for line in lines if line.startswith("FAILED ")]
    seconds = time.perf_counter() - start
    return {"ok": done["ok"], "summary": summary.strip("= "), "failures": failures,
            "seconds": seconds}  # fmt: skip


def _leatengine(env: dict, *args: object) -> dict:
    done = _run(
        [sys.executable, "-m", "leatengine", *map(str, args), "--json"], env, 3600, cwd=ROOT
    )
    if not done["ok"]:
        return {"error": _last_line(done)}
    return json.loads(done["out"].strip().splitlines()[-1])


def _llama_bench(bin_dir: Path, model: Path, reps: str) -> dict:
    args = [str(bin_dir / "llama-bench"), "-m", str(model), "-p", "512", "-n", "128", "-r", reps,
            "-ngl", "99", "-fa", "1", "-o", "json"]  # fmt: skip
    done = _run(args, os.environ.copy(), 3600)
    if not done["ok"]:
        return {"error": _last_line(done)}
    rows = json.loads(done["out"])
    speeds = {("decode" if r["n_gen"] else "prefill"): r["avg_ts"] for r in rows}
    return speeds | {"build": rows[0].get("build_commit"), "backend": rows[0].get("backends")}


def _name(bin_dir: Path) -> str:
    # a build's name: its directory's, build-vulkan's vulkan
    build = bin_dir.parent if bin_dir.name == "bin" else bin_dir
    return build.name.removeprefix("build-") or str(bin_dir)


def _best(builds: dict[str, dict]) -> dict:
    """Of each measure, the best of the builds that gave it, and which it was, of several; each
    build's results whole under "builds"; of one build, its results as they are."""
    if len(builds) == 1:
        return next(iter(builds.values()))
    best: dict = {"builds": builds}
    for field in ("prefill", "decode"):
        given = {name: r[field] for name, r in builds.items() if field in r}
        if given:
            by = max(given, key=lambda name: given[name])
            best |= {field: given[by], f"{field} by": by}
    if "prefill" not in best and "decode" not in best:  # every build failed
        best["error"] = "; ".join(f"{name}: {r['error']}" for name, r in builds.items())
    return best


def _agreement(env: dict, model: Path, builds: list[Path], wikitext: Path, out: Path) -> dict:
    """Of each build of llama.cpp, its perplexity of wikitext's first 8 chunks of 512 tokens and
    its commit, and LeatEngine's scores of its logits of them, each way of AGREEMENT's."""
    out.mkdir(parents=True, exist_ok=True)
    base, results = out / f".{model.stem}.kld", {}
    for bin_dir in builds:
        result = _llama_perplexity(bin_dir, model, wikitext, base)
        if "error" not in result:
            for name, flags in AGREEMENT.items():
                result[name] = _leatengine(env, "perplexity", model, "--kl-base", base, *flags)
        base.unlink(missing_ok=True)  # hundreds of MB
        results[_name(bin_dir)] = result
    return results


def _llama_perplexity(bin_dir: Path, model: Path, text: Path, base: Path) -> dict:
    # llama.cpp's perplexity of the text's first 8 chunks of 512, its logits saved to base
    args = [str(bin_dir / "llama-perplexity"), "-m", str(model), "-f", str(text), "-c", "512",
            "--chunks", "8", "-ngl", "99", "-fa", "on",
            "--kl-divergence-base", str(base)]  # fmt: skip
    done = _run(args, os.environ.copy(), 3600)
    found = re.search(r"Final estimate: PPL = ([\d.]+)", done["out"] + done["err"])
    if not done["ok"] or found is None:
        return {"error": _last_line(done)}
    version = _run([args[0], "--version"], os.environ.copy(), 60)
    commit = re.search(r"commit (\w+)", version["out"] + version["err"])
    return {"perplexity": float(found.group(1)), "commit": commit.group(1) if commit else "?"}


def _llama_batched(bin_dir: Path, model: Path) -> dict:
    # llama-batched-bench's speed of 4 sequences generating 128 tokens each after 1-token prompts
    args = [str(bin_dir / "llama-batched-bench"), "-m", str(model), "-c", "2048", "-ngl", "99",
            "-fa", "on", "-npp", "1", "-ntg", "128", "-npl", "4"]  # fmt: skip
    done = _run(args, os.environ.copy(), 3600)
    rows = [line.split("|") for line in done["out"].splitlines() if re.match(r"\|\s+1 \|", line)]
    if not done["ok"] or not rows:
        return {"error": _last_line(done)}
    return {"decode": float(rows[0][8])}  # the column S_TG t/s


def _server(env: dict, model: Path, out: Path) -> dict:
    # serve on a free port, wait until the model is loaded, then request a streamed chat reply
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    start = time.perf_counter()
    args = [sys.executable, "-m", "leatengine", "serve", str(model), "--port", str(port)]
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / ".server.log", "w")  # noqa: SIM115
    server = subprocess.Popen(args, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    url, result = f"http://127.0.0.1:{port}", {"ok": False}
    try:
        while time.perf_counter() - start < 1800 and server.poll() is None:
            with contextlib.suppress(OSError, ValueError):
                listed = json.loads(urllib.request.urlopen(url + "/v1/models", timeout=5).read())
                if any(m.get("status") == "loaded" for m in listed["data"]):
                    break
            time.sleep(2)
        result["ready after"] = f"{time.perf_counter() - start:.0f} s"
        body = {"messages": [{"role": "user", "content": SERVER_PROMPT}], "max_tokens": 200,
                "stream": True, "chat_template_kwargs": {"enable_thinking": False}}  # fmt: skip
        request = urllib.request.Request(
            url + "/v1/chat/completions", json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )  # fmt: skip
        sent, first, text = time.perf_counter(), None, []
        with urllib.request.urlopen(request, timeout=600) as response:
            for line in response:
                if not line.startswith(b"data: {"):
                    continue
                for choice in json.loads(line[6:]).get("choices", []):
                    if piece := choice.get("delta", {}).get("content"):
                        first = first or time.perf_counter()
                        text.append(piece)
        if first is not None:
            result["first token"] = f"{1000 * (first - sent):.0f} ms"
        result |= {"ok": bool("".join(text).strip()), "reply": "".join(text)}
    except Exception as e:  # noqa: BLE001
        result["error"] = repr(e)
    finally:
        server.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            server.wait(30)
        log.close()
    return result


def _run(args: list[str], env: dict, timeout: int, cwd: Path | None = None) -> dict:
    try:
        done = subprocess.run(args, env=env, cwd=cwd, capture_output=True, text=True,
                              timeout=timeout)  # fmt: skip
    except subprocess.TimeoutExpired:
        return {"ok": False, "out": "", "err": f"timed out after {timeout} s"}
    except OSError as e:
        return {"ok": False, "out": "", "err": str(e)}
    return {"ok": done.returncode == 0, "out": done.stdout, "err": done.stderr}


def _last_line(done: dict) -> str:
    lines = (done["err"] or done["out"]).strip().splitlines()
    return lines[-1][:300] if lines else "failed"


if __name__ == "__main__":
    main()
