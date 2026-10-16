"""Measures LeatEngine's agreement with llama.cpp on the same files, validate_hardware.py's section
of it alone: of each model and build of llama.cpp, both engines' perplexity of wikitext-2's first
8 chunks of 512 tokens, and LeatEngine's KL divergence from the build's next-token distributions
and how often both pick the same token, as prompts, one token at a time as generation runs, and
with a q8_0 KV cache. Writes a Markdown report to reports/, or --out, and prints it.

    uv run python scripts/agreement.py MODEL.gguf|DIR ... --llama-cpp BIN [BIN ...] \\
        --wikitext wiki.test.raw

wiki.test.raw is wikitext-2's test set, as llama.cpp's scripts/get-wikitext-2.sh fetches it.
Every run is a process of its own. DEV picks the device, AMD by default for Strix Halo.
"""

import argparse
import os
import sys
from pathlib import Path

from validate_hardware import ROOT, Report, _agreement, _ggufs, _is_drafter, _machine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("models", nargs="+", type=Path, help="GGUF files, or directories of them")
    parser.add_argument(
        "--llama-cpp", type=Path, nargs="+", required=True,
        help="llama.cpp's build/bin, or several builds', to compare against each",
    )  # fmt: skip
    parser.add_argument("--wikitext", type=Path, required=True, help="wikitext-2's wiki.test.raw")
    parser.add_argument("--out", type=Path, default=ROOT / "reports", help="report directory")
    args = parser.parse_args()
    env = os.environ | {"DEV": os.environ.get("DEV", "AMD")}
    report = Report(env["DEV"], "agreement")
    report.machine(_machine(env))
    for model in (m for m in _ggufs(args.models) if not _is_drafter(m)):
        print(f"== {model.name}", file=sys.stderr, flush=True)
        agreement = _agreement(env, model, args.llama_cpp, args.wikitext, args.out)
        report.model({"model": model.stem, "agreement": agreement})
    path = report.write(args.out)
    print(report.text())
    print(f"\nwritten to {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
