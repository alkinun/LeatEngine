# Development

The runtime is the `leatengine` package, its GPU kernels in `leatengine.kernels`. Model and
weight handling, inference, chat formatting, serving, and benchmarking are separate modules.
Keep application features in clients that communicate over HTTP.

## Environment and checks

```bash
uv sync --locked --python 3.14
uv run ruff check
uv run ruff format --check
uv run mypy
DEV=CPU uv run pytest -n auto --cov --cov-report= --durations=20
uv run python scripts/check_package.py
```

Use Python 3.14 to match the primary CI environment; the CPU suite also runs on 3.12.

The runtime requires tinygrad, Jinja2, and Pillow. Development dependencies provide independent
GGUF/numerical references, HTTP-client tests, and quality checks. tinygrad is pinned to commit
`9d7054488aee0e455e226514ba4b8b2f32329c29`; changing it is a deliberate compatibility update.
Retain the lockfile for reproducible development installs.

tinygrad keeps compiled kernels in `~/.cache/tinygrad/cache.db`, and LeatEngine keeps the lowered
programs there too (`leatengine.kernels.lowering`), so that a model's later starts skip lowering.
`CACHELEVEL=0` turns both off, as for timing a first start; the tests give each pytest-xdist
worker a database of its own.

The default tests use generated tiny GGUFs and CPU execution. GPU and real-model tests are
opt-in. `LEATENGINE_KERNELS=ref` disables custom kernels for numerical comparisons.

```bash
DEV=NV uv run pytest -m gpu
DEV=AMD uv run pytest -m gpu
DEV=NV LEATENGINE_MODEL=/models/model.gguf \
  LLAMA_CPP=/path/to/llama.cpp/build/bin WIKITEXT=/path/to/wiki.test.raw uv run pytest
```

Use one test process on a real GPU so models do not compete for its memory. The real-model
tests compare tokenization and logits with the llama.cpp build in `LLAMA_CPP`, which must support
the model's architecture (Muse Glimmer needs llama.cpp `1b70350` or newer), and check tool calls,
which need a tool-capable model. A module-scoped server keeps one copy of the model loaded while
other tests load another: with a model over half the GPU's memory, run the test files one at a
time. The logits comparison skips architectures without an agreement limit in
`tests/test_model.py`, whose comments record the measurements behind each limit.

## Emulated AMD tests

Install clang with its AMDGPU target, which compiles the kernels as on a real GPU. Obtain
tinygrad's source at the same pinned commit: the emulator lives in its tests and is not included
in the installed package.

```bash
git clone https://github.com/tinygrad/tinygrad.git /tmp/leatengine-tinygrad
git -C /tmp/leatengine-tinygrad checkout 9d7054488aee0e455e226514ba4b8b2f32329c29
PYTHONPATH=/tmp/leatengine-tinygrad DEV=MOCK+AMD uv run pytest -m gpu -n auto \
  --cov --cov-report= --durations=20
```

The tests turn off unaligned memory access, which the emulator lacks, and isolate compilation
caches per worker. This checks RDNA 3 kernel behavior in emulation; physical Strix Halo
validation is separate. The emulator skips decode attention over heads wider than 512
dimensions, which it computes wrongly; the [Strix Halo guide](strix-halo.md) covers that check.

## Continuous integration

- Ruff and mypy.
- CPU tests on Python 3.12 and 3.14.
- AMD kernel tests on tinygrad's emulator.
- Tokenizer comparisons against llama.cpp at commit `889edf43ddae0cfe9a4564a882764dc879759870`.
- Combined CPU/emulator coverage of at least 80%.
- Wheel and source-distribution installation checks.

CPU and emulated GPU jobs rebalance queued tests between workers with `--dist worksteal`, so
long model cases do not leave the remaining tests queued behind one busy worker. AMD primitive
kernels and the remaining GPU integration tests run in separate CI jobs; both contribute to the
combined coverage gate.

The packaging check builds both artifacts and installs each into a fresh environment. It runs
outside the source checkout, checks both CLI entry points and package boundaries, and serves
a streamed completion from generated CPU weights. No original Leat checkout or development
dependencies are available to the installed runtime.

## Building and installing

```bash
uv build
uv venv /tmp/leatengine-runtime
uv pip install --python /tmp/leatengine-runtime/bin/python dist/leatengine-0.0.1-py3-none-any.whl
cd /tmp
/tmp/leatengine-runtime/bin/python -m leatengine --help
```

The wheel contains only the runtime package and distribution metadata. The source distribution
also includes tests, scripts, and documentation. Installation needs access to the pinned
tinygrad Git source; PyPI publication and offline bundles are separate work.

## Provenance

The README records the source this repository was extracted from. Keep its MIT notice, and tell
inherited measurements apart from results measured on a LeatEngine commit.
