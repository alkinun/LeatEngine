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
`797924c3715f72f907b0a02e14aa327c3b626efc`; changing it is a deliberate compatibility update.
Retain the lockfile for reproducible development installs.

tinygrad keeps compiled kernels in `~/.cache/tinygrad/cache.db`, and LeatEngine keeps the schedules
and lowered programs there too (`leatengine.kernels.lowering`), so that a model's later starts skip
scheduling and lowering. `CACHELEVEL=0` turns all three off, as for timing a first start. The tests'
pytest-xdist workers share the database, so that each kernel is compiled once among them.

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

The hero models, the large models LeatEngine aims to run fastest on Strix Halo, have tests of their
layers at full size: `tests/heroes.py` lists, from each GGUF's header, the shapes and types of a
layer or two of gpt-oss 120B, of Qwen3 235B A22B at three low-bit quantizations, and of Qwen3.6 35B
A3B, every expert included, and writes them with random weights. The kernels prefill and decode a
prompt through each, against the reference ops on the CPU. They take up to 1.7 GB of disk each, and
minutes, the more on the emulated GPU, so they run only when selected; where `/tmp` is in memory,
give `--basetemp` a directory on disk.

```bash
DEV=NV uv run pytest -m hero --basetemp ~/.cache/leatengine-heroes
```

## Emulated AMD tests

Install clang with its AMDGPU target, which compiles the kernels as on a real GPU. Obtain
tinygrad's source at the same pinned commit: the emulator lives in its tests and is not included
in the installed package.

```bash
git clone https://github.com/tinygrad/tinygrad.git /tmp/leatengine-tinygrad
git -C /tmp/leatengine-tinygrad checkout 797924c3715f72f907b0a02e14aa327c3b626efc
PYTHONPATH=/tmp/leatengine-tinygrad DEV=MOCK+AMD uv run pytest -m gpu -n auto \
  --cov --cov-report= --durations=20
```

The tests turn off unaligned memory access, which the emulator lacks, and have the emulated GPU say
what Strix Halo's does: gfx1151, of 40 CUs, by which matmuls split their work. The emulator runs
RDNA 3.5's code as RDNA 3's, and is otherwise a gfx1100's. This checks kernel behavior in
emulation; physical Strix Halo validation is separate. The emulator skips decode attention over
heads wider than 512 dimensions, which it computes wrongly; the [Strix Halo guide](strix-halo.md)
covers that check.

## Continuous integration

- Ruff and mypy.
- CPU tests on Python 3.12 and 3.14.
- AMD kernel tests on tinygrad's emulator, as Strix Halo's GPU.
- Tokenizer comparisons against llama.cpp at commit `889edf43ddae0cfe9a4564a882764dc879759870`.
- Combined CPU/emulator coverage of at least 80%.
- Wheel and source-distribution installation checks.
- The release tarball, built twice for identical bytes, checked unpacked, and kept as the run's
  artifact.

CPU and emulated GPU jobs rebalance queued tests between workers with `--dist worksteal`, so
long model cases do not leave the remaining tests queued behind one busy worker. The emulated GPU
tests run in three jobs: the attention kernels', the other kernels', and the tiny models through
them; all contribute to the combined coverage gate. Each test job starts from the tinygrad cache
of its last run, kept by `uv.lock`, clang and the runner's CPU, so that a run compiles, lowers and
schedules only what changed.

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
tinygrad Git source, which PyPI does not accept as a dependency; the release tarball below needs
neither.

## Release tarball

```bash
uv run python scripts/build_release.py
```

This builds `dist/leatengine-0.0.1-linux-x86_64.tar.gz`, about 37 MiB, and its SHA-256 beside it.
The tarball holds python-build-standalone's CPython 3.14 in `python/`, with LeatEngine and its
locked dependencies installed into it, and `bin/leatengine`, which runs that Python from wherever
the tarball is unpacked, ignoring `PYTHONPATH` and the like. `BUILD-INFO.json` records the
commit, the CPython build, and the tinygrad commit. Running it needs no network, uv, or system
Python:

```bash
tar -xzf leatengine-0.0.1-linux-x86_64.tar.gz
leatengine-0.0.1-linux-x86_64/bin/leatengine serve /path/to/model.gguf
```

The host provides glibc 2.28 or newer and what [the README](../README.md#install) lists for the
GPU: clang with its AMDGPU target is not bundled. Tk, IDLE, pip, and Python's headers and shared
library are left out. Bytecode is compiled at build time and checked against its source's hash,
so a read-only unpack starts without writing to it.

A commit builds the same bytes. The CPython archive is pinned by SHA-256, dependencies come from
`uv.lock`, tinygrad is built afresh rather than taken from uv's cache, build backends resolve as
of the commit's time, and the archive has fixed file times, owners, and order. The bundled Python
compresses it, so the building machine's zlib does not matter. `--reproducible` builds a second
time in another directory and requires identical bytes. To bump CPython, change `PYTHON` and
`PYTHON_SHA256` in the script, taking the hash from the python-build-standalone release's
`SHA256SUMS`.

After building, the script unpacks the tarball into a temporary directory whose path has a space and
checks it: the launcher, directly and through a symlink; `leatengine doctor`'s report; the packaging
check's isolation and streamed completion from generated CPU weights; and `bin/leatengine serve` as
a supervisor such as Lemonade would start it, waiting on `/health`, streaming a completion, and
stopping it with SIGINT. Where unprivileged user namespaces are allowed, the last two run in a
network namespace with only loopback; elsewhere the script says so and checks with the network.
`--no-check` skips the checks.

## Provenance

The README records the source this repository was extracted from. Keep its MIT notice, and tell
inherited measurements apart from results measured on a LeatEngine commit.
