# Development

The runtime is one flat `leatengine` package. Model/weight handling, inference, kernels,
chat formatting, serving, and benchmarking remain separate modules. Keep application
features in clients that communicate over HTTP.

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
Local stress tests with uv-managed CPython 3.13.14 encountered intermittent native worker
crashes during CPU and emulated AMD execution, with and without coverage. Their cause is
not yet isolated; passing individual reruns do not establish stability. Use Python 3.14
for development and installation while this is investigated.

The runtime requires tinygrad, Jinja2, and Pillow. Development dependencies provide independent
GGUF/numerical references, HTTP-client tests, and quality checks. tinygrad is pinned to commit
`9d7054488aee0e455e226514ba4b8b2f32329c29`; changing it is a deliberate compatibility update.
Retain the lockfile for reproducible development installs.

The default tests use generated tiny GGUFs and CPU execution. GPU and real-model tests are
opt-in. `LEATENGINE_KERNELS=ref` disables custom kernels for numerical comparisons.

```bash
DEV=NV uv run pytest -m gpu
DEV=AMD uv run pytest -m gpu
DEV=NV LEATENGINE_MODEL=/models/model.gguf \
  LLAMA_CPP=/path/to/llama.cpp/build/bin WIKITEXT=/path/to/wiki.test.raw uv run pytest
```

Use one test process on a real GPU so models do not compete for its memory. Model-specific
tests may require suitable tool-capable weights and a llama.cpp reference build.

## Emulated AMD tests

Install clang with its AMDGPU target and lld. Obtain tinygrad's source at the same pinned
commit: the emulator lives in its tests and is not included in the installed package.

```bash
git clone https://github.com/tinygrad/tinygrad.git /tmp/leatengine-tinygrad
git -C /tmp/leatengine-tinygrad checkout 9d7054488aee0e455e226514ba4b8b2f32329c29
PYTHONPATH=/tmp/leatengine-tinygrad DEV=MOCK+AMD uv run pytest -m gpu -n auto \
  --cov --cov-report= --durations=20
```

The test harness installs a clang compiler adapter and isolates compilation caches per worker.
This checks RDNA 3 kernel behavior in emulation; physical Strix Halo validation is separate.

## Continuous integration

- Ruff and mypy.
- CPU tests on Python 3.12 and 3.14.
- AMD kernel tests on tinygrad's emulator.
- Tokenizer comparisons against llama.cpp at commit `889edf43ddae0cfe9a4564a882764dc879759870`.
- Combined CPU/emulator coverage of at least 80%.
- Wheel and source-distribution installation checks.

CPU and emulated GPU jobs rebalance queued tests between workers with `--dist worksteal`,
so long model cases do not leave the remaining tests queued behind one busy worker.
AMD primitive kernels and the remaining GPU integration tests run in separate CI jobs;
both contribute to the combined coverage gate.

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

## Extraction baseline

The original 14 selected test modules collected 1,013 cases, including one application-only
CLI case removed here. The remaining 1,012 cases cover the inherited engine. Naming and packaging
checks may add cases; GPU/model prerequisites still determine which execute in each environment.

The source provenance is recorded in the README. Maintain the MIT notice and distinguish
inherited measurements from results obtained on a specific LeatEngine commit.

The isolated installation check exposed two inherited uses of NumPy for reading generated
token IDs. They now use tinygrad's list conversion, so decoding and speculation work with
the three declared runtime dependencies. Model computations and the tinygrad pin are unchanged.

The source snapshot's Python 3.12 CI also exposed a request-queue comparison bug: removing a
queued completion could read its not-yet-populated finish state. Completion requests now compare
by identity; the existing background-scheduling test verifies admission order on both CI versions.
