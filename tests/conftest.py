import gc
import os
from collections.abc import Callable
from pathlib import Path

import pytest

# tinygrad would pick a GPU when it finds one: the default run stays on the CPU on any machine.
# It reads DEV when imported, so this comes first.
os.environ.setdefault("DEV", "CPU")
GPU_BACKENDS = {"NV", "CUDA", "AMD", "MOCK+AMD"}
# What the emulated GPU's KFD properties say, as Strix Halo's would: gfx1151, of 40 CUs. The CUs
# set how matmuls split their work; the arrays, 2 of 10 CUs to each of 2 shader engines, keep the
# counts whole. The emulator is a gfx1100's otherwise, its registers and IP versions too.
STRIX_HALO = {"gfx_target_version": 110501, "simd_count": 80, "array_count": 4,
              "cu_per_simd_array": 10}  # fmt: skip
if os.environ["DEV"].split(":")[0] == "MOCK+AMD":
    # tinygrad's emulated RDNA 3 GPU, from the tests of tinygrad's source tree at the pinned commit,
    # which PYTHONPATH must hold. It does not allow unaligned access, as GPUs do: clang would merge
    # the halfword loads of blocks only halfword aligned into wide ones. Compiles run in this
    # process, which has the flag: tinygrad's compile workers make their renderers anew.
    os.environ.setdefault("PARALLEL", "0")
    from leatengine.kernels import clang

    clang.FLAGS = ("-Xclang", "-target-feature", "-Xclang", "-unaligned-access-mode")
    # The emulator runs RDNA 3.5's code as RDNA 3's, which is the same for what clang emits from
    # the kernels. tinygrad makes its devices' files as it is imported, so they change here, before
    # the device opens and reads them.
    from test.mockgpu import mockgpu

    for file in (f for driver in mockgpu.drivers for f in driver.tracked_files):
        if file.path.startswith("/sys/devices/virtual/kfd/") and file.path.endswith("/properties"):
            lines = file.fdcls.keywords["text"].splitlines()
            said = (line.split(" ", 1) for line in lines)
            file.fdcls.keywords["text"] = "\n".join(
                f"{key} {STRIX_HALO.get(key, value)}" for key, value in said
            )


@pytest.fixture(autouse=True)
def _collect(request):
    # a real model fills most of a GPU: free one test's before the next test loads another
    yield
    if request.node.get_closest_marker("model"):
        gc.collect()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # GPU and real-model tests are opt-in so the default run stays hermetic and runs on any CPU.
    # Large models' layers take gigabytes and minutes each: only when -m selects them.
    has_gpu = os.environ.get("DEV", "").split(":")[0] in GPU_BACKENDS
    has_model = bool(os.environ.get("LEATENGINE_MODEL"))
    large = "large" in config.option.markexpr
    for item in items:
        if "gpu" in item.keywords and not has_gpu:
            item.add_marker(pytest.mark.skip(reason="needs DEV=NV, CUDA, AMD or MOCK+AMD"))
        if "model" in item.keywords and not has_model:
            item.add_marker(pytest.mark.skip(reason="needs LEATENGINE_MODEL=path.gguf"))
        if "large" in item.keywords and not large:
            item.add_marker(pytest.mark.skip(reason="takes gigabytes: select it with -m large"))


@pytest.fixture
def reference_ops(monkeypatch):
    # the plain tinygrad ops alone, for exact comparisons with the f64 references: the kernels
    # quantize activations to int8, and on the GPU the logits of up to 8 positions take them
    monkeypatch.setenv("LEATENGINE_KERNELS", "ref")


@pytest.fixture(scope="session")
def tiny(tmp_path_factory) -> Callable[[str], tuple[Path, dict]]:
    # tiny(arch): a random GGUF of an architecture and its weights decoded independently by
    # gguf-py, written once per session
    from tests.helpers import write_tiny_model

    made: dict[str, tuple[Path, dict]] = {}

    def model(arch: str) -> tuple[Path, dict]:
        if arch not in made:
            path = tmp_path_factory.mktemp(arch) / "tiny.gguf"
            made[arch] = path, write_tiny_model(path, arch)
        return made[arch]

    return model


@pytest.fixture(scope="session")
def tiny_model(tiny) -> tuple[Path, dict]:
    return tiny("llama")


@pytest.fixture(scope="session")
def tiny_assistant(tmp_path_factory) -> tuple[Path, dict]:
    # a random Gemma 4 assistant, which drafts for tiny("gemma4"), and its weights
    from tests.helpers import write_tiny_assistant

    path = tmp_path_factory.mktemp("assistant") / "assistant.gguf"
    return path, write_tiny_assistant(path)


@pytest.fixture(scope="session")
def model_path() -> Path:
    return Path(os.environ["LEATENGINE_MODEL"]).expanduser()


@pytest.fixture(scope="session")
def llama_cpp() -> Path:
    # llama.cpp's build/bin directory, used as the reference implementation
    if not (path := os.environ.get("LLAMA_CPP")):
        pytest.skip("needs LLAMA_CPP=path/to/llama.cpp/build/bin")
    return Path(path).expanduser()


@pytest.fixture(scope="session")
def wikitext() -> Path:
    # wikitext-2's wiki.test.raw, the usual text for perplexity
    if not (path := os.environ.get("WIKITEXT")):
        pytest.skip("needs WIKITEXT=path/to/wiki.test.raw")
    return Path(path).expanduser()
