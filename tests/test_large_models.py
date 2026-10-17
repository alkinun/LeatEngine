import gc
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from tinygrad import Device, UOp

from leatengine.gguf import GGUF
from leatengine.model import Config
from tests.helpers import V
from tests.large_models import LARGE_MODELS, load, run, write_layers

# a GPU's, and of gigabytes each: run only when asked for, with -m large
pytestmark = [pytest.mark.gpu, pytest.mark.large]
CONTEXT, DECODED = 256, 8


@pytest.fixture(autouse=True)
def _free():
    # a model's buffers, which tinygrad would keep for reuse, freed before the next one's
    yield
    gc.collect()
    Device[Device.DEFAULT].allocator.free_cache()


@pytest.mark.parametrize("name", list(LARGE_MODELS))
def test_large_layers_match_reference(tmp_path, name):
    # the kernels on a large model's layers, prefilling a prompt past its sliding window, if any,
    # then decoding its last DECODED tokens one at a time, against the reference ops on the CPU,
    # as test_model's tiny models against the f64 references: within the kernels' int8
    # activations' rounding; and its MTP layer's, if any, on the model's hidden states, as it
    # takes in a prompt and then drafts. The reference runs in a process of its own, whose memory
    # goes when it ends.
    path, reference = tmp_path / f"{name}.gguf", tmp_path / "reference.npz"
    write_layers(path, LARGE_MODELS[name])
    config = Config.from_gguf(GGUF.open(path).metadata)
    prompt = np.random.default_rng(3).integers(0, V, max(config.windows) + 24).tolist()
    args = [sys.executable, "-m", "tests.large_models", path, reference, str(CONTEXT)]
    args += map(str, prompt)
    root = Path(__file__).parents[1]
    subprocess.run(args, cwd=root, env=os.environ | {"DEV": "CPU"}, check=True)
    expected = np.load(reference)
    model, mtp = load(path, CONTEXT)
    path.unlink()  # loaded: pytest keeps its last runs' files, as in a /tmp of memory
    split = len(prompt) - DECODED
    pos = UOp.variable("start_pos", 0, CONTEXT - 1)
    got, states, drafted = run(model, mtp, prompt[:split], np.zeros(config.dim, np.float32))
    parts, layer = [got], [drafted]
    for i, t in enumerate(prompt[split:]):
        got, states, drafted = run(model, mtp, [t], states[-1], pos.bind(split + i))
        parts.append(got)
        layer.append(drafted)
    want = expected["logits"]
    np.testing.assert_allclose(np.concatenate(parts), want, atol=0.05 * np.abs(want).max())
    if mtp is not None:
        want = expected["mtp"]
        np.testing.assert_allclose(np.concatenate(layer), want, atol=0.05 * np.abs(want).max())
    assert (mtp is None) == (expected["mtp"].size == 0)
