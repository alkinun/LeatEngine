import gc
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from tinygrad import Device, UOp

from leatengine.gguf import GGUF
from leatengine.model import Config, Transformer
from tests.helpers import V
from tests.heroes import HEROES, logits, write_hero

# a GPU's, and of gigabytes each: run only when asked for, with -m hero
pytestmark = [pytest.mark.gpu, pytest.mark.hero]
CONTEXT, DECODED = 256, 8


@pytest.fixture(autouse=True)
def _free():
    # a hero's buffers, which tinygrad would keep for reuse, freed before the next one's
    yield
    gc.collect()
    Device[Device.DEFAULT].allocator.free_cache()


@pytest.mark.parametrize("name", list(HEROES))
def test_hero_layers_match_reference(tmp_path, name):
    # the kernels on a hero model's layers, prefilling a prompt past its sliding window, if any,
    # then decoding its last DECODED tokens one at a time, against the reference ops on the CPU,
    # as test_model's tiny models against the f64 references: within the kernels' int8
    # activations' rounding. The reference runs in a process of its own, whose memory goes when
    # it ends.
    path, reference = tmp_path / f"{name}.gguf", tmp_path / "reference.npy"
    write_hero(path, HEROES[name])
    f = GGUF.open(path)
    config = Config.from_gguf(f.metadata)
    prompt = np.random.default_rng(3).integers(0, V, max(config.windows) + 24).tolist()
    args = [sys.executable, "-m", "tests.heroes", path, reference, str(CONTEXT), *map(str, prompt)]
    root = Path(__file__).parents[1]
    subprocess.run(args, cwd=root, env=os.environ | {"DEV": "CPU"}, check=True)
    expected = np.load(reference)
    model = Transformer(config, f.load(), CONTEXT)
    path.unlink()  # loaded: pytest keeps its last runs' files, as in a /tmp of memory
    split = len(prompt) - DECODED
    pos = UOp.variable("start_pos", 0, CONTEXT - 1)
    got = [logits(model, prompt[:split])]
    got += [logits(model, [t], pos.bind(split + i)) for i, t in enumerate(prompt[split:])]
    np.testing.assert_allclose(np.concatenate(got), expected, atol=0.05 * np.abs(expected).max())
