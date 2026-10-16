import json
import math
import subprocess
from dataclasses import replace

import numpy as np
import pytest
from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Context, diskcache_get, diskcache_put

from leatengine import bench, machine, ops
from leatengine.engine import FEW_TOKENS, KEEP_BACK, RESERVE, Engine, _fitted, graph
from leatengine.gguf import GGUF
from leatengine.grammar import Grammar
from leatengine.kernels import VARIABLES
from leatengine.machine import Memory
from leatengine.model import (
    CACHE_TILE,
    Config,
    Rope,
    Transformer,
    _stack,
    cache_bytes,
    rope_table,
    zeros,
)
from leatengine.ops import Span
from leatengine.quant import GGMLType, QTensor
from leatengine.sampler import GREEDY, Sampling
from tests.helpers import (
    CONTEXT,
    P3_MSCALE,
    P3_ORIGINAL,
    P3_ROTATED,
    q8_0,
    random_blocks,
    reference_embedding,
    reference_logits,
    write_tiny_model,
)

PROMPT = [5, 77, 120, 3, 299, 42, 8, 150, 61, 200, 9, 33]
ARCHS = [
    "llama",
    "qwen2",
    "qwen3",
    "qwen3moe",
    "gemma4",
    "gemma4-dense",
    "gemma3",
    "gpt-oss",
    "phi3",
    "qwen35",
    "qwen35moe",
    "mellum",
    "muse-glimmer",
    "lfm2moe",
    "deepseek2",
]
# of plain layers, of rings with sinks, of keys that are values, beside recurrent layers, and of
# MLA's one head: the ways a q8_0 cache holds keys and values
Q8_0_ARCHS = ["llama", "gpt-oss", "gemma4", "qwen35moe", "deepseek2"]


def transformer(
    path, max_context: int = CONTEXT, slots: int = 1, cache_type: str = "f16"
) -> Transformer:
    f = GGUF.open(path)
    return Transformer(
        Config.from_gguf(f.metadata), f.load(), max_context, slots, cache_type=cache_type
    )


@pytest.mark.usefixtures("reference_ops")
@pytest.mark.parametrize("arch", ARCHS)
def test_forward_matches_reference(tiny, arch):
    path, weights = tiny(arch)
    model, expected = transformer(path), reference_logits(weights, PROMPT, arch)
    logits = model.logits(model(Tensor([PROMPT], dtype=dtypes.int32), 0)).numpy()[0]
    np.testing.assert_allclose(logits, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.usefixtures("reference_ops")
@pytest.mark.parametrize("arch", Q8_0_ARCHS)
def test_q8_0_cache_matches_reference(tiny, monkeypatch, arch):
    # keys and values held as q8_0, as the f64 reference rounds them
    monkeypatch.setattr("tests.helpers.cached", q8_0)
    path, weights = tiny(arch)
    model, expected = transformer(path, cache_type="q8_0"), reference_logits(weights, PROMPT, arch)
    logits = model.logits(model(Tensor([PROMPT], dtype=dtypes.int32), 0)).numpy()[0]
    np.testing.assert_allclose(logits, expected, rtol=2e-3, atol=2e-3)


def test_q8_0_cache_needs_whole_blocks(tiny_model):
    path, _ = tiny_model
    f = GGUF.open(path)
    config = replace(Config.from_gguf(f.metadata), head_dims=(48, 48))
    with pytest.raises(ValueError, match="heads of a multiple of 32 dimensions.*of 48"):
        Transformer(config, f.load(), CONTEXT, cache_type="q8_0")
    with pytest.raises(ValueError, match="cache_type must be f16 or q8_0"):
        transformer(path, cache_type="q4_0")


@pytest.mark.gpu
@pytest.mark.parametrize(
    "arch, cache_type", [(a, "f16") for a in ARCHS] + [(a, "q8_0") for a in Q8_0_ARCHS]
)
def test_kernels_match_reference(tiny, monkeypatch, arch, cache_type):
    # the kernels, where they take the tiny models' shapes, run every architecture as the f64
    # reference does but for their int8 activations' rounding, some hundredths of the largest
    # logit: a whole prompt, as prefilling, and a token at a time, as decoding; and with a q8_0
    # cache, as the reference rounds keys and values to q8_0
    if cache_type == "q8_0":
        monkeypatch.setattr("tests.helpers.cached", q8_0)
    path, weights = tiny(arch)
    model = transformer(path, cache_type=cache_type)
    expected = reference_logits(weights, PROMPT, arch)
    whole = model.logits(model(Tensor([PROMPT], dtype=dtypes.int32), 0)).numpy()[0]
    pos = UOp.variable("start_pos", 0, CONTEXT - 1)
    each = [model.logits(model(Tensor([[t]], dtype=dtypes.int32), pos.bind(i))).numpy()[0, 0]
            for i, t in enumerate(PROMPT)]  # fmt: skip
    # LFM2's short convolutions multiply the outputs of one int8-activated projection, gate by
    # input by gate, which compounds its rounding: as llama.cpp's MMQ and MMVQ would round it
    tolerance = 0.08 if arch == "lfm2moe" else 0.05
    for got in (whole, np.stack(each)):
        np.testing.assert_allclose(got, expected, atol=tolerance * np.abs(expected).max())


def test_cache_whole_tiles(tiny_model):
    # attention kernels take caches of whole tiles; the model still holds max_context positions
    model = transformer(tiny_model[0], 50)
    assert model.cache[0].shape[3] == CACHE_TILE and model.max_context == 50


@pytest.mark.parametrize("arch", ["llama", "gemma3", "gpt-oss", "qwen35", "deepseek2"])
@pytest.mark.parametrize("cache_type", ["f16", "q8_0"])
def test_cache_bytes(tiny, arch, cache_type):
    # what a Transformer allocates of a context, of rings, recurrent layers and MLA too
    f = GGUF.open(tiny(arch)[0])
    config = Config.from_gguf(f.metadata)
    model = Transformer(config, f.load(), 50, 3, run=8, cache_type=cache_type)
    made = sum(cache.nbytes() for cache in model.cache if cache is not None)
    assert made == cache_bytes(config, 50, 3, 8, cache_type) > 0


@pytest.fixture
def fit(tiny, monkeypatch):
    # the context a tiny Qwen3.5 of 100000 tokens fits in a given memory, without its drafter and
    # with it, as Engine sizes it, of 2 slots, a spare slot beside, and runs of 8
    path = tiny("qwen35")[0]
    config = replace(Config.from_gguf(GGUF.open(path).metadata), context_length=100000)

    def fitted(free: int | None, cache_type="f16", draft=None) -> tuple[int, str]:
        memory = None if free is None else Memory("RAM", 2 * free, free)
        monkeypatch.setattr(machine, "memory", lambda device: memory)
        return _fitted(config, 2, 3, 8, cache_type, draft)

    return config, path, fitted


def test_default_context(fit):
    # the model's own, as far as the cache fits in the memory free beside the weights and
    # RESERVE, in whole tiles; twice as much of q8_0, the drafter's weights and its own cache less
    config, path, fitted = fit
    assert fitted(None) == (4096, "of the model's 100000, as CPU does not say how much memory "
                                  "is free")  # fmt: skip
    context, why = fitted(RESERVE + 1 << 40)
    assert context == 100000 and why.startswith("the model's own: its 2 slots' f16 KV cache takes")
    budget = cache_bytes(config, 10000, 3, 8)  # of 40 whole tiles, 10240 positions
    context, why = fitted(RESERVE + budget)
    assert context == 40 * CACHE_TILE
    assert why.startswith("of the model's 100000: as many as its 2 slots' f16 KV cache fits in")
    assert cache_bytes(config, context, 3, 8) <= budget < cache_bytes(config, context + 256, 3, 8)
    assert fitted(RESERVE + budget, "q8_0")[0] > 1.8 * context
    assert fitted(RESERVE + budget, draft=path)[0] < context
    with pytest.raises(MemoryError, match="hold no 2 slots' f16 KV cache of 256 tokens"):
        fitted(RESERVE + cache_bytes(config, CACHE_TILE, 3, 8) - 1)


def test_engine_default_context(tiny_model, monkeypatch):
    # a model of the context it is given, or of its own, as long as it fits: the tiny one's
    path = tiny_model[0]
    assert Engine(path, max_context=50).fit == "as given"
    monkeypatch.setattr(machine, "memory", lambda device: Memory("RAM", 1 << 40, 1 << 39))
    engine = Engine(path)
    assert engine.max_context == CONTEXT and engine.fit.startswith("the model's own")


def test_bad_models(tiny_model):
    path, _ = tiny_model
    with pytest.raises(ValueError, match=r"max_context must be in \[1, 64\]"):
        transformer(path, CONTEXT + 1)
    f = GGUF.open(path)
    weights = {n: w for n, w in f.load().items() if n != "blk.1.ffn_down.weight"}
    with pytest.raises(ValueError, match="missing 1 tensors, first: blk.1.ffn_down"):
        Transformer(Config.from_gguf(f.metadata), weights, CONTEXT)
    # values, which only Gemma 4's full-attention layers take from their keys
    weights = {n: w for n, w in f.load().items() if n != "blk.0.attn_v.weight"}
    with pytest.raises(ValueError, match="missing 1 tensors, first: blk.0.attn_v"):
        Transformer(Config.from_gguf(f.metadata), weights, CONTEXT)
    with pytest.raises(NotImplementedError, match="architecture 'mamba'"):
        config("mamba")


def config(arch: str, values: dict | None = None, vocab: int = 0) -> Config:
    # the Config of the least metadata an architecture takes, and `values`, without the prefix
    least = {"block_count": 4, "embedding_length": 256, "attention.head_count": 4,
             "context_length": 64, "attention.layer_norm_rms_epsilon": 1e-5}  # fmt: skip
    metadata = {f"{arch}.{k}": v for k, v in (least | (values or {})).items()}
    tokens = {"tokenizer.ggml.tokens": [""] * vocab}
    return Config.from_gguf(metadata | {"general.architecture": arch} | tokens)


def test_config():
    # Gemma 3 27B scales its scores by dim / heads, as llama.cpp, rather than its heads' size
    gemma = {"block_count": 62, "embedding_length": 5376, "attention.head_count": 32,
             "attention.key_length": 128}  # fmt: skip
    assert config("gemma3", gemma).scales == (168**-0.5,) * 62
    # Qwen3.5's layers are Gated DeltaNet's but every full_attention_interval-th, 4 by default
    delta_net = {"ssm.inner_size": 64, "ssm.time_step_rank": 2, "ssm.group_count": 1,
                 "ssm.state_size": 32, "ssm.conv_kernel": 4, "block_count": 8}  # fmt: skip
    assert config("qwen35moe", delta_net).recurrent == (True, True, True, False) * 2
    delta_net["full_attention_interval"] = 2
    assert config("qwen35moe", delta_net).recurrent == (True, False) * 4
    # or as recurrent_layers says, where given
    layers = [True, True, False, True, False, False, True, False]
    assert config("qwen35moe", delta_net | {"attention.recurrent_layers": layers}).recurrent == (
        tuple(layers)
    )
    # a sliding_window_pattern of a period n, as llama.cpp's: every n-th layer sees all positions,
    # the others the window, and 0 all of them; with none, Gemma 3's own period of 6
    gemma = {"attention.sliding_window": 4, "block_count": 12}
    full = config("gemma3", gemma | {"attention.sliding_window_pattern": 6}).windows
    assert full == config("gemma3", gemma).windows == ((4,) * 5 + (0,)) * 2
    assert config("gemma3", gemma | {"attention.sliding_window_pattern": 0}).windows == (4,) * 12
    # sliding-window layers rotate all their dimensions
    partial = {"attention.sliding_window": 4, "attention.sliding_window_pattern": [True, False],
               "rope.dimension_count_swa": 32, "block_count": 2}  # fmt: skip
    with pytest.raises(NotImplementedError, match="partial rotary"):
        config("gemma4", partial)
    # Gemma 4 E2B's and E4B's embeddings per layer and shared keys and values are refused, not
    # left unread
    for values in ({"embedding_length_per_layer_input": 256}, {"attention.shared_kv_layers": 20}):
        with pytest.raises(NotImplementedError, match="per-layer embeddings"):
            config("gemma4", values)


YARN, YARN_SCALE = {"rope.scaling.type": "yarn", "rope.scaling.factor": 4.0}, 1 + 0.1 * math.log(4)


@pytest.mark.parametrize(
    "scaling, scale",
    [
        ({}, 1.0),
        (YARN, 1 + 0.1 * math.log(4)),
        (YARN | {"rope.scaling.attn_factor": 0.5}, 0.5 * (1 + 0.1 * math.log(4))),
        ({"rope.scaling.attn_factor": 0.5}, 0.5),  # with no scaling too, as llama.cpp
        (
            YARN | {"rope.scaling.yarn_log_multiplier": 0.5},
            (1 + 0.1 * math.log(4)) / (1 + 0.05 * math.log(4)),
        ),
    ],
)
def test_rope_scale(scaling, scale):
    # RoPE's cos and sin times YaRN's 1 + 0.1 ln(factor) and rope.scaling.attn_factor, or with a
    # log multiplier, as llama.cpp, mscale(factor, 1) / mscale(factor, multiplier) instead
    cos, sin = rope_table(config("llama", scaling).ropes[0], 8, None)
    np.testing.assert_allclose(np.hypot(cos.numpy(), sin.numpy()), scale, rtol=1e-6)


@pytest.mark.parametrize(
    "rope, factors",
    [
        (Rope(1e6, 128), False),  # Qwen's
        (Rope(500000.0, 128), True),  # Llama 3's, divided by its factors, as LongRoPE's are too
        (Rope(150000.0, 64, scale=1 / 32, yarn=(4096, 32.0, 1.0)), False),  # gpt-oss's YaRN
    ],
)
def test_rope_long_positions(rope, factors):
    # cos and sin at positions to 131072 of each angle as llama.cpp and the models' trainers take
    # it, pos * theta^(-2i/d) in f32, divided and scaled in f32 in its turn; worked out in f64,
    # as tinygrad's f32 cos and sin are 7e-3 off there
    length, d = 131072, rope.dims
    rng = np.random.default_rng(d)
    divide = rng.uniform(1, 8, d // 2).astype(np.float32) if factors else None
    cos, sin = rope_table(rope, length, None if divide is None else Tensor(divide))
    freqs = np.array([rope.theta ** (-2 * i / d) for i in range(d // 2)], dtype=np.float32)
    angles = np.arange(length, dtype=np.float32)[:, None] * freqs
    if divide is not None:
        angles = angles / divide
    mscale = 1.0
    if rope.yarn is not None:
        original, fast, slow = rope.yarn
        span = (d * math.log(original / (beta * 2 * math.pi)) / (2 * math.log(rope.theta))
                for beta in (fast, slow))  # fmt: skip
        low, high = max(0.0, math.floor(next(span))), min(d - 1.0, math.ceil(next(span)))
        ramp = 1 - np.clip((np.arange(d // 2) - low) / max(0.001, high - low), 0, 1)
        angles = angles * (rope.scale * (1 - ramp) + ramp).astype(np.float32)
        mscale = 1 + 0.1 * math.log(1 / rope.scale)
    wide = angles.astype(np.float64)
    np.testing.assert_allclose(cos.numpy(), np.cos(wide) * mscale, rtol=0, atol=1e-5)
    np.testing.assert_allclose(sin.numpy(), np.sin(wide) * mscale, rtol=0, atol=1e-5)


def test_stack_alpha_beta():
    # Qwen3.5's alpha and beta projections stacked as F32, whatever each is stored as
    rng = np.random.default_rng(0)
    alpha = rng.standard_normal((4, 32)).astype(np.float32)
    beta = QTensor(Tensor(random_blocks(GGMLType.Q8_0, 4, rng)), GGMLType.Q8_0, (4, 32))
    layer = {"ssm_alpha": QTensor(Tensor(alpha.ravel()), GGMLType.F32, (4, 32)), "ssm_beta": beta}
    _stack(layer)
    stacked = layer["ssm_alpha_beta"]
    assert stacked.type == GGMLType.F32 and stacked.shape == (8, 32)
    expected = np.concatenate([alpha, beta.dequant().numpy()])
    np.testing.assert_array_equal(stacked.dequant().numpy(), expected)


def test_lfm2_config():
    # LFM2's layers of no kv heads are short convolutions; the first leading_dense_block_count
    # take an MLP, and its experts are weighed by the sigmoid, normalized, as llama.cpp's
    lfm2 = {"attention.head_count_kv": [0, 8, 0, 0], "shortconv.l_cache": 3,
            "leading_dense_block_count": 2, "expert_gating_func": 2}  # fmt: skip
    c = config("lfm2moe", lfm2)
    assert c.recurrent == (True, False, True, True) and c.short_conv == 3 and c.dense_lead == 2
    assert c.sigmoid_gating and c.expert_weights_norm and c.expert_weights_scale == 1.0
    assert not config("qwen3moe").sigmoid_gating


def test_mla_config():
    # DeepSeek V2's multi-head latent attention: scores scaled by each head's whole query and key;
    # its YaRN and its groups of experts, as DeepSeek V3's, are refused rather than run wrong
    mla = {"attention.kv_lora_rank": 512, "attention.q_lora_rank": 768,
           "attention.key_length": 576, "attention.key_length_mla": 256,
           "attention.value_length_mla": 256, "rope.dimension_count": 64,
           "expert_gating_func": 2}  # fmt: skip
    c = config("deepseek2", mla)
    assert c.mla is not None and (c.mla.nope, c.mla.rope, c.mla.v_dim) == (192, 64, 256)
    assert c.scales == (1 / 16,) * 4 and c.head_dims == (576,) * 4 and not c.rope_halves
    for unsupported in (YARN, {"expert_group_used_count": 4}):
        with pytest.raises(NotImplementedError, match="multi-head latent attention"):
            config("deepseek2", mla | unsupported)
    # its experts weighed as llama.cpp's: GLM-4.7-Flash's of a GGUF that leaves the gating function
    # out by the sigmoid, as llama.cpp tells it by its sizes, unnormalized unless the GGUF says so;
    # an unnormalized softmax, or another function, refused
    gated_by_size = {k: v for k, v in mla.items() if k != "expert_gating_func"}
    glm = config("deepseek2", gated_by_size | {"block_count": 47}, vocab=154880)
    assert glm.sigmoid_gating and not glm.expert_weights_norm
    assert not config("deepseek2", mla | {"expert_gating_func": 2}).expert_weights_norm
    for refused in ({"expert_weights_norm": False}, {"expert_gating_func": 3}):
        with pytest.raises(NotImplementedError, match="gating function"):
            config("deepseek2", mla | {"expert_gating_func": 1} | refused)


@pytest.mark.usefixtures("reference_ops")
def test_short_conv_saves_each_state():
    # the state after each token of a run, as a run that stopped there would leave it
    rng = np.random.default_rng(3)
    bcx, conv = rng.standard_normal((1, 5, 24)), rng.standard_normal((8, 3))
    bcx, conv = Tensor(bcx.astype(np.float32)), Tensor(conv.astype(np.float32))
    state, saved = zeros(1, 2, 8), zeros(5, 2, 8)
    whole = ops.short_conv(bcx, conv, state, [Span(0, 0, 5)], saved).numpy()
    for t in range(5):
        alone = zeros(1, 2, 8)
        part = ops.short_conv(bcx[:, : t + 1], conv, alone, [Span(0, 0, t + 1)]).numpy()
        np.testing.assert_allclose(part, whole[:, : t + 1], rtol=1e-6)
        np.testing.assert_allclose(saved[t].numpy(), alone[0].numpy(), rtol=1e-6)


def test_rope_metadata():
    # as llama.cpp: a scaling factor of no type, or of the old key, scales positions linearly;
    # attn_factor scales sliding layers' cos and sin too; gpt-oss's sliding layers take
    # freq_base_swa where given, scaled as the others
    for key in ("rope.scaling.factor", "rope.scale_linear"):
        assert config("llama", {key: 4.0}).ropes[0].scale == 0.25
    gemma = {"attention.sliding_window": 4, "rope.scaling.attn_factor": 0.5, "block_count": 6}
    assert {r.mscale for r in config("gemma3", gemma).ropes} == {0.5}
    oss = {"attention.sliding_window": 4, "rope.freq_base": 1e5, "block_count": 2} | YARN
    assert [r.theta for r in config("gpt-oss", oss).ropes] == [1e5, 1e5]
    ropes = config("gpt-oss", oss | {"rope.freq_base_swa": 5e5}).ropes
    assert [r.theta for r in ropes] == [5e5, 1e5] and ropes[0].yarn == ropes[1].yarn
    # Mellum's sliding layers, every 4th but one by default, take the others' base, unscaled
    mellum = {"attention.sliding_window": 4, "rope.freq_base": 5e5, "block_count": 4} | YARN
    ropes = config("mellum", mellum).ropes
    assert [r.theta for r in ropes] == [5e5] * 4
    assert [r.yarn is not None for r in ropes] == [False, False, False, True]


def test_longrope(tiny):
    # Phi-3's frequencies are divided by LongRoPE's long factors past its original context, and
    # by its short ones within it
    path, weights = tiny("phi3")
    for context, kind in ((P3_ORIGINAL, "short"), (CONTEXT, "long")):
        cos, _ = transformer(path, context).rope[0]
        freqs = 10000.0 ** (-np.arange(0, P3_ROTATED, 2) / P3_ROTATED)
        angles = np.arange(context)[:, None] * freqs / weights[f"rope_factors_{kind}.weight"]
        np.testing.assert_allclose(cos.numpy(), np.cos(angles) * P3_MSCALE, rtol=1e-5, atol=1e-5)


@pytest.mark.usefixtures("reference_ops")
@pytest.mark.parametrize("arch", ARCHS)
def test_generate_matches_reference(tiny, arch):
    path, weights = tiny(arch)
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=5)  # chunks of 5, 5 and 2
    out = list(engine.generate(PROMPT, 8))
    # every generated token is the reference argmax given all tokens before it
    expected = reference_logits(weights, PROMPT + out, arch)[len(PROMPT) - 1 :].argmax(-1)
    assert out == expected[: len(out)].tolist()

    # a prompt that extends the previous one reuses the cache: only the new tokens are prefilled,
    # but for a model with recurrent state, which holds every token its slot ran
    longer = PROMPT + out[:3] + [7]
    recurrent = arch.startswith("qwen35") or arch == "lfm2moe"
    assert engine.cached_prefix(longer) == (0 if recurrent else len(PROMPT) + 3)
    again = list(engine.generate(longer, 4))
    assert again == list(Engine(path, max_context=CONTEXT, prefill_chunk=5).generate(longer, 4))


def test_split_model(tmp_path):
    # a model of a GGUF in parts, opened by its first, generates as the whole file does
    whole, parts = tmp_path / "whole.gguf", tmp_path / "parts" / "m.gguf"
    parts.parent.mkdir()
    write_tiny_model(whole)
    write_tiny_model(parts, split=5)
    first = sorted(parts.parent.iterdir())[0]
    expected = list(Engine(whole, max_context=CONTEXT).generate(PROMPT, 6))
    assert list(Engine(first, max_context=CONTEXT).generate(PROMPT, 6)) == expected


@pytest.mark.usefixtures("reference_ops")
def test_generate_fills_context(tiny_model):
    # one capture of the decode graph replays correctly at every position up to the last
    path, weights = tiny_model
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=8)
    out = list(engine.generate(PROMPT, 1000))
    assert len(out) == CONTEXT - len(PROMPT) + 1
    expected = reference_logits(weights, PROMPT + out[:-1])[len(PROMPT) - 1 :].argmax(-1)
    assert out == expected.tolist()
    captured = engine._decode[1].captured
    engine.reset()
    assert list(engine.generate(PROMPT, 1000)) == out and engine._decode[1].captured is captured


@pytest.mark.usefixtures("reference_ops")
def test_prefill_graphs(tiny_model):
    # chunks of more than FEW_TOKENS tokens take the graph bound to prefill_chunk, and of up to
    # FEW_TOKENS, a single token too, the one bound to FEW_TOKENS
    path, _ = tiny_model
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=FEW_TOKENS + 4)
    prompt = PROMPT * 3  # chunks of FEW_TOKENS + 4 and 36 - FEW_TOKENS - 4
    out = list(engine.generate(prompt, 4))
    assert out == generated(path, prompt, 4)
    assert engine._chunk.captured is not None and engine._few_chunk.captured is not None
    longer = prompt + out[:3] + [7]  # a chunk of one token past the cached ones
    calls = engine._chunk.cnt, engine._few_chunk.cnt
    assert list(engine.generate(longer, 4)) == generated(path, longer, 4)
    assert (engine._chunk.cnt, engine._few_chunk.cnt) == (calls[0], calls[1] + 1)


@pytest.mark.usefixtures("reference_ops")
def test_embed(tiny):
    # an embedding: the normed hidden state of the prompt's last token, of unit length, as the
    # reference's is, through chunks of either graph; the slot it ran in shares no prefix after
    path, weights = tiny("qwen3")
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=FEW_TOKENS + 4)
    prompt = PROMPT * 3  # chunks of FEW_TOKENS + 4 and 36 - FEW_TOKENS - 4
    expected = reference_embedding(weights, prompt, "qwen3")
    np.testing.assert_allclose(engine.embed(prompt), expected, rtol=2e-3, atol=2e-3)
    assert engine._embedding.captured is not None and engine._few_embedding.captured is not None
    assert engine.cached_prefix(prompt) == 0
    assert list(engine.generate(prompt, 3)) == generated(path, prompt, 3)
    with pytest.raises(ValueError, match="prompt must have"):
        engine.embed([])


@pytest.mark.usefixtures("reference_ops")
def test_embed_encoder(tiny):
    # EmbeddingGemma 2's embedding: the mean of its tokens' normed hidden states, each of which
    # sees those within its window either way, projected, of unit length, as the reference's is,
    # of a prompt in one chunk of either graph; one longer than a chunk is refused, and so is
    # generating
    path, weights = tiny("gemma-embedding2")
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=len(PROMPT) * 3)
    for prompt in (PROMPT * 3, PROMPT[:5]):
        expected = reference_embedding(weights, prompt, "gemma-embedding2")
        np.testing.assert_allclose(engine.embed(prompt), expected, rtol=2e-3, atol=2e-3)
    assert engine._embedding.captured is not None and engine._few_embedding.captured is not None
    with pytest.raises(ValueError, match="prompt must have 1 to 36 tokens"):
        engine.embed(PROMPT * 4)
    with pytest.raises(ValueError, match="embeds"):
        engine.start(PROMPT, 1)


@pytest.mark.gpu
def test_embed_encoder_kernels(tiny):
    # the kernels embed as the f64 reference does but for their int8 activations' rounding
    path, weights = tiny("gemma-embedding2")
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=len(PROMPT) * 3)
    for prompt in (PROMPT * 3, PROMPT[:5]):
        expected = reference_embedding(weights, prompt, "gemma-embedding2")
        assert np.dot(engine.embed(prompt), expected) > 0.999


def generated(path, prompt: list[int], n: int, sampling=GREEDY, seed=None) -> list[int]:
    # what an engine with nothing cached generates
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=8)
    return list(engine.generate(prompt, n, sampling, seed))


def prefill_starts(engine: Engine, monkeypatch) -> list[int]:
    # records the first position of each prefilled chunk
    starts, prefill = [], engine._prefill

    def spy(sequence, size):
        starts.append(len(engine._cached[sequence.slot]))
        return prefill(sequence, size)

    monkeypatch.setattr(engine, "_prefill", spy)
    return starts


@pytest.mark.usefixtures("reference_ops")
def test_slots_keep_their_sequences(tiny_model, monkeypatch):
    # a sequence in one slot leaves the keys and values of the others as they were
    path, _ = tiny_model
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=8, slots=2)
    out = list(engine.generate(PROMPT, 6))
    held = len(PROMPT) + 5  # the last token was never run
    before = engine.model.cache[0][:, 0, :, :held].numpy()
    other = [9, 8, 7, 6, 5]
    assert list(engine.generate(other, 6)) == generated(path, other, 6)  # in the other slot
    np.testing.assert_array_equal(engine.model.cache[0][:, 0, :, :held].numpy(), before)

    # the conversation continues in its slot, from where it was
    starts, longer = prefill_starts(engine, monkeypatch), PROMPT + out + [7]
    assert list(engine.generate(longer, 4)) == generated(path, longer, 4)
    assert starts == [held]


@pytest.mark.usefixtures("reference_ops")
def test_shared_prefix_is_copied(tiny_model, monkeypatch):
    # a prompt that leaves a slot's tokens takes another slot, starting from a copy of the prefix
    path, _ = tiny_model
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=8, slots=2)
    list(engine.generate(PROMPT, 4))
    starts, branch = prefill_starts(engine, monkeypatch), PROMPT[:9] + [1, 2, 3]
    assert list(engine.generate(branch, 6)) == generated(path, branch, 6)
    assert starts == [9]
    # and the first conversation's tokens are still cached
    longer = PROMPT + [4]
    assert list(engine.generate(longer, 4)) == generated(path, longer, 4)
    assert starts == [9, len(PROMPT)]


@pytest.mark.usefixtures("reference_ops")
def test_q8_0_cache_generates_as_reference(tiny_model, monkeypatch):
    # every token the reference's argmax as it rounds keys and values to q8_0, as decoding reads
    # them; and a prefix copied to another slot as its rows
    monkeypatch.setattr("tests.helpers.cached", q8_0)
    path, weights = tiny_model
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=8, slots=2, cache_type="q8_0")
    out = list(engine.generate(PROMPT, 8))
    expected = reference_logits(weights, PROMPT + out)[len(PROMPT) - 1 :].argmax(-1)
    assert out == expected[: len(out)].tolist()
    starts, branch = prefill_starts(engine, monkeypatch), PROMPT[:9] + [1, 2, 3]
    fresh = Engine(path, max_context=CONTEXT, prefill_chunk=8, cache_type="q8_0")
    assert list(engine.generate(branch, 6)) == list(fresh.generate(branch, 6))
    assert starts == [9]


@pytest.mark.usefixtures("reference_ops")
def test_owners_share_their_own_prefixes(tiny_model, monkeypatch):
    # a slot's tokens are its owner's: another's prompt that shares them starts over in a slot of
    # its own, and the owner's still goes on from them; a slot of another's that a prompt takes,
    # the least recently used, starts anew
    path, _ = tiny_model
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=8, slots=2)
    list(engine.generate(PROMPT, 4, owner="a"))
    longer = PROMPT + [4]
    assert engine.cached_prefix(longer, "a") == len(PROMPT)
    assert engine.cached_prefix(longer, "b") == engine.cached_prefix(longer) == 0
    starts, branch = prefill_starts(engine, monkeypatch), PROMPT[:9] + [1, 2, 3]
    assert list(engine.generate(branch, 6, owner="b")) == generated(path, branch, 6)
    assert starts == [0, 8]
    assert list(engine.generate(longer, 4, owner="a")) == generated(path, longer, 4)
    assert starts == [0, 8, len(PROMPT)]
    assert list(engine.generate(branch + [5], 4, owner="c")) == generated(path, branch + [5], 4)
    assert starts == [0, 8, len(PROMPT), 0, 8]
    assert engine.cached_prefix(branch + [6], "b") == 0
    assert engine.cached_prefix(longer + [6], "a") == len(longer)


@pytest.mark.usefixtures("reference_ops")
def test_recurrent_state_resumes_for_its_owner(tiny):
    # a slot's kept recurrent state goes on only for its owner's prompts
    path, _ = tiny("qwen35moe")
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=8)
    prompt = (PROMPT * 2)[:20]
    list(engine.generate(prompt, 3, owner="a"))
    turn = prompt[:18] + [7, 1, 2]
    assert engine.cached_prefix(turn, "a") == len(prompt) - KEEP_BACK
    assert engine.cached_prefix(turn, "b") == 0


@pytest.mark.usefixtures("reference_ops")
def test_recurrent_state_shares_whole_slots(tiny, monkeypatch):
    # recurrent state holds all a slot ran: a prompt that shares part of a slot's tokens starts
    # over in another, and one that shares all of them goes on from there
    path, _ = tiny("qwen35moe")
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=8, slots=2)
    out = list(engine.generate(PROMPT, 4))
    starts, branch = prefill_starts(engine, monkeypatch), PROMPT[:9] + [1, 2, 3]
    assert list(engine.generate(branch, 6)) == generated(path, branch, 6)
    longer = PROMPT + out + [7]
    assert list(engine.generate(longer, 4)) == generated(path, longer, 4)
    assert starts == [0, 8, len(PROMPT) + 3]


@pytest.mark.usefixtures("reference_ops")
def test_recurrent_state_resumes_where_kept(tiny, monkeypatch):
    # a prompt that shares another but for its last tokens, as a chat's next turn, goes on from
    # the state kept KEEP_BACK tokens before that one's end: here from 4 of 20, a chunk of 1
    # ending where the new prompt's state is kept, at 5 of 21, then chunks of 8
    path, _ = tiny("qwen35moe")
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=8)
    prompt = (PROMPT * 2)[:20]
    list(engine.generate(prompt, 3))
    starts, turn = prefill_starts(engine, monkeypatch), prompt[:18] + [7, 1, 2]
    assert engine.cached_prefix(turn) == len(prompt) - KEEP_BACK  # as the server reports it
    assert list(engine.generate(turn, 4)) == generated(path, turn, 4)
    assert starts == [len(prompt) - KEEP_BACK, len(turn) - KEEP_BACK, 13]


@pytest.mark.usefixtures("reference_ops")
def test_copy_takes_recurrent_state(tiny):
    model = transformer(tiny("qwen35moe")[0], slots=2)
    model(Tensor([PROMPT], dtype=dtypes.int32), 0, 0)
    model.copy(0, 1)
    after = Tensor([[7]], dtype=dtypes.int32)
    first, copied = (model.logits(model(after, len(PROMPT), slot)).numpy() for slot in (0, 1))
    np.testing.assert_array_equal(first, copied)


@pytest.mark.parametrize(
    "arch, slots, batches",
    [("llama", 5, [1, 2, 4, 5]), ("qwen35moe", 2, [1, 2]), ("gemma4", 2, [1, 2])],
)
def test_warm_up(tiny, tiny_assistant, monkeypatch, arch, slots, batches):
    # compiles every graph, the copy's and every batch's too, and keeping and restoring recurrent
    # state's, and a drafter's speculative step's, and leaves nothing cached that a generation
    # could see. Few tokens are 4 here: the reference Gated DeltaNet's graphs grow with their
    # tokens.
    monkeypatch.setattr("leatengine.engine.FEW_TOKENS", 4)
    monkeypatch.setattr("leatengine.engine.KEEP_BACK", 4)
    path, _ = tiny(arch)
    draft = {"gemma4": tiny_assistant[0], "qwen35moe": path}.get(arch)
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=8, slots=slots, draft=draft)
    engine.warm_up()
    graphs = [engine._chunk, engine._few_chunk, *engine._decode.values(), engine._copy]
    graphs += [engine._keep, engine._restore] if arch == "qwen35moe" else []
    if draft:  # of each number of sequences
        graphs += [*engine._speculate.values(), *engine._settle.values()]
        assert list(engine._speculate) == list(range(1, min(slots, engine.drafter.sequences) + 1))
    assert list(engine._decode) == batches
    captured = [jit.captured for jit in graphs]
    assert all(captured) and engine.cached_prefix(PROMPT) == 0
    assert list(engine.generate(PROMPT, 6)) == generated(path, PROMPT, 6)
    assert [jit.captured for jit in graphs] == captured


def test_graph_binds_kernel_variables():
    # a graph whose kernels bind a variable inside, as matmul's on RDNA, replays with its value:
    # TinyJit alone takes values from a graph's arguments
    groups = VARIABLES["groups"].unbind()[0]
    replayed = graph(lambda x: (x + Tensor.arange(8).float()[: groups.bind(4)].sum()).realize())
    assert [replayed(Tensor([1.0])).item() for _ in range(3)] == [7.0] * 3


def test_kernel_bodies_canonicalized_once():
    # tinygrad canonicalizes a call's body when it schedules it, numbering its local buffers: the
    # calls of one body share the form tinygrad's own function gives, worked out once
    import tinygrad.schedule as schedule

    from leatengine.kernels.lowering import _CANONICAL

    local = UOp.alloc((4,), dtypes.float32, addrspace=AddrSpace.LOCAL)
    body = UOp.sink(local.index(UOp.const(0, dtypes.int32)).store(UOp.const(1.0, dtypes.float32)))
    calls = [body.call(UOp.variable(name, 0, 9)) for name in "ab"]
    canonical = [schedule.canonicalize_call_body(c) for c in calls]
    own = [schedule.canonicalize_call_body.__wrapped__(c) for c in calls]
    assert all(c is o for c, o in zip(canonical, own, strict=True)) and body in _CANONICAL
    assert canonical[0].body is canonical[1].body is not body
    plain = UOp.sink(UOp.const(1.0, dtypes.float32)).call(UOp.variable("a", 0, 9))
    assert schedule.canonicalize_call_body(plain) is plain and _CANONICAL[plain.body] is None


def test_lowered_programs_kept(monkeypatch):
    # a program tinygrad lowers goes to its disk cache, where a process that has not lowered it
    # finds it; a change to the sources that lower it lowers it anew
    import tinygrad.codegen as codegen

    from leatengine.kernels import lowering

    programs, before = codegen.to_program_cache, set(dict.keys(codegen.to_program_cache))
    (Tensor.arange(13) * 17 + Tensor.ones(13)).realize()
    key = next(iter(set(dict.keys(programs)) - before))
    program = dict.pop(programs, key)
    assert key in programs and programs[key].key == program.key
    dict.pop(programs, key)
    with Context(CCACHE=0):  # as tinygrad keeps no binaries, no programs either
        assert key not in programs
    fingerprint = lowering._fingerprint()
    monkeypatch.setenv("ALIGNED", "0")  # a switch tinygrad's C renderers read, which programs hold
    assert lowering._fingerprint.__wrapped__() != fingerprint
    with monkeypatch.context() as m:  # of another tinygrad, while the row is whole
        m.setattr(lowering, "_fingerprint", lambda: "another tinygrad")
        assert key not in programs and programs.get(key) is None
    # a row a flipped bit damaged is lowered anew rather than served
    digest, data = diskcache_get(lowering.PROGRAMS, name := lowering._program(key))
    diskcache_put(lowering.PROGRAMS, name, (digest, bytes([data[0] ^ 1]) + data[1:]))
    assert key not in programs


def test_schedules_kept(monkeypatch):
    # a schedule tinygrad makes goes to its disk cache too, where a process that has not made it
    # finds it; another tinygrad, or a damaged row, schedules anew
    import tinygrad.schedule as schedule

    from leatengine.kernels import lowering

    schedules, before = schedule.schedule_cache, set(dict.keys(schedule.schedule_cache))
    (Tensor.arange(11) * 19 - Tensor.ones(11)).realize()
    key = next(iter(set(dict.keys(schedules)) - before))
    made = dict.pop(schedules, key)
    assert key in schedules and schedules[key].key == made.key
    dict.pop(schedules, key)
    with monkeypatch.context() as m:
        m.setattr(lowering, "_fingerprint", lambda: "another tinygrad")
        assert key not in schedules and schedules.get(key) is None
    digest, data = diskcache_get(lowering.SCHEDULES, name := lowering._schedule(key))
    diskcache_put(lowering.SCHEDULES, name, (digest, bytes([data[0] ^ 1]) + data[1:]))
    assert key not in schedules


def test_one_generation_at_a_time(tiny_model):
    engine = Engine(tiny_model[0], max_context=CONTEXT, prefill_chunk=8, slots=2)
    tokens = engine.generate(PROMPT, 6)
    next(tokens)
    with pytest.raises(RuntimeError, match="unfinished"):
        next(engine.generate(PROMPT, 6))
    tokens.close()
    next(engine.generate(PROMPT, 6))


def test_end_of_generation(tiny_model, monkeypatch):
    # a generation ends at an end-of-generation token, which it yields, unless it ignores them
    engine = Engine(tiny_model[0], max_context=CONTEXT, prefill_chunk=8)
    out = list(engine.generate(PROMPT, 8))
    monkeypatch.setattr(engine.tokenizer, "eog_ids", {out[3]})
    assert list(engine.generate(PROMPT, 8)) == out[: out.index(out[3]) + 1]
    assert list(engine.generate(PROMPT, 8, ignore_eog=True)) == out


def test_least_recently_used_slot(tiny_model):
    # a prompt that shares no slot's tokens takes the slot that least recently started one
    engine = Engine(tiny_model[0], max_context=CONTEXT, prefill_chunk=8, slots=2)
    for prompt in (PROMPT, [9, 8, 7, 6], PROMPT + [1], [3, 3, 3]):
        list(engine.generate(prompt, 2))
    assert engine.cached_prefix(PROMPT + [5]) == len(PROMPT)
    assert engine.cached_prefix([9, 8, 7, 6, 5]) == 0


def test_seeded_sampling(tiny_model):
    # a seed repeats a sampled generation, whatever part of its prompt was cached; graphs captured
    # on the first call draw anew in every generation without one
    engine = Engine(tiny_model[0], max_context=CONTEXT, prefill_chunk=8)
    first = list(engine.generate(PROMPT, 8, Sampling(1.0), seed=1))
    assert list(engine.generate(PROMPT, 8, Sampling(1.0), seed=1)) == first  # prefix cached
    engine.reset()
    assert list(engine.generate(PROMPT, 8, Sampling(1.0), seed=1)) == first
    assert len({tuple(engine.generate(PROMPT, 8, Sampling(1.0))) for _ in range(3)}) == 3


def test_presence_penalty(tiny_model):
    # off the logits of every token a generation has generated, the first from its prompt's last
    # chunk, a single token here, and none of another generation's
    engine = Engine(tiny_model[0], max_context=CONTEXT, prefill_chunk=8)
    penalized = Sampling(presence_penalty=100.0)
    out = list(engine.generate(PROMPT, 12, penalized, ignore_eog=True))
    assert len(set(out)) == len(out)
    list(engine.generate(PROMPT[:11], 4, ignore_eog=True))  # 11 of its 12 tokens cached
    assert list(engine.generate(PROMPT, 12, penalized, ignore_eog=True)) == out
    # but for a format's markup, which a reply writes again and again
    special = engine.tokenizer.special_ids
    assert special and engine._text.numpy().tolist() == [
        i not in special for i in range(len(engine._text.numpy()))]  # fmt: skip


@pytest.mark.usefixtures("reference_ops")
def test_penalties(tiny_model):
    # greedy tokens under every penalty, each the reference's likeliest once they change its
    # logits: repetition_penalty of the text tokens of the prompt and those generated, then the
    # others of those generated, counted; the prompt's whether cached or not
    path, weights = tiny_model
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=8)
    sampling = Sampling(presence_penalty=0.3, frequency_penalty=0.4, repetition_penalty=1.5)
    out = list(engine.generate(PROMPT, 12, sampling, ignore_eog=True))
    text = engine._text.numpy()
    logits = reference_logits(weights, PROMPT + out)[len(PROMPT) - 1 :]
    for i, token in enumerate(out):
        row, counts = logits[i].copy(), np.bincount(out[:i], minlength=len(text)) * text
        repeated = (counts > 0) | np.isin(np.arange(len(text)), PROMPT) & text
        row = np.where(repeated, np.where(row > 0, row / 1.5, row * 1.5), row)
        assert token == np.argmax(row - 0.4 * counts - 0.3 * (counts > 0)), i
    list(engine.generate(PROMPT[:11], 4, ignore_eog=True))  # 11 of its 12 tokens cached
    assert list(engine.generate(PROMPT, 12, sampling, ignore_eog=True)) == out


def test_logit_bias(tiny_model):
    # a token biased up is all a greedy sequence takes, and its slot's next sequence, unbiased,
    # generates as it would have
    path, _ = tiny_model
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=8)
    assert list(engine.generate(PROMPT, 5, logit_bias={7: 100.0}, ignore_eog=True)) == [7] * 5
    assert list(engine.generate(PROMPT[::-1], 5)) == generated(path, PROMPT[::-1], 5)
    with pytest.raises(ValueError, match="logit_bias names tokens past the vocab"):
        engine.start(PROMPT, 1, logit_bias={10**6: 1.0})


# ******** constrained replies ********

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"enum": ["x", "yz"]}},
          "required": ["a", "b"], "additionalProperties": False}  # fmt: skip


def constrained(engine: Engine, grammar: Grammar, n: int, seed: int) -> tuple[list[int], bool]:
    # a sampled reply held to a grammar, and whether its JSON is whole
    constraint = engine.constrain(grammar)
    out = list(engine.generate(PROMPT, n, Sampling(1.0), seed=seed, constraint=constraint))
    return out, constraint.matcher.done(constraint.state)


def check_constrained(engine: Engine) -> None:
    # sampled replies held to a schema: each a JSON value of it where whole, ended there, as the
    # tiny model has no end token, and a prefix of one where max_tokens cut it
    grammar, whole = Grammar(SCHEMA), 0
    for seed in range(6):
        out, done = constrained(engine, grammar, 40, seed)
        if not done:
            assert len(out) == 40
            continue
        value = json.loads(engine.tokenizer.decode(out))
        assert isinstance(value["a"], int) and value["b"] in ("x", "yz") and len(value) == 2
        whole += 1
    assert whole


@pytest.mark.usefixtures("reference_ops")
def test_constrained(tiny_model):
    check_constrained(Engine(tiny_model[0], max_context=CONTEXT, prefill_chunk=8))


@pytest.mark.usefixtures("reference_ops")
def test_constrained_beside_free(tiny_model):
    # a constrained sequence leaves those decoding beside it as they would be alone, and its slot,
    # once it ends, free to the next
    path, _ = tiny_model
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=8, slots=2)
    constraint = engine.constrain(Grammar(None))
    held = engine.start(PROMPT, 12, Sampling(1.0), seed=1, constraint=constraint)
    free = engine.start(PROMPT[::-1], 12, Sampling(1.0), seed=2)
    while engine.active:
        engine.step()
    assert held.tokens and free.tokens == generated(path, PROMPT[::-1], 12, Sampling(1.0), 2)
    again = list(engine.generate(PROMPT, 12, Sampling(1.0), seed=3))
    assert again == generated(path, PROMPT, 12, Sampling(1.0), 3)


@pytest.mark.gpu
def test_constrained_kernels(tiny_model):
    # through the kernels, the cutoff's too, of the few tokens a grammar leaves
    check_constrained(Engine(tiny_model[0], max_context=CONTEXT, prefill_chunk=8))


def check_logprobs(engine: Engine, weights: dict, tolerance: float) -> None:
    # each token's log-probability and the likeliest tokens', of the reference's logits as they
    # are, of two sequences beside each other: one sampled, held to JSON, keeping 5 of the
    # likeliest, the first of the prompt's last chunk; one greedy keeping none
    many = engine.start(
        PROMPT, 6, Sampling(1.0, top_k=3), seed=3, constraint=engine.constrain(Grammar(None)),
        logprobs=5,
    )  # fmt: skip
    engine.step()
    none = engine.start(PROMPT[::-1], 6, logprobs=0)
    while engine.active:
        engine.step()
    for s, prompt in ((many, PROMPT), (none, PROMPT[::-1])):
        start = len(prompt) - 1
        logits = reference_logits(weights, prompt + s.tokens)[start : start + len(s.tokens)]
        shifted = logits - logits.max(-1, keepdims=True)
        expected = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
        assert len(s.scores) == len(s.tokens) == 6
        for row, token, (chosen, likeliest) in zip(expected, s.tokens, s.scores, strict=True):
            assert chosen == pytest.approx(row[token], abs=tolerance)
            ids, values = [i for i, _ in likeliest], [v for _, v in likeliest]
            assert len(ids) == s.logprobs and values == sorted(values, reverse=True)
            assert values == pytest.approx(row[ids].tolist(), abs=tolerance)
            assert values == pytest.approx(np.sort(row)[::-1][: s.logprobs].tolist(), abs=tolerance)


@pytest.mark.usefixtures("reference_ops")
def test_logprobs(tiny_model):
    path, weights = tiny_model
    check_logprobs(Engine(path, max_context=CONTEXT, prefill_chunk=8, slots=2), weights, 1e-3)


@pytest.mark.gpu
def test_logprobs_kernels(tiny_model):
    path, weights = tiny_model
    check_logprobs(Engine(path, max_context=CONTEXT, prefill_chunk=8, slots=2), weights, 0.05)


def test_logprobs_of_twenty_at_most(tiny_model):
    engine = Engine(tiny_model[0], max_context=CONTEXT)
    with pytest.raises(ValueError, match="logprobs is of 0 to 20 tokens, not 21"):
        engine.start(PROMPT, 1, logprobs=21)


# ******** several sequences at once ********


def run_all(engine: Engine, starts: dict[int, tuple]) -> dict[int, list[int]]:
    # steps until every sequence is done, starting each of `starts` {step: start() arguments} at
    # its step; returns each one's tokens, as step() gave them, by its step
    sequences, out, n = {}, {}, 0
    while n == 0 or engine.active or n <= max(starts):
        if n in starts:
            sequences[n] = engine.start(*starts[n])
            out[n] = []
        for sequence, token in engine.step():
            out[next(k for k, s in sequences.items() if s is sequence)].append(token)
        n += 1
    assert all(sequences[k].tokens == tokens for k, tokens in out.items())
    return out


@pytest.mark.usefixtures("reference_ops")
@pytest.mark.parametrize("arch", ["llama", "qwen3moe", "gemma4", "gpt-oss", "qwen35moe"])
def test_batched_matches_alone(tiny, arch):
    # sequences that join and leave a batch, 3 padded to 4 too, generate what each would alone:
    # greedy or seeded, and with prompts as long as a chunk or shared in part
    path, _ = tiny(arch)
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=8, slots=4)
    starts = {
        0: (PROMPT, 9),
        1: (PROMPT[:9] + [1, 2], 6, Sampling(1.0, top_k=3, presence_penalty=1.0), 5),
        3: ([4, 2], 4),
        12: (PROMPT[::-1] * 2, 5, Sampling(0.8, top_p=0.9), 7),
    }
    got = run_all(engine, starts)
    for n, args in starts.items():
        assert got[n] == generated(path, *args), n


@pytest.mark.usefixtures("reference_ops")
def test_batches_past_the_largest(tiny_model, monkeypatch):
    # more sequences than a decode graph takes run in batches in turn, two of the same graph
    monkeypatch.setattr("leatengine.engine.BATCH", 2)
    path, _ = tiny_model
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=8, slots=5)
    assert list(engine._decode) == [1, 2]
    starts = {
        0: (PROMPT, 7),
        1: ([4, 2], 6),
        2: (PROMPT[3:], 5, Sampling(1.0, min_p=0.1), 3),
        3: ([7, 7, 1], 6),
        4: ([9], 4),
    }
    got = run_all(engine, starts)
    for n, args in starts.items():
        assert got[n] == generated(path, *args), n


@pytest.mark.usefixtures("reference_ops")
def test_prefill_shares_steps(tiny_model):
    # a long prompt prefills a chunk per step, while the sequences past their prompts each get a
    # token in every one of those steps
    path, _ = tiny_model
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=4, slots=2)
    first = engine.start(PROMPT, 20)
    while not first.tokens:
        engine.step()
    second = engine.start(PROMPT[::-1] + [3, 1], 2)  # 14 tokens: 4 chunks
    steps = [[s for s, _ in engine.step()] for _ in range(4)]
    assert steps == [[first]] * 3 + [[second, first]]
    assert len(first.tokens) == 5 and len(second.tokens) == 1


@pytest.mark.usefixtures("reference_ops")
def test_prefill_shares_smaller_chunks(tiny_model, monkeypatch):
    # while others decode, a prompt prefills in chunks of SHARED_CHUNK at most, and alone in
    # chunks of prefill_chunk
    monkeypatch.setattr("leatengine.engine.SHARED_CHUNK", 3)
    path, _ = tiny_model
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=8, slots=2)
    starts = prefill_starts(engine, monkeypatch)
    first = engine.start(PROMPT, 20)  # 12 tokens: chunks of 8 and 4, alone
    while not first.tokens:
        engine.step()
    second = engine.start([9, 8, 7, 6, 5, 4, 3], 2)  # chunks of 3, 3 and 1, beside the first
    while not second.tokens:
        engine.step()
    assert starts == [0, 8, 0, 3, 6]
    assert second.tokens == generated(path, [9, 8, 7, 6, 5, 4, 3], 1)


@pytest.mark.usefixtures("reference_ops")
def test_cancel_mid_prompt(tiny_model, monkeypatch):
    # a sequence cancelled partway through its prompt frees its slot, which keeps the chunks it
    # prefilled: the same prompt continues from there, and generates as it would from scratch
    path, _ = tiny_model
    engine = Engine(path, max_context=CONTEXT, prefill_chunk=4, slots=1)
    sequence = engine.start(PROMPT, 6)
    engine.step()
    engine.step()
    engine.cancel(sequence)
    assert not engine.active and engine.cached_prefix(PROMPT) == 8
    starts = prefill_starts(engine, monkeypatch)
    assert list(engine.generate(PROMPT, 6)) == generated(path, PROMPT, 6)
    assert starts == [8]


def test_start_needs_a_free_slot(tiny_model):
    engine = Engine(tiny_model[0], max_context=CONTEXT, prefill_chunk=8, slots=2)
    first, _ = engine.start(PROMPT, 4), engine.start([3, 1, 4], 4)
    with pytest.raises(RuntimeError, match="all 2 slots"):
        engine.start([1, 5, 9], 4)
    with pytest.raises(RuntimeError, match="unfinished"):
        next(engine.generate([1, 5, 9], 4))
    engine.cancel(first)
    assert first.done and first not in engine.active
    engine.start([1, 5, 9], 4)


# the most mean KL divergence from llama.cpp and the least agreement on the top token each
# architecture allows. Both paths score about 0.0012 and 98% on Llama 3.1 8B, int8 activations
# adding noise as in llama.cpp; for scale, dropping its rope frequency factors, a subtle bug, scored
# 0.0026 on the reference ops. Qwen3 8B, more sensitive, scores 0.0025 and 98% on those too,
# Qwen2.5 7B 0.0031 and both paths 0.0036, and Qwen3 30B A3B 0.003 to 0.0053, where noise also
# flips a token's choice of experts. The top token is the noisier measure: on the 1020 positions
# here, a change of rounding in the matrix kernels took Llama's decode path's KL from 0.00128 to
# 0.00126 and its agreement from 98.2 to 97.9%. Qwen3.6 35B A3B, of 256 experts and recurrent
# state, scores 0.0082 and 97.4% on both paths. Mistral Small 3.2 24B, of the llama
# architecture, scores 0.0021 and 98.1% on the prompt's path, and 0.0025 and 98.4% on the decode
# path, its depth adding to the noise, as its 0.0017 over 20 chunks did in the original Leat.
# Qwen3.8 27B, dense, of recurrent state, scores 0.0014 and 98.1% on 20 chunks of the prompt's
# path; its decode path's scoring takes more than the 3090's 24 GB. The newer architectures'
# measurements, over 8 chunks against llama.cpp 1b70350, are in docs/benchmarks.md: Mellum's
# 0.008 and 95.9%; LFM2.5's 0.057 and 88.6%, a model whose short convolutions make it sensitive
# to rounding on every backend (llama.cpp's CPU backend scores 0.039 against its CUDA one); and
# Muse Glimmer's 0.0021 and 98.4% or more. GLM-4.7-Flash scores 0.032 to 0.040 and 95% over 7
# chunks, where llama.cpp 1b70350 and 2e88c49c9 differ from each other by 0.039; leatengine scores
# 0.011 against the older build. Over the 4 chunks here, the two builds differ by 0.063 and agree
# on 93.5% of top tokens, and leatengine scores 0.051 and 0.066 against the newer one. Unsloth's
# 3-bit quantizations of Qwen3 30B A3B, of Q3_K and of IQ3_XXS and IQ3_S, score 0.0050 to 0.0061 and
# 96.9 to 97.2% over 8 chunks, where llama.cpp's CPU backend scores 0.0088 to 0.0098 and 95.8 to
# 96.1% against its CUDA one. 2-bit files pick their top token among closer candidates still, and
# get twice their architecture's KL: UD-IQ2_XXS, of IQ2_XXS, IQ2_XS and IQ2_S, scores 0.0052 to
# 0.0054 and 96.3 to 96.8% over 8 chunks, where llama.cpp's CPU backend scores 0.0105 and 95.2%
# against its CUDA one; UD-Q2_K_XL scores 0.0086 and 97% over 8 chunks and 0.011 over the 4 here,
# more than half of it at one token: a digit llama.cpp's CUDA backend gives 94%, which its CPU
# backend and leatengine all but rule out, at a KL of 5.7 and 6.7 from it.
TWO_BIT = {10, 19, 20, 21, 28, 29}  # general.file_type: Q2_K, IQ2_XXS, IQ2_XS, Q2_K_S, IQ2_S, IQ2_M
LIMITS = {
    "llama": (0.0015, 0.975), "qwen2": (0.0045, 0.97), "qwen3": (0.0035, 0.97),
    "qwen3moe": (0.007, 0.96), "qwen35": (0.003, 0.975), "qwen35moe": (0.01, 0.965),
    "Mistral Small": (0.003, 0.975), "mellum": (0.012, 0.95), "lfm2moe": (0.065, 0.88),
    "deepseek2": (0.08, 0.92), "muse-glimmer": (0.004, 0.975),
}  # fmt: skip


@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.parametrize("decode", [False, True], ids=["prefill", "decode"])
def test_matches_llama_cpp(model_path, llama_cpp, wikitext, tmp_path, decode):
    metadata = GGUF.open(model_path).metadata
    arch, name = metadata["general.architecture"], metadata.get("general.name", "")
    if (limits := next((LIMITS[k] for k in LIMITS if k in name), LIMITS.get(arch))) is None:
        pytest.skip(f"{arch} is instruction-tuned only, and scores raw text badly")
    args = ["-m", model_path, "-f", wikitext, "-c", "512", "--chunks", "4"]
    args += ["--kl-divergence-base", base := tmp_path / "base.kld"]
    subprocess.run([llama_cpp / "llama-perplexity", *args], check=True, capture_output=True)
    engine = Engine(model_path, max_context=512)
    quality = bench.kl_divergence(engine, base, decode=decode)
    del engine  # a failure's traceback would hold the model on the GPU through the next tests
    kl, top1 = limits
    if metadata.get("general.file_type") in TWO_BIT:
        kl *= 2
    assert quality.kl_mean is not None and quality.kl_mean < kl
    assert quality.top1 is not None and quality.top1 > top1


@pytest.mark.gpu
@pytest.mark.model
def test_chunked_prefill(model_path):
    # prefilling in chunks of a bound length, through the kernels' symbolic paths, leaves the first
    # layer's keys and values, or its recurrent states, of one pass over the whole prompt, with its
    # fixed shapes, and a next token it scores as likely. Deeper layers drift apart: kernels
    # compiled for either differ in the last bit here and there, which can move an activation to
    # the next int8 step or flip a choice of experts, and so a near tie: on tinygrad's own
    # attention, Qwen2.5 7B's chunked pass once picked a token 0.07 below the best.
    engine = Engine(model_path, max_context=512, prefill_chunk=128)
    vocab, bos = engine.config.vocab_size, engine.tokenizer.bos_id
    # as many tokens as every layer's cache holds in one run: gpt-oss's windows of 128 keep rings
    # of 256 positions, which a run of more than 129 would wrap over the keys its first tokens read
    m = engine.model
    n = min([300] + [size - window + 1 for size, window, ring
                     in zip(m.sizes, m.config.windows, m.rings, strict=True) if ring])  # fmt: skip
    prompt = [0 if bos is None else bos] + [(i * 7919) % vocab for i in range(n - 1)]

    def held() -> list[np.ndarray]:  # what the first layer holds of the prompt
        if (cache := engine.model.cache[0]) is not None:
            return [cache[:, :, :, : len(prompt)].numpy().astype(np.float32)]
        return [state[0].numpy() for state in engine.model.states[0]]

    token = next(engine.generate(prompt, 1))
    chunked = held()
    hidden = engine.model(Tensor([prompt], dtype=dtypes.int32), 0)  # the same positions again
    logits = engine.model.logits(hidden[:, -1]).numpy().reshape(-1)
    assert logits[token] > logits.max() - 0.1
    for got, whole in zip(chunked, held(), strict=True):
        np.testing.assert_allclose(got, whole, rtol=1e-2, atol=1e-2)
