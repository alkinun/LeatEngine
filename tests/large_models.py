# Large models' layers, for tests at their real shapes: of each, its metadata and the tensors
# of a layer or two, of the shapes and types the GGUF named holds (read from its header), with
# random weights and the tiny tokenizer's vocab.
#
# A matrix's blocks are scaled so that inputs of RMS 1 give outputs of about TARGET's, as a real
# model's activations stay. Tokens' embeddings lie along GROUPS directions, a token's by its id
# modulo GROUPS, and each router's rows rank its experts along each direction with wide margins:
# every token takes the same experts whether the kernels or the reference ops score them, as
# their rounding differs by far less. An MTP layer's router ranks them along the directions its
# projection of its inputs takes those to.

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gguf
import numpy as np
from gguf.quants import dequantize
from tinygrad import Tensor, UOp, dtypes

from leatengine.draft import Mtp
from leatengine.gguf import GGUF
from leatengine.model import Config, Transformer
from leatengine.ops import Span
from leatengine.quant import BLOCK, GGMLType
from tests.helpers import V, add_vocab, random_blocks

GROUPS = 4
# the RMS of the outputs of matrices that end a block, of inputs of RMS 1, so that the residual
# stays mostly the embeddings' directions; of others, 1
TARGET = {"attn_output": 0.5, "ssm_out": 0.5, "ffn_down": 0.5, "ffn_down_exps": 0.5,
          "ffn_down_shexp": 0.5}  # fmt: skip
ROUTED = 2.5  # the router's score of a token's best expert, about

Tensors = dict[str, tuple[str, tuple[int, ...]]]  # by name: type and shape, as GGUF has them


@dataclass(frozen=True)
class LargeModel:
    source: str  # the GGUF of the shapes and types: its repository, quantization and layers
    arch: str
    metadata: dict[str, Any]  # the architecture's, without its prefix, but the count of layers
    layers: tuple[Tensors, ...]
    output: str  # the output matrix's type


def _gpt_oss() -> Tensors:
    # Q8_0 attention of 64 heads over 8 kv heads of 64, with biases and sinks; 128 experts of
    # MXFP4, with biases
    dim, q, kv, experts = 2880, 64 * 64, 8 * 64, 128
    layer: Tensors = {
        "attn_norm.weight": ("F32", (dim,)), "post_attention_norm.weight": ("F32", (dim,)),
        "attn_sinks.weight": ("F32", (64,)),
        "ffn_gate_inp.weight": ("F32", (experts, dim)), "ffn_gate_inp.bias": ("F32", (experts,)),
    }  # fmt: skip
    for name, rows, cols in (("attn_q", q, dim), ("attn_k", kv, dim), ("attn_v", kv, dim),
                             ("attn_output", dim, q)):  # fmt: skip
        layer |= {f"{name}.weight": ("Q8_0", (rows, cols)), f"{name}.bias": ("F32", (rows,))}
    for name in ("gate", "up", "down"):
        layer[f"ffn_{name}_exps.weight"] = ("MXFP4", (experts, dim, dim))
        layer[f"ffn_{name}_exps.bias"] = ("F32", (experts, dim))
    return layer


def _qwen3moe(q: str, k: str, v: str, o: str, gate_up: str, down: str) -> Tensors:
    # attention of 64 heads over 4 kv heads of 128, groups of 16; 128 experts of 1536 rows
    dim, heads, kv, head, experts, hidden = 4096, 64, 4, 128, 128, 1536
    return {
        "attn_norm.weight": ("F32", (dim,)), "ffn_norm.weight": ("F32", (dim,)),
        "attn_q_norm.weight": ("F32", (head,)), "attn_k_norm.weight": ("F32", (head,)),
        "attn_q.weight": (q, (heads * head, dim)), "attn_k.weight": (k, (kv * head, dim)),
        "attn_v.weight": (v, (kv * head, dim)), "attn_output.weight": (o, (dim, heads * head)),
        "ffn_gate_inp.weight": ("F32", (experts, dim)),
        "ffn_gate_exps.weight": (gate_up, (experts, hidden, dim)),
        "ffn_up_exps.weight": (gate_up, (experts, hidden, dim)),
        "ffn_down_exps.weight": (down, (experts, dim, hidden)),
    }  # fmt: skip


def _qwen35moe(attention: bool) -> Tensors:
    # 256 experts of 512 rows beside a shared one, after full attention of 16 heads over 2 kv
    # heads of 256, its queries with their gates, or after Gated DeltaNet of 16 key heads and 32
    # value heads of 128
    dim, experts, hidden = 2048, 256, 512
    layer: Tensors = {
        "attn_norm.weight": ("F32", (dim,)), "post_attention_norm.weight": ("F32", (dim,)),
        "ffn_gate_inp.weight": ("F32", (experts, dim)),
        "ffn_gate_exps.weight": ("Q4_K", (experts, hidden, dim)),
        "ffn_up_exps.weight": ("Q4_K", (experts, hidden, dim)),
        "ffn_down_exps.weight": ("Q5_K", (experts, dim, hidden)),
        "ffn_gate_inp_shexp.weight": ("F32", (dim,)),
        "ffn_gate_shexp.weight": ("Q8_0", (hidden, dim)),
        "ffn_up_shexp.weight": ("Q8_0", (hidden, dim)),
        "ffn_down_shexp.weight": ("Q8_0", (dim, hidden)),
    }  # fmt: skip
    if attention:
        return layer | {
            "attn_q.weight": ("Q8_0", (2 * 16 * 256, dim)), "attn_k.weight": ("Q8_0", (512, dim)),
            "attn_v.weight": ("Q8_0", (512, dim)), "attn_output.weight": ("Q8_0", (dim, 4096)),
            "attn_q_norm.weight": ("F32", (256,)), "attn_k_norm.weight": ("F32", (256,)),
        }  # fmt: skip
    return layer | {
        "attn_qkv.weight": ("Q8_0", (8192, dim)), "attn_gate.weight": ("Q8_0", (4096, dim)),
        "ssm_a": ("F32", (32,)), "ssm_alpha.weight": ("F32", (32, dim)),
        "ssm_beta.weight": ("F32", (32, dim)), "ssm_conv1d.weight": ("F32", (8192, 4)),
        "ssm_dt.bias": ("F32", (32,)), "ssm_norm.weight": ("F32", (128,)),
        "ssm_out.weight": ("Q8_0", (dim, 4096)),
    }  # fmt: skip


def _glm4moe(q: str, k: str, v: str, o: str, *mlp: str) -> Tensors:
    # attention of 96 heads over 8 kv heads of 128, with biases of q, k and v; then an MLP of
    # 10944 rows, of the types of its gate and up and of its down, or 128 experts of 1408 rows
    # beside a shared one, of those of the experts' and then the shared one's
    dim, heads, kv, head, experts, hidden = 4096, 96, 8, 128, 128, 1408
    layer: Tensors = {
        "attn_norm.weight": ("F32", (dim,)), "post_attention_norm.weight": ("F32", (dim,)),
        "attn_output.weight": (o, (dim, heads * head)),
    }  # fmt: skip
    for name, kind, rows in (("attn_q", q, heads * head), ("attn_k", k, kv * head),
                             ("attn_v", v, kv * head)):  # fmt: skip
        layer |= {f"{name}.weight": (kind, (rows, dim)), f"{name}.bias": ("F32", (rows,))}
    if len(mlp) == 2:
        gate_up, down = mlp
        return layer | {
            "ffn_gate.weight": (gate_up, (10944, dim)), "ffn_up.weight": (gate_up, (10944, dim)),
            "ffn_down.weight": (down, (dim, 10944)),
        }  # fmt: skip
    gate_up, down, shared_gate_up, shared_down = mlp
    return layer | {
        "ffn_gate_inp.weight": ("F32", (experts, dim)), "exp_probs_b.bias": ("F32", (experts,)),
        "ffn_gate_exps.weight": (gate_up, (experts, hidden, dim)),
        "ffn_up_exps.weight": (gate_up, (experts, hidden, dim)),
        "ffn_down_exps.weight": (down, (experts, dim, hidden)),
        "ffn_gate_shexp.weight": (shared_gate_up, (hidden, dim)),
        "ffn_up_shexp.weight": (shared_gate_up, (hidden, dim)),
        "ffn_down_shexp.weight": (shared_down, (dim, hidden)),
    }  # fmt: skip


def _nextn(dim: int, proj: str, head: str) -> Tensors:
    # an MTP layer's own tensors, before its layer's, as its router reads its projection: the
    # norms of its inputs and its output, the projection of its inputs joined, and its embeddings
    # and output head, its embeddings of F32, as the model's are here
    return {
        "nextn.eh_proj.weight": (proj, (dim, 2 * dim)), "nextn.enorm.weight": ("F32", (dim,)),
        "nextn.hnorm.weight": ("F32", (dim,)), "nextn.shared_head_norm.weight": ("F32", (dim,)),
        "nextn.embed_tokens.weight": ("F32", (V, dim)),
        "nextn.shared_head_head.weight": (head, (V, dim)),
    }  # fmt: skip


GPT_OSS = {
    "context_length": 131072, "embedding_length": 2880, "feed_forward_length": 2880,
    "attention.head_count": 64, "attention.head_count_kv": 8, "attention.key_length": 64,
    "attention.value_length": 64, "attention.sliding_window": 128,
    "attention.layer_norm_rms_epsilon": 1e-5, "rope.freq_base": 150000.0,
    "rope.scaling.type": "yarn", "rope.scaling.factor": 32.0,
    "rope.scaling.original_context_length": 4096, "rope.scaling.yarn_beta_fast": 32.0,
    "rope.scaling.yarn_beta_slow": 1.0, "expert_count": 128, "expert_used_count": 4,
    "expert_feed_forward_length": 2880,
}  # fmt: skip
QWEN3_235B = {
    "context_length": 262144, "embedding_length": 4096, "feed_forward_length": 12288,
    "attention.head_count": 64, "attention.head_count_kv": 4, "attention.key_length": 128,
    "attention.value_length": 128, "attention.layer_norm_rms_epsilon": 1e-6,
    "rope.freq_base": 5e6, "expert_count": 128, "expert_used_count": 8,
    "expert_feed_forward_length": 1536,
}  # fmt: skip
# with every second layer of full attention, of the two kept, rather than every fourth
QWEN36_35B = {
    "context_length": 262144, "embedding_length": 2048, "attention.head_count": 16,
    "attention.head_count_kv": 2, "attention.key_length": 256, "attention.value_length": 256,
    "attention.layer_norm_rms_epsilon": 1e-6, "rope.freq_base": 1e7, "rope.dimension_count": 64,
    "rope.dimension_sections": [11, 11, 10, 0], "expert_count": 256, "expert_used_count": 8,
    "expert_feed_forward_length": 512, "expert_shared_feed_forward_length": 512,
    "ssm.conv_kernel": 4, "ssm.state_size": 128, "ssm.group_count": 16,
    "ssm.time_step_rank": 32, "ssm.inner_size": 4096, "full_attention_interval": 2,
}  # fmt: skip
GLM45_AIR = {
    "context_length": 131072, "embedding_length": 4096, "feed_forward_length": 10944,
    "attention.head_count": 96, "attention.head_count_kv": 8, "attention.key_length": 128,
    "attention.value_length": 128, "attention.layer_norm_rms_epsilon": 1e-5,
    "rope.freq_base": 1e6, "rope.dimension_count": 64, "expert_count": 128,
    "expert_used_count": 8, "expert_feed_forward_length": 1408, "expert_shared_count": 1,
    "expert_group_count": 1, "expert_group_used_count": 1, "leading_dense_block_count": 1,
    "expert_gating_func": 2, "expert_weights_scale": 1.0, "expert_weights_norm": True,
}  # fmt: skip
LARGE_MODELS = {
    # a layer of each kind: of the sliding window and of all positions
    **{
        f"gpt-oss-120b-{kind}": LargeModel(
            f"ggml-org/gpt-oss-120b-GGUF MXFP4, {layer}", "gpt-oss",
            GPT_OSS | {"attention.sliding_window_pattern": [kind == "sliding"]}, (_gpt_oss(),),
            "Q8_0",
        )
        for kind, layer in (("sliding", "blk.0"), ("full", "blk.1"))
    },
    "qwen3-235b-q2_k": LargeModel(
        "unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF UD-Q2_K_XL, blk.0", "qwen3moe", QWEN3_235B,
        (_qwen3moe("Q4_K", "Q4_K", "Q4_K", "Q4_K", "Q2_K", "Q3_K"),), "Q6_K",
    ),
    "qwen3-235b-iq2_xxs": LargeModel(
        "bartowski/Qwen_Qwen3-235B-A22B-Instruct-2507-GGUF IQ2_XS, blk.11", "qwen3moe",
        QWEN3_235B, (_qwen3moe("IQ2_XS", "Q4_K", "Q4_K", "Q4_K", "IQ2_XXS", "IQ2_XXS"),), "Q5_K",
    ),
    "qwen3-235b-iq3_xxs": LargeModel(
        "bartowski/Qwen_Qwen3-235B-A22B-Instruct-2507-GGUF IQ3_XXS, blk.0", "qwen3moe",
        QWEN3_235B, (_qwen3moe("IQ2_S", "Q8_0", "Q4_K", "Q5_K", "IQ3_XXS", "IQ3_S"),), "Q5_K",
    ),
    "qwen3.6-35b-a3b": LargeModel(
        "unsloth/Qwen3.6-35B-A3B-GGUF UD-Q4_K_XL, blk.0 and blk.3", "qwen35moe", QWEN36_35B,
        (_qwen35moe(attention=False), _qwen35moe(attention=True)), "Q8_0",
    ),
    # its dense first layer, a layer of experts whose down projections are Q5_1, as 16 of its 46
    # layers of experts' are, and its MTP layer
    "glm-4.5-air-ud-q4_k_xl": LargeModel(
        "unsloth/GLM-4.5-Air-GGUF UD-Q4_K_XL, blk.0, blk.1 and blk.46", "glm4moe",
        GLM45_AIR | {"nextn_predict_layers": 1}, (
            _glm4moe("Q4_K", "Q4_K", "Q4_K", "Q4_K", "Q4_K", "Q5_0"),
            _glm4moe("Q5_K", "Q5_K", "Q5_K", "Q5_K", "Q4_K", "Q5_1", "Q4_K", "Q8_0"),
            _nextn(4096, "Q4_K", "Q4_K")
            | _glm4moe("Q4_K", "Q4_K", "Q4_K", "Q4_K", "Q4_K", "Q5_0", "Q4_K", "Q8_0"),
        ), "Q6_K",
    ),
    "glm-4.5-air-q4_k_m": LargeModel(
        "bartowski/zai-org_GLM-4.5-Air-GGUF Q4_K_M, blk.0 and blk.1", "glm4moe", GLM45_AIR, (
            _glm4moe("Q4_K", "Q8_0", "Q6_K", "Q5_K", "Q4_K", "Q8_0"),
            _glm4moe("Q4_K", "Q8_0", "Q6_K", "Q5_K", "Q4_K", "Q8_0", "Q8_0", "Q8_0"),
        ), "Q6_K",
    ),
}  # fmt: skip


def write_layers(path: Path, large: LargeModel) -> None:
    """A GGUF of a large model's layers, its tensors written one at a time, as some take
    gigabytes."""
    rng = np.random.default_rng(0)
    dim = large.metadata["embedding_length"]
    tensors: Tensors = {"token_embd.weight": ("F32", (V, dim)),
                        "output.weight": (large.output, (V, dim)),
                        "output_norm.weight": ("F32", (dim,))}  # fmt: skip
    for i, layer in enumerate(large.layers):
        tensors |= {f"blk.{i}.{name}": spec for name, spec in layer.items()}
    w = gguf.GGUFWriter(path, arch=large.arch)
    for key, value in ({"block_count": len(large.layers)} | large.metadata).items():
        key = f"{large.arch}.{key}"
        if isinstance(value, list):
            w.add_array(key, value)
        elif isinstance(value, str):
            w.add_string(key, value)
        else:
            (w.add_float32 if isinstance(value, float) else w.add_uint32)(key, value)
    add_vocab(w)
    for name, (kind, shape) in tensors.items():
        t = GGMLType[kind]
        elements, nbytes = BLOCK[t]
        size = math.prod(shape) // elements * nbytes
        if t == GGMLType.F32:
            w.add_tensor_info(name, shape, np.dtype(np.float32), size)
        else:
            rows = (*shape[:-1], shape[-1] // elements * nbytes)
            w.add_tensor_info(name, rows, np.dtype(np.uint8), size, gguf.GGMLQuantizationType(t))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    # the directions of the groups of tokens, of RMS 1, and their unit vectors
    directions = rng.standard_normal((GROUPS, dim))
    units = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    embeddings = directions[np.arange(V) % GROUPS] + rng.standard_normal((V, dim)) * 0.5
    for name, (kind, shape) in tensors.items():
        if kind != "F32":
            blocks = _blocks(GGMLType[kind], shape, rng, _target(name))
            w.write_tensor_data(blocks)
            if name.endswith("nextn.eh_proj.weight"):  # of the MTP layer, whose router is next
                proj = dequantize(blocks, gguf.GGMLQuantizationType(GGMLType[kind]))
                units = (proj[:, :dim] @ units.T).T
                units /= np.linalg.norm(units, axis=-1, keepdims=True)
        elif name.endswith(("token_embd.weight", "nextn.embed_tokens.weight")):
            w.write_tensor_data(embeddings.astype(np.float32))
        elif name.endswith("ffn_gate_inp.weight"):
            used = large.metadata["expert_used_count"]
            w.write_tensor_data(_router(shape[0], used, units, rng))
        else:
            w.write_tensor_data(_values(name, shape, rng).astype(np.float32))
    w.close()


def _target(name: str) -> float:
    # the RMS of a matrix's outputs, of inputs of RMS 1
    return next((v for k, v in TARGET.items() if f".{k}." in f".{name}"), 1.0)


def _blocks(
    t: GGMLType, shape: tuple[int, ...], rng: np.random.Generator, target: float
) -> np.ndarray:
    # random blocks of a matrix, or of a stack of them, of outputs of RMS about `target`: their
    # f16 fields, or MXFP4's exponents, scaled from those of a sample at scale 1
    sample = random_blocks(t, 256, np.random.default_rng(1), 1.0)
    rms = math.sqrt(np.mean(dequantize(sample, gguf.GGMLQuantizationType(t)) ** 2))
    scale = target / (math.sqrt(shape[-1]) * rms)
    blocks = random_blocks(t, math.prod(shape) // BLOCK[t][0], rng, scale)
    return blocks.reshape(*shape[:-1], -1)


def _router(experts: int, used: int, units: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # each expert's row: of `used` experts of each group, none of two, along the group's
    # direction their ranks from 1 down to 0.2, and of the others 0; the last expert among the
    # first group's, so that the far end of the experts' stacks is read. A token's best expert
    # scores about ROUTED: the router's input is normed to RMS 1, by weights of 1.25 on average.
    ranks = np.zeros((experts, GROUPS))
    chosen = rng.choice(experts - 1, (GROUPS, used), replace=False)
    chosen[0, 0] = experts - 1
    for g in range(GROUPS):
        ranks[chosen[g], g] = np.linspace(1.0, 0.2, used)
    return (ROUTED / (1.25 * math.sqrt(units.shape[1])) * ranks @ units).astype(np.float32)


def _values(name: str, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    # an F32 tensor's: norms' weights about 1; gpt-oss's sinks, Gated DeltaNet's decays and its
    # convolution as the tiny models'; a router's biases, other vectors and biases small;
    # matrices of outputs of RMS about 1
    if name.endswith("norm.weight"):
        return rng.uniform(1.0, 1.5, shape)
    if name.endswith("attn_sinks.weight"):
        return rng.standard_normal(shape) * 2.0
    if name.endswith("ssm_a"):
        return -rng.uniform(0.2, 2.0, shape)
    if name.endswith("ssm_conv1d.weight"):
        return rng.standard_normal(shape) * 0.5
    if name.endswith(("ffn_gate_inp.bias", "exp_probs_b.bias")):  # less than the routers' margins
        return rng.standard_normal(shape) * 0.01
    if name.endswith("ffn_gate_inp_shexp.weight") or len(shape) == 2 and "bias" not in name:
        return rng.standard_normal(shape) / math.sqrt(shape[-1])
    return rng.standard_normal(shape) * 0.1


def load(path: Path, context: int) -> tuple[Transformer, Mtp | None]:
    """A large model's layers, and its MTP layer's, if it has one, as the engine loads them."""
    f = GGUF.open(path)
    config = Config.from_gguf(f.metadata)
    model = Transformer(config, f.load(names=filter(config.uses, f.tensors)), context)
    arch = f.metadata["general.architecture"]
    return model, Mtp(f, model) if f.metadata.get(f"{arch}.nextn_predict_layers") else None


def run(
    model: Transformer, mtp: Mtp | None, tokens: list[int], before: np.ndarray,
    start: int | UOp = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:  # fmt: skip
    """The logits of tokens at positions start..., and their normed hidden states; and the MTP
    layer's logits of them, if it has one, each with the model's hidden state before it, of
    `before` (dim,) for the first."""
    hidden = model(Tensor([tokens], dtype=dtypes.int32), start)
    states = hidden.numpy()[0]
    if mtp is None:
        return model.logits(hidden).numpy()[0], states, None
    inputs = np.concatenate([before[None], states[:-1]])
    x = mtp._inputs(Tensor([tokens], dtype=dtypes.int32), Tensor(inputs[None]))
    layer = mtp.layer.forward(x, [Span(0, start, len(tokens))])
    return model.logits(hidden).numpy()[0], states, mtp.layer.logits(layer).numpy()[0]


def main() -> None:
    # python -m tests.large_models MODEL OUT CONTEXT TOKEN...: a prompt's logits, and its MTP
    # layer's, if any, of the device DEV names, saved to OUT, as test_large_models has the
    # reference ops' of the CPU, in a process apart.
    # The prompt goes in chunks of fewer pairs of tokens and experts than there are experts, of
    # which the reference ops then read only those taken, rather than every one; and of few enough
    # that the experts taken, which they decode whole to f32, take 2 GiB at most.
    path, out, context, *tokens = sys.argv[1:]
    model, mtp = load(Path(path), int(context))
    config, dim = model.config, model.config.dim
    hidden = next(w.shape[1] for n, w in GGUF.open(Path(path)).tensors.items()
                  if n.endswith("ffn_gate_exps.weight"))  # fmt: skip
    decoded = config.experts_used * 3 * hidden * dim * 4  # bytes a token's experts take
    chunk = max(1, min(config.experts // config.experts_used, (2 << 30) // decoded))
    ids, pos = [int(t) for t in tokens], UOp.variable("start_pos", 0, int(context) - 1)
    parts, layer, before = [], [], np.zeros(dim, np.float32)
    for i in range(0, len(ids), chunk):
        got, states, drafted = run(model, mtp, ids[i : i + chunk], before, pos.bind(i))
        parts.append(got)
        layer += [] if drafted is None else [drafted]
        before = states[-1]
    np.savez(out, logits=np.concatenate(parts), mtp=np.concatenate(layer) if layer else [])


if __name__ == "__main__":
    main()
