"""Hand-written kernels, in tinygrad's UOp DSL, rendered as CUDA C or HIP C.

The warp-level kernels, which carry decoding, run on NVIDIA GPUs (DEV=NV or CUDA) and on AMD's RDNA
GPUs (DEV=AMD), as Strix Halo's; those on tensor cores, matmul's and FlashAttention's, on NVIDIA's
mma.sync and RDNA 3's WMMA. Each family the model's ops dispatch to has a `supports_*` check beside
the op itself: leatengine.ops chooses between them and the reference ops.
"""

from leatengine.kernels.argmax import argmax, supports_argmax
from leatengine.kernels.attention import (
    attention,
    flash_attention,
    rotate,
    supports_attention,
    supports_flash_attention,
    supports_rotate,
)
from leatengine.kernels.common import GROUP, on_matrix_cores
from leatengine.kernels.cutoff import cutoff, supports_cutoff
from leatengine.kernels.delta import delta_net, supports_delta_net
from leatengine.kernels.experts import mixture, route, scores, supports_mixture, supports_scores
from leatengine.kernels.matmul import (
    FEW_TOKENS,
    STEP,
    STEP_GROUPS,
    feed_forward,
    matmuls,
    supports_matmul,
)
from leatengine.kernels.matvec import MATVEC_TOKENS, matvecs, supports_matvec, swiglu
from leatengine.kernels.norms import add_normed, supports_add_normed
from leatengine.kernels.quantize import quantize_q8

# the bound variables kernels bind inside a graph, by name: TinyJit replays a graph with the
# values its arguments carry, and these ride along there
VARIABLES = {"groups": STEP_GROUPS.bind(STEP // GROUP)}

__all__ = [
    "FEW_TOKENS", "GROUP", "MATVEC_TOKENS", "VARIABLES", "add_normed", "argmax", "attention",
    "cutoff", "delta_net", "feed_forward", "flash_attention", "matmuls", "matvecs", "mixture",
    "on_matrix_cores", "quantize_q8", "rotate", "route", "scores", "supports_add_normed",
    "supports_argmax", "supports_attention", "supports_cutoff", "supports_delta_net",
    "supports_flash_attention", "supports_matmul", "supports_matvec", "supports_mixture",
    "supports_rotate", "supports_scores", "swiglu",
]  # fmt: skip
