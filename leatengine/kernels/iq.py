"""The IQ2 and IQ3 formats' sub-blocks of 32 weights, as signed bytes for __dp4a and the tensor
cores: each weight is d * (2s + 1) * factor * value, for the block's d, a scale s per sub-block or
per 16 weights, and a value of the format's grid, signed.

`raw` loads a sub-block's words from a block's halfwords, by a reader of halfword i as a word, and
`decode` makes them its 8 words of 4 values, against the grids common.iq_grids holds, and its
scales: apart, the tensor-core kernels fetch a step's words before the products and decode them
after.
"""

from collections.abc import Callable
from dataclasses import dataclass

from tinygrad import UOp, dtypes

from leatengine.kernels.common import GRID_AT, byte_signs, ksigns, load_words
from leatengine.quant import GGMLType

Halfword = Callable[[UOp], UOp]


def _word(half: Halfword, i: UOp) -> UOp:
    # 4 bytes from halfword i on
    return half(i) | (half(i + 1) << 16)


def _byte(word: UOp, i: UOp | int) -> UOp:
    # byte i of a word, for a byte index that may vary
    shift = (8 * i).cast(dtypes.uint32) if isinstance(i, UOp) else 8 * i
    return (word >> shift) & 0xFF


def _iq2_entry(grid: UOp, kind: GGMLType, entry: UOp, signs: UOp) -> list[UOp]:
    # the 8 values of an IQ2 grid entry, signed by the 8 bits of `signs`, as 2 words
    low, high = load_words(grid, GRID_AT[kind] + 2 * entry.cast(dtypes.int32), 2)
    return [byte_signs(low, signs), byte_signs(high, signs >> 4)]


def _iq3_entry(grid: UOp, kind: GGMLType, entry: UOp, signs: UOp) -> UOp:
    # the 4 values of an IQ3 grid entry, signed by the low 4 bits of `signs`, as a word
    return byte_signs(grid[GRID_AT[kind] + entry.cast(dtypes.int32)].load(), signs)


def _iq2_xxs_raw(half: Halfword, base: UOp, sub: UOp) -> list[UOp]:
    # IQ2_XXS block, 33 halfwords: d as f16, then per sub-block 4 bytes of indices into the grid,
    # each of 8 values, and a word of 4 runs of 7 sign bits and a scale, its top 4 bits
    return [_word(half, base + 1 + 4 * sub), _word(half, base + 3 + 4 * sub)]


def _iq2_xxs(grid: UOp, raw: list[UOp], sub: UOp) -> tuple[list[UOp], list[UOp]]:
    indices, aux = raw
    values = []
    for k in range(4):
        signs = ksigns((aux >> (7 * k)) & 127)
        values += _iq2_entry(grid, GGMLType.IQ2_XXS, _byte(indices, k), signs)
    return values, [aux >> 28]


def _iq2_xs_raw(half: Halfword, base: UOp, sub: UOp) -> list[UOp]:
    # IQ2_XS block, 37 halfwords: d as f16, a halfword per 8 values of a 9-bit index into the
    # grid and 7 sign bits, then a byte per sub-block of a scale per 16 values, the low first
    words = [_word(half, base + 1 + 4 * sub + 2 * i) for i in range(2)]
    return [*words, half(base + 33 + sub // 2)]


def _iq2_xs(grid: UOp, raw: list[UOp], sub: UOp) -> tuple[list[UOp], list[UOp]]:
    values = []
    for j in range(4):
        q = (raw[j // 2] >> (16 * (j % 2))) & 0xFFFF
        values += _iq2_entry(grid, GGMLType.IQ2_XS, q & 511, ksigns(q >> 9))
    scales = _byte(raw[2], sub % 2)
    return values, [scales & 15, scales >> 4]


def _iq2_s_raw(half: Halfword, base: UOp, sub: UOp) -> list[UOp]:
    # IQ2_S block, 41 halfwords: d as f16, 32 bytes of indices' low bits and 32 of signs, each
    # for 8 values, a byte per sub-block of its indices' high bits, 2 each, then IQ2_XS's scales
    indices, signs = _word(half, base + 1 + 2 * sub), _word(half, base + 17 + 2 * sub)
    return [indices, signs, half(base + 33 + sub // 2), half(base + 37 + sub // 2)]


def _iq2_s(grid: UOp, raw: list[UOp], sub: UOp) -> tuple[list[UOp], list[UOp]]:
    indices, signs, high, scales = raw[0], raw[1], _byte(raw[2], sub % 2), _byte(raw[3], sub % 2)
    values = []
    for j in range(4):
        entry = _byte(indices, j) | (((high >> (2 * j)) & 3) << 8)
        values += _iq2_entry(grid, GGMLType.IQ2_S, entry, _byte(signs, j))
    return values, [scales & 15, scales >> 4]


def _iq3_xxs_raw(half: Halfword, base: UOp, sub: UOp) -> list[UOp]:
    # IQ3_XXS block, 49 halfwords: d as f16, 64 bytes of indices into the grid, each of 4
    # values, then a word per sub-block of IQ2_XXS's signs and scale
    indices = [_word(half, base + 1 + 4 * sub + 2 * i) for i in range(2)]
    return [*indices, _word(half, base + 33 + 2 * sub)]


def _iq3_xxs(grid: UOp, raw: list[UOp], sub: UOp) -> tuple[list[UOp], list[UOp]]:
    aux, values = raw[2], []
    for e in range(8):
        signs = ksigns((aux >> (7 * (e // 2))) & 127) >> (4 * (e % 2))
        values.append(_iq3_entry(grid, GGMLType.IQ3_XXS, _byte(raw[e // 4], e % 4), signs))
    return values, [aux >> 28]


def _iq3_s_raw(half: Halfword, base: UOp, sub: UOp) -> list[UOp]:
    # IQ3_S block, 55 halfwords: d as f16, 64 bytes of indices' low bits, each index of 4 values,
    # a byte per sub-block of their high bits, a byte of signs per 8 values, then a scale per
    # sub-block of 4 bits, the low first
    indices = [_word(half, base + 1 + 4 * sub + 2 * i) for i in range(2)]
    signs = _word(half, base + 37 + 2 * sub)
    return [*indices, half(base + 33 + sub // 2), signs, half(base + 53 + sub // 4)]


def _iq3_s(grid: UOp, raw: list[UOp], sub: UOp) -> tuple[list[UOp], list[UOp]]:
    high, signs, values = _byte(raw[2], sub % 2), raw[3], []
    for e in range(8):
        entry = _byte(raw[e // 4], e % 4) | (((high >> e) & 1) << 8)
        sign = _byte(signs, e // 2) >> (4 * (e % 2))
        values.append(_iq3_entry(grid, GGMLType.IQ3_S, entry, sign))
    scale = _byte(raw[4], sub // 2 % 2) >> (4 * (sub % 2)).cast(dtypes.uint32)
    return values, [scale & 15]


@dataclass(frozen=True)
class Format:
    """An IQ format's blocks of 256 weights: halfwords each, its sub-blocks' raw loads and their
    decoding, and the factor of its weights."""

    halfwords: int
    raw: Callable[[Halfword, UOp, UOp], list[UOp]]
    decode: Callable[[UOp, list[UOp], UOp], tuple[list[UOp], list[UOp]]]
    factor: float


IQ_FORMATS = {
    GGMLType.IQ2_XXS: Format(33, _iq2_xxs_raw, _iq2_xxs, 1 / 8),
    GGMLType.IQ2_XS: Format(37, _iq2_xs_raw, _iq2_xs, 1 / 8),
    GGMLType.IQ2_S: Format(41, _iq2_s_raw, _iq2_s, 1 / 8),
    GGMLType.IQ3_XXS: Format(49, _iq3_xxs_raw, _iq3_xxs, 1 / 4),
    GGMLType.IQ3_S: Format(55, _iq3_s_raw, _iq3_s, 1.0),
}
