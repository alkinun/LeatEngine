import numpy as np
import pytest
from gguf import GGMLQuantizationType
from gguf.quants import dequantize, quantize
from tinygrad import Tensor

from leatengine.quant import BLOCK, GGMLType, QTensor, from_q8_0, q8_0_row, to_q8_0
from tests.helpers import F16_FIELDS, random_blocks


@pytest.mark.parametrize("ggml_type", list(F16_FIELDS))
def test_dequant_matches_ggml(ggml_type):
    rng = np.random.default_rng(int(ggml_type))
    blocks = random_blocks(ggml_type, 64, rng)
    shape = (8, 64 * BLOCK[ggml_type][0] // 8)
    out = QTensor(Tensor(blocks), ggml_type, shape).dequant().numpy()
    expected = dequantize(blocks, GGMLQuantizationType(ggml_type)).reshape(shape)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("ggml_type", [GGMLType.F32, GGMLType.F16, GGMLType.BF16])
def test_dequant_native(ggml_type):
    values = np.random.default_rng(0).standard_normal((4, 32)).astype(np.float32)
    data = Tensor(values).cast(
        {GGMLType.F32: "float32", GGMLType.F16: "half", GGMLType.BF16: "bfloat16"}[ggml_type]
    )
    out = QTensor(data.flatten(), ggml_type, (4, 32)).dequant().numpy()
    np.testing.assert_array_equal(out, data.float().numpy())


def test_dequant_unsupported():
    with pytest.raises(NotImplementedError, match="IQ1_S"):
        QTensor(Tensor.zeros(1, 50), GGMLType.IQ1_S, (256,)).dequant()


@pytest.mark.parametrize("dim", [64, 96, 576])
def test_q8_0_rows_match_ggml(dim):
    # a KV cache's q8_0 rows: the values and scales gguf-py quantizes q8_0 to, of a block of zeros
    # too, padded to whole words; and from them the values gguf-py dequantizes
    x = np.random.default_rng(dim).standard_normal((2, 3, dim)).astype(np.float32)
    x[0, 0, :32] = 0
    rows = to_q8_0(Tensor(x)).numpy()
    blocks = quantize(x.reshape(6, dim), GGMLQuantizationType.Q8_0)
    assert rows.shape == (2, 3, q8_0_row(dim)) and q8_0_row(dim) % 4 == 0
    values, scales = (rows[..., :dim], rows[..., dim : dim + dim // 16])
    per_block = blocks.reshape(2, 3, dim // 32, 34)
    np.testing.assert_array_equal(values, per_block[..., 2:].reshape(2, 3, dim))
    np.testing.assert_array_equal(scales, per_block[..., :2].reshape(2, 3, -1))
    assert not rows[..., dim + dim // 16 :].any()
    expected = dequantize(blocks, GGMLQuantizationType.Q8_0).reshape(x.shape)
    np.testing.assert_array_equal(from_q8_0(Tensor(rows), dim).numpy(), expected)
