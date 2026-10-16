import numpy as np
import pytest
from tinygrad import Tensor, dtypes

from leatengine.sampler import Sampling, sample

LOGITS = [2.0, 1.0, 0.0, -1.0, 0.5]


def draws(
    logits: list[list[float]], sampling: Sampling | list[Sampling], seed: int | list[int],
    position: int | list[int], seen: list[list[int]] | None = None,
    prompted: list[list[int]] | None = None, bias: list[list[float]] | None = None,
) -> list[int]:  # fmt: skip
    # one value for every row, or one per row; seen, the times each token was generated, and
    # prompted, the tokens of each prompt
    options = Tensor([s.values() for s in (sampling if isinstance(sampling, list) else [sampling])])
    seeds = Tensor(seed if isinstance(seed, list) else [seed], dtype=dtypes.uint32)
    positions = Tensor(position, dtype=dtypes.int32) if isinstance(position, list) else position
    counted = None if seen is None else Tensor(seen, dtype=dtypes.int32)
    words = None
    if prompted is not None:
        words = Tensor([[sum(1 << t for t in row)] for row in prompted], dtype=dtypes.uint32)
    added = None if bias is None else Tensor(bias)
    out = sample(Tensor(logits), options, seeds, positions, counted, prompted=words, bias=added)
    return out.numpy().ravel().tolist()


def frequencies(sampling: Sampling, n: int = 20000, rows: int = 1000) -> np.ndarray:
    # each row a sequence of its own seed, so many rows at a time
    got = [
        t
        for i in range(0, n, rows)
        for t in draws([LOGITS] * rows, sampling, list(range(i, i + rows)), 0)
    ]
    return np.bincount(got, minlength=len(LOGITS)) / n


def softmax(logits: list[float], temperature: float = 1.0) -> np.ndarray:
    p = np.exp(np.array(logits) / temperature)
    return p / p.sum()


def test_greedy():
    assert draws([LOGITS, LOGITS[::-1]], Sampling(), 0, 0) == [0, 4]


def test_temperature_matches_softmax():
    expected = softmax(LOGITS, 0.7)
    np.testing.assert_allclose(frequencies(Sampling(0.7)), expected, atol=0.015)


@pytest.mark.parametrize(
    "sampling, kept",
    [
        (Sampling(1.0, top_k=2), [0, 1]),
        # the likeliest tokens whose probabilities reach top_p of the top 4's: 0, 1, then 4
        (Sampling(1.0, top_k=4, top_p=0.8), [0, 1, 4]),
        (Sampling(1.0, top_p=0.5), [0]),
        # at least a fifth as likely as token 0: e^-1.5 is 0.22, e^-2 0.14
        (Sampling(1.0, min_p=0.2), [0, 1, 4]),
    ],
)
def test_cut_tokens_never_drawn(sampling, kept):
    # the rest renormalized
    expected = np.zeros(len(LOGITS))
    expected[kept] = softmax([LOGITS[i] for i in kept])
    np.testing.assert_allclose(frequencies(sampling), expected, atol=0.015)


def test_top_p_zero_keeps_the_likeliest():
    # as llama.cpp and the kernels: not none, of which every draw would be the first token
    reversed_logits = [LOGITS[::-1]] * 100
    assert set(draws(reversed_logits, Sampling(1.0, top_p=0.0), list(range(100)), 0)) == {4}


def test_presence_penalty():
    # off the logits of tokens seen, greedy too, however many times
    for seen in ([[1, 0, 0, 0, 0]], [[3, 0, 0, 0, 0]]):
        assert draws([LOGITS], Sampling(presence_penalty=1.5), 0, 0, seen) == [1]
        assert draws([LOGITS], Sampling(presence_penalty=0.5), 0, 0, seen) == [0]


def test_frequency_penalty():
    # off the logits of tokens seen, as many times as they were
    assert draws([LOGITS], Sampling(frequency_penalty=0.6), 0, 0, [[1, 0, 0, 0, 0]]) == [0]
    assert draws([LOGITS], Sampling(frequency_penalty=0.6), 0, 0, [[2, 0, 0, 0, 0]]) == [1]
    assert draws([LOGITS], Sampling(frequency_penalty=-2), 0, 0, [[0, 1, 0, 0, 0]]) == [1]


def test_logit_bias():
    # onto the logits before all else, sampled or greedy, the penalties after
    assert draws([LOGITS], Sampling(), 0, 0, bias=[[0, 1.5, 0, 0, 0]]) == [1]
    assert set(draws([LOGITS] * 50, Sampling(1.0), list(range(50)), 0,
                     bias=[[-100, 0, 0, 0, 100]] * 50)) == {4}  # fmt: skip
    penalized = Sampling(repetition_penalty=2.0)  # (2 + 2) / 2 > 1.5, but not 2 / 2
    assert draws([LOGITS], penalized, 0, 0, [[1, 0, 0, 0, 0]], bias=[[2, 0.5, 0, 0, 0]]) == [0]


def test_repetition_penalty():
    # a repeated token's positive logit divided, a negative one multiplied, of tokens generated
    # or of the prompt; first, then the other penalties
    penalized = Sampling(repetition_penalty=2.5)
    assert draws([LOGITS], penalized, 0, 0, [[1, 0, 0, 0, 0]]) == [1]  # 2 / 2.5 < 1
    assert draws([LOGITS], penalized, 0, 0, prompted=[[0]]) == [1]
    assert draws([LOGITS], Sampling(repetition_penalty=1.5), 0, 0, prompted=[[0]]) == [0]
    negative = [[-1.0, -1.2, -5.0, -5.0, -5.0]]
    assert draws(negative, penalized, 0, 0, prompted=[[0]]) == [1]  # -1 * 2.5 < -1.2
    both = Sampling(presence_penalty=0.3, repetition_penalty=1.6)  # 2 / 1.6 - 0.3 < 1
    assert draws([LOGITS], both, 0, 0, [[1, 0, 0, 0, 0]]) == [1]
    # and unpenalized, the logits as they are
    assert draws([[1.0, 1.0 + 2**-20]], Sampling(), 0, 0, [[0, 3]], [[1]]) == [1]


def test_seeded():
    # draws repeat for a seed and position, and change with either
    flat = [[0.0] * 1000]
    first = draws(flat, Sampling(1.0), 7, 3)
    assert draws(flat, Sampling(1.0), 7, 3) == first
    assert draws(flat, Sampling(1.0), 8, 3) != first and draws(flat, Sampling(1.0), 7, 4) != first


def test_rows_are_their_own():
    # a row draws as it would alone, whatever the other rows of its batch: greedy or not, cut or
    # not, and whatever their seeds and positions
    ramp, peaked, flat = [i / 100 for i in range(1000)], LOGITS + [-9.0] * 995, [0.0] * 1000
    cut = Sampling(1.0, top_k=10)
    alone = [draws([ramp], cut, 7, 3)[0], draws([peaked], Sampling(), 0, 9)[0]]
    batched = draws([ramp, peaked, flat], [cut, Sampling(), Sampling(1.0)], [7, 0, 8], [3, 9, 5])
    assert batched[:2] == alone and batched[2] == draws([flat], Sampling(1.0), 8, 5)[0]
    assert alone[0] >= 990
