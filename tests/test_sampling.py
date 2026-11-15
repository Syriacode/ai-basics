import math

import pytest

from ai_basics.sampling import pick_token, prob_from_logprob, softmax_with_temperature

# The guide's paper example (section 2c).
PAPER_LOGITS = [2.0, 1.0, 0.0]


@pytest.mark.parametrize(
    ("temperature", "expected"),
    [
        (1.0, [0.665, 0.245, 0.090]),
        (0.5, [0.867, 0.117, 0.016]),
        (2.0, [0.506, 0.307, 0.186]),
    ],
)
def test_softmax_matches_the_paper_numbers(temperature, expected):
    probs = softmax_with_temperature(PAPER_LOGITS, temperature)

    assert probs == pytest.approx(expected, abs=0.001)


def test_temperature_zero_is_greedy():
    assert softmax_with_temperature(PAPER_LOGITS, 0.0) == [1.0, 0.0, 0.0]


def test_big_logits_do_not_overflow():
    # math.exp(1000 / 0.1) overflows unless the biggest score is subtracted first.
    probs = softmax_with_temperature([1000.0, 999.0], 0.1)

    assert sum(probs) == pytest.approx(1.0)


@pytest.mark.parametrize(("r", "expected"), [(0.0, 0), (0.5, 0), (0.7, 1), (0.95, 2)])
def test_pick_token_walks_the_running_total(r, expected):
    assert pick_token([0.665, 0.245, 0.090], r) == expected


def test_pick_token_falls_back_to_the_last_index_when_rounding_leaves_a_gap():
    # The probabilities add up to 0.9999999, and r is bigger than that.
    assert pick_token([0.5, 0.4999999], 0.99999999) == 1


def test_prob_from_logprob_undoes_the_log():
    assert prob_from_logprob(math.log(0.665)) == pytest.approx(0.665)
    assert prob_from_logprob(0.0) == 1.0
