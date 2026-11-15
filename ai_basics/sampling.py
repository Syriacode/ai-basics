"""Softmax with temperature, and picking a token with a random number."""

import math


def softmax_with_temperature(logits: list[float], temperature: float) -> list[float]:
    """Turn raw scores (logits) into probabilities that add up to 1."""
    if temperature == 0:
        best = logits.index(max(logits))
        return [1.0 if i == best else 0.0 for i in range(len(logits))]
    scaled = [score / temperature for score in logits]
    biggest = max(scaled)
    exps = [math.exp(s - biggest) for s in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def pick_token(probs: list[float], r: float) -> int:
    """Pick an index with a random number r in [0, 1): walk the running total until it passes r."""
    running = 0.0
    for i, p in enumerate(probs):
        running += p
        if r < running:
            return i
    return len(probs) - 1


def prob_from_logprob(logprob: float) -> float:
    """Turn a log probability back into a probability."""
    return math.exp(logprob)
