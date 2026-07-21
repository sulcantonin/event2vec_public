"""Arithmetic sequences --- the minimal interpretability example.

This is the simplest possible setting in which additivity *is* the task. Each
event is a fixed signed integer increment on the number line (``+1``, ``+5``,
``-2``, ...); a sequence of increments is a walk whose endpoint is the running
total. Because integer addition is exactly additive (and commutative), the
Event2Vec additive hypothesis is *exactly true by construction* here, giving the
cleanest possible ground truth:

* a single 1-D linear readout from the sequence embedding recovers the total
  (held-out ``R^2 = 1.00``), and
* the *same* readout applied to each event embedding returns that event's integer
  value exactly.

It is the number-line analogue of the nuclear-decay test (``decay.py``), stripped
of all physics down to bare arithmetic, so the recovered "operators" are just the
integers themselves. As in the abelian-group case, the additive endpoint is
order-invariant, which is exactly correct here: ``a + b = b + a``.
"""
import random
from typing import Dict, List, Tuple

import torch

# event name -> integer value (a fixed displacement on the number line). ``+0`` is
# an identity element (the arithmetic analogue of the gamma transition in decay):
# it must be recovered as exactly zero.
GROUND_TRUTH: Dict[str, int] = {
    "+0": 0,
    "+1": 1,
    "+2": 2,
    "+3": 3,
    "+5": 5,
    "+10": 10,
    "-1": -1,
    "-2": -2,
    "-5": -5,
}
TOKENS: List[str] = list(GROUND_TRUTH)


def make_arithmetic(num_seq: int = 4000, min_len: int = 2, max_len: int = 16,
                    seed: int = 0) -> dict:
    """Generate increment-token sequences as the uniform dataset record.

    Returns the usual keys plus ``ground_truth`` (token-id -> ``(value,)``), so
    the physics ``operator_recovery`` machinery applies unchanged with a 1-D
    target.
    """
    rng = random.Random(seed)
    names = TOKENS
    raw: List[List[str]] = []
    for _ in range(num_seq):
        L = rng.randint(min_len, max_len)
        raw.append([rng.choice(names) for _ in range(L)])

    vocab = {t: i for i, t in enumerate(names)}
    processed = [torch.tensor([vocab[t] for t in c], dtype=torch.long) for c in raw]
    ground_truth: Dict[int, Tuple[int]] = {vocab[t]: (GROUND_TRUTH[t],) for t in vocab}
    return {
        "sequences": raw,
        "processed_sequences": processed,
        "vocab": vocab,
        "inv_vocab": {i: t for t, i in vocab.items()},
        "labels": None,
        "ground_truth": ground_truth,
        "graph": None,
    }
