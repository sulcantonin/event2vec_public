"""Nuclear decay-chain dataset (physics) --- additivity *is* the physics.

Each decay event is an exact fixed displacement in (Z, N) = (protons, neutrons):
alpha = (-2,-2), beta- = (+1,-1), beta+ = (-1,+1), etc. So the Event2Vec
additive hypothesis is *exactly true by construction* here, and the learned
event embeddings should be a linear image of the true (dZ, dN) lattice
(``operator_recovery`` metric). The net transmutation of a chain is the sum of
its decay vectors and is **order-invariant** --- a domain where the Euclidean
bag-of-events endpoint is the physically correct invariant (a positive control).

The vocabulary is the set of decay *modes*; ``GROUND_TRUTH`` gives each mode's
true (dZ, dN). ``CANONICAL_SERIES`` are the real natural decay series, used for
exact decomposition tests (the multiset of decays is known).
"""
import random
from typing import Dict, List, Tuple

import torch

# decay mode -> (dZ, dN); dA = dZ + dN. Each mode is a fixed displacement in the
# proton-neutron plane (a "chart of nuclides" operator). Electron capture shares
# the operator of beta+ (both p -> n), a built-in test of whether two distinct
# modes with the same (dZ,dN) collapse to the same recovered vector. Spontaneous
# fission is deliberately absent: it is a *variable* fragment split, not a fixed
# displacement, so it is non-additive (discussed as a limitation in the paper).
GROUND_TRUTH: Dict[str, Tuple[int, int]] = {
    "alpha": (-2, -2),            # emit He-4
    "beta_minus": (1, -1),        # n -> p
    "beta_plus": (-1, 1),         # p -> n
    "electron_capture": (-1, 1),  # p + e- -> n (same operator as beta+)
    "proton_emission": (-1, 0),
    "neutron_emission": (0, -1),
    "double_beta": (2, -2),       # 2n -> 2p
    "cluster_c14": (-6, -8),      # emit a 14C nucleus (Z=6, N=8)
    "gamma": (0, 0),              # isomeric transition: no transmutation
}
MODES: List[str] = list(GROUND_TRUTH)

# Real natural decay series as ordered mode sequences (main decay path).
CANONICAL_SERIES: Dict[str, dict] = {
    "thorium_4n": {  # Th-232 (90,142) -> Pb-208 (82,126): 6 alpha + 4 beta-
        "parent": (90, 142), "daughter": (82, 126),
        "modes": ["alpha", "beta_minus", "beta_minus", "alpha", "alpha", "alpha",
                  "alpha", "beta_minus", "beta_minus", "alpha"],
    },
    "uranium_4n+2": {  # U-238 (92,146) -> Pb-206 (82,124): 8 alpha + 6 beta-
        "parent": (92, 146), "daughter": (82, 124),
        "modes": ["alpha", "beta_minus", "beta_minus", "alpha", "alpha", "alpha",
                  "alpha", "alpha", "beta_minus", "beta_minus", "alpha",
                  "beta_minus", "beta_minus", "alpha"],
    },
}


def _n_stable(z: int) -> float:
    """Rough line of beta stability N(Z) (semi-empirical)."""
    return z + 0.0060 * z * z


def _generate_chain(rng: random.Random, max_length: int) -> List[str]:
    """A physically-motivated walk toward stability that exercises every mode:
    superheavy nuclei alpha/cluster-decay, neutron-rich emit beta-/n/double-beta,
    proton-rich emit beta+/EC/p."""
    z = rng.randint(50, 100)
    offset = rng.choice([rng.uniform(-14, -6), rng.uniform(-3, 3),
                         rng.uniform(6, 18), rng.uniform(0, 8)])
    n = max(1, int(_n_stable(z) + offset))
    modes = []
    for _ in range(max_length):
        a = z + n
        ex = n - _n_stable(z)
        w = {}
        if a > 200:
            w["alpha"] = 1.0 + 0.02 * (a - 200)
        if a > 215 and z > 86:
            w["cluster_c14"] = 0.5
        if ex > 0.5:
            w["beta_minus"] = 0.5 + 0.3 * ex
        if ex > 6:
            w["neutron_emission"] = 0.5 * (ex - 6)
        if ex > 1 and z % 2 == 0 and n % 2 == 0:
            w["double_beta"] = 0.4
        if ex < -0.5:
            w["beta_plus"] = 0.4 * (0.5 - ex)
            w["electron_capture"] = 0.4 * (0.5 - ex)
        if ex < -6:
            w["proton_emission"] = 0.5 * (-ex - 6)
        w["gamma"] = 0.05
        if not w or sum(w.values()) < 1e-6:
            break
        modes_, ws_ = zip(*w.items())
        mode = rng.choices(modes_, weights=ws_, k=1)[0]
        dz, dn = GROUND_TRUTH[mode]
        z, n = z + dz, n + dn
        modes.append(mode)
        if z < 1 or n < 1:
            break
        if abs(ex) < 1 and a < 210:
            break
    return modes or ["alpha"]


def make_decay(num_seq: int = 4000, max_length: int = 16, seed: int = 0) -> dict:
    """Generate decay-mode sequences as the uniform dataset record.

    Returns the usual keys plus ``ground_truth`` (mode-id -> (dZ,dN)) and
    ``canonical`` (named real series as token sequences) for exact tests.

    Note the dataset holds ``num_seq + len(CANONICAL_SERIES)`` sequences, not
    ``num_seq``: the named real series are appended so they are in-distribution
    for anything trained on this data.
    """
    rng = random.Random(seed)
    raw = []
    for _ in range(num_seq):
        chain = _generate_chain(rng, max_length)
        # inject occasional gamma (no transmutation) to test the ~zero vector
        if rng.random() < 0.3 and len(chain) >= 2:
            chain.insert(rng.randrange(len(chain)), "gamma")
        raw.append(chain)
    # always include the canonical real series so they are in-distribution
    for s in CANONICAL_SERIES.values():
        raw.append(list(s["modes"]))

    # vocabulary = decay modes that actually occur (ordered by GROUND_TRUTH).
    present = {m for c in raw for m in c}
    vocab = {m: i for i, m in enumerate(m for m in MODES if m in present)}
    processed = [torch.tensor([vocab[m] for m in c], dtype=torch.long) for c in raw]
    ground_truth = {vocab[m]: GROUND_TRUTH[m] for m in vocab}
    canonical = {
        name: {"tokens": [vocab[m] for m in s["modes"]],
               "net_dZdN": (s["daughter"][0] - s["parent"][0],
                            s["daughter"][1] - s["parent"][1]),
               "modes": s["modes"]}
        for name, s in CANONICAL_SERIES.items()
    }
    return {
        "sequences": raw,
        "processed_sequences": processed,
        "vocab": vocab,
        "inv_vocab": {i: m for m, i in vocab.items()},
        "labels": None,
        "ground_truth": ground_truth,
        "canonical": canonical,
        "graph": None,
    }
