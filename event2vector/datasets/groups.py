"""Group-element sequences (math) --- the definitive commutativity-vs-order test.

Generators are the event vocabulary; a sequence of generators is a walk on the
group's Cayley graph whose product is a group element. The additive Event2Vec
endpoint depends only on the *multiset* of generators (Theorem: order invariance),
so it can represent a group element exactly iff the element is determined by
generator counts --- i.e. iff the group is abelian. For a non-abelian group the
Euclidean model can recover at most the **abelianization** G/[G,G]; the order /
curvature dial is what recovers the rest.

This module builds groups in their permutation representation and labels each
sequence with both its full element id and its abelianization class, so a probe
can measure the exact "Euclidean = abelianization" gap.
"""
import random
from typing import Dict, List, Tuple

import torch

Perm = Tuple[int, ...]


def _identity(n: int) -> Perm:
    return tuple(range(n))


def _compose(a: Perm, b: Perm) -> Perm:
    # (a o b)[i] = a[b[i]]  (apply b, then a)
    return tuple(a[b[i]] for i in range(len(a)))


def _transposition(n: int, i: int) -> Perm:
    p = list(range(n))
    p[i], p[i + 1] = p[i + 1], p[i]
    return tuple(p)


def _rotation(n: int, k: int) -> Perm:
    return tuple((i + k) % n for i in range(n))


def _sign(p: Perm) -> int:
    inv = 0
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] > p[j]:
                inv += 1
    return inv % 2


def _enumerate(generators: List[Perm], n: int) -> Dict[Perm, int]:
    """BFS closure of the generators -> {element: id}."""
    ident = _identity(n)
    seen = {ident: 0}
    frontier = [ident]
    while frontier:
        nxt = []
        for el in frontier:
            for g in generators:
                e2 = _compose(g, el)
                if e2 not in seen:
                    seen[e2] = len(seen)
                    nxt.append(e2)
        frontier = nxt
    return seen


def _build_group(name: str):
    """Return (n_points, generators, gen_names, is_abelian, abelian_fn)."""
    name = name.lower()
    if name in ("z12", "cyclic"):
        n = 12
        gens = [_rotation(n, 1), _rotation(n, 5)]  # both rotations => abelian
        return n, gens, ["+1", "+5"], True, None
    if name == "z6xz4":
        # Z6 x Z4 on 10 points (independent cycles): abelian, 2 generators.
        def rot_block(n, lo, hi, k):
            p = list(range(n))
            size = hi - lo
            for i in range(lo, hi):
                p[i] = lo + ((i - lo + k) % size)
            return tuple(p)
        n = 10
        gens = [rot_block(n, 0, 6, 1), rot_block(n, 6, 10, 1)]
        return n, gens, ["a", "b"], True, None
    if name == "s3":
        n = 3
        gens = [_transposition(n, 0), _transposition(n, 1)]
        return n, gens, ["(01)", "(12)"], False, _sign
    if name == "s4":
        n = 4
        gens = [_transposition(n, 0), _transposition(n, 1), _transposition(n, 2)]
        return n, gens, ["(01)", "(12)", "(23)"], False, _sign
    raise ValueError(f"unknown group {name}")


def make_group(name: str = "s4", num_seq: int = 4000, min_len: int = 2,
               max_len: int = 12, seed: int = 0) -> dict:
    """Generate generator-token sequences as the uniform dataset record.

    Extra keys: ``sequence_labels_element`` (full element id per sequence),
    ``sequence_labels_abelian`` (abelianization class per sequence),
    ``is_abelian``, ``n_elements``, ``n_abelian``.
    """
    rng = random.Random(seed)
    n, gens, gen_names, is_abelian, abelian_fn = _build_group(name)
    elem_id = _enumerate(gens, n)
    k = len(gens)
    vocab = {g: i for i, g in enumerate(gen_names)}

    raw, processed, elem_labels, abel_labels = [], [], [], []
    abelian_classes: Dict = {}
    for _ in range(num_seq):
        L = rng.randint(min_len, max_len)
        toks = [rng.randrange(k) for _ in range(L)]
        el = _identity(n)
        for t in toks:
            el = _compose(gens[t], el)
        raw.append([gen_names[t] for t in toks])
        processed.append(torch.tensor(toks, dtype=torch.long))
        elem_labels.append(elem_id[el])
        if is_abelian:
            cls = elem_id[el]
        else:
            cls = abelian_fn(el)
        abel_labels.append(cls)
        abelian_classes.setdefault(cls, len(abelian_classes))

    # remap abelian classes to contiguous ids
    abel_remap = {c: i for i, c in enumerate(sorted(set(abel_labels)))}
    abel_labels = [abel_remap[c] for c in abel_labels]

    return {
        "sequences": raw,
        "processed_sequences": processed,
        "vocab": vocab,
        "inv_vocab": {i: g for g, i in vocab.items()},
        "labels": None,
        "sequence_labels_element": elem_labels,
        "sequence_labels_abelian": abel_labels,
        "is_abelian": is_abelian,
        "n_elements": len(elem_id),
        "n_abelian": len(set(abel_labels)),
        "group_name": name,
        "graph": None,
    }
