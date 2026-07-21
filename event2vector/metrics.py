"""Quantitative metrics for the Event2Vec experiments.

These are the measurement primitives behind the paper's experiments:

- ``additivity_cosine``      -> A1 (linear additive hypothesis / decay curve)
- ``order_sensitivity``      -> A2 (order-sensitivity vs. curvature, the hero metric)
- ``retrieval_precision_at_k`` -> I1 (mechanistic decomposition by subtraction)

All functions are geometry-aware where it matters and return plain Python /
numpy values so they can be logged directly into result tables.
"""
import random
from collections import Counter
from functools import reduce
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .models import HyperbolicUtils

EPS = 1e-9


# --------------------------------------------------------------------------- #
# Distances (row-wise; operate on [N, d] tensors and return [N])
# --------------------------------------------------------------------------- #
def euclidean_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.norm(a - b, dim=-1)


def poincare_distance(c: float) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def _d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return HyperbolicUtils.poincare_dist_sq(a, b, c).clamp_min(0).sqrt().squeeze(-1)

    return _d


def distance_for(geometry: str, curvature: float = 1.0):
    return poincare_distance(curvature) if geometry == "hyperbolic" else euclidean_distance


# --------------------------------------------------------------------------- #
# A1 — linear additive hypothesis
# --------------------------------------------------------------------------- #
def additivity_cosine(estimator, sequences: Sequence[Sequence[int]]) -> np.ndarray:
    """Cosine between the recurrent final state and the direct composition of
    constituent event embeddings.

    Euclidean: composition is the vector sum ``Σ_t e_{s_t}`` and the result is
    a true cosine. Hyperbolic: composition is the left-fold of Möbius addition;
    the cosine is computed in the ambient space (an approximation — see
    ``most_similar`` note in ``models.py``).
    """
    emb = estimator.model.embedding.weight.detach().cpu()
    h = estimator.transform(sequences, as_numpy=False).cpu()
    cos = np.empty(len(sequences), dtype=np.float64)
    c = estimator.curvature
    hyperbolic = estimator.geometry == "hyperbolic"
    for i, s in enumerate(sequences):
        toks = list(int(t) for t in s)
        vecs = emb[toks]
        if hyperbolic:
            ideal = reduce(lambda x, y: HyperbolicUtils.mobius_add(x, y, c), vecs)
        else:
            ideal = vecs.sum(dim=0)
        cos[i] = F.cosine_similarity(h[i : i + 1], ideal[None], dim=-1).item()
    return cos


def additivity_decay(estimator, dataset_record, lengths=(2, 4, 8, 16, 32, 64),
                     n_per_length: int = 256, seed: int = 0) -> dict:
    """Mean additivity cosine as a function of synthetic sequence length.

    Random token sequences of each length are drawn from the vocabulary; this
    isolates the effect of length (and norm clipping) on additivity. Returns
    ``{length: (mean, std)}``.
    """
    rng = random.Random(seed)
    vocab_size = len(dataset_record["vocab"])
    out = {}
    for L in lengths:
        seqs = [[rng.randrange(vocab_size) for _ in range(L)] for _ in range(n_per_length)]
        cos = additivity_cosine(estimator, seqs)
        out[L] = (float(cos.mean()), float(cos.std()))
    return out


# --------------------------------------------------------------------------- #
# A2 — order sensitivity vs. curvature (the hero metric)
# --------------------------------------------------------------------------- #
def _shuffled(seq: List[int], rng: random.Random) -> List[int]:
    s = list(seq)
    rng.shuffle(s)
    return s


def order_sensitivity(
    encode: Callable[[List[List[int]]], torch.Tensor],
    sequences: Sequence[Sequence[int]],
    n_perms: int = 8,
    distance: Optional[Callable] = None,
    seed: int = 0,
    z_sample: int = 256,
) -> dict:
    """Permutation Sensitivity of a sequence encoder.

        PS(s) = mean_m d(phi(s), phi(pi_m . s)) / Z,
        Z      = E[ d(phi(s'), phi(s'')) ]  over independent random sequences,

    so PS is scale-free and comparable across geometries: 0 = order-blind,
    ~1 = permuting destroys as much information as drawing a fresh sequence.

    ``encode`` maps a list of token sequences to an ``[N, d]`` tensor (e.g.
    ``estimator.transform(seqs, as_numpy=False)`` or a baseline encoder).
    ``distance`` is a row-wise distance (default Euclidean L2); pass
    ``poincare_distance(c)`` for hyperbolic encoders.
    """
    rng = random.Random(seed)
    distance = distance or euclidean_distance
    seqs = [list(int(t) for t in s) for s in sequences]

    base = encode(seqs).detach().cpu()
    perm_d = torch.zeros(len(seqs))
    for _ in range(n_perms):
        perm = [_shuffled(s, rng) for s in seqs]
        perm_d += distance(base, encode(perm).detach().cpu())
    perm_d /= n_perms

    # Z: distance between two independent random pairings of encoded sequences.
    idx = list(range(len(seqs)))
    rng.shuffle(idx)
    a_idx = idx[:z_sample]
    b_idx = a_idx[1:] + a_idx[:1]  # cyclic shift => independent pairing
    z = distance(base[a_idx], base[b_idx]).mean().item()

    ps = (perm_d / (z + EPS)).numpy()
    return {
        "mean": float(ps.mean()),
        "std": float(ps.std()),
        "Z": float(z),
        "raw_mean": float(perm_d.mean()),
        "per_seq": ps,
    }


def order_sensitivity_estimator(estimator, sequences, n_perms: int = 8, **kw) -> dict:
    """Convenience wrapper that uses an Event2Vec estimator as the encoder with
    the geometry-appropriate distance."""
    dist = distance_for(estimator.geometry, estimator.curvature)

    def encode(seqs):
        return estimator.transform(seqs, as_numpy=False)

    return order_sensitivity(encode, sequences, n_perms=n_perms, distance=dist, **kw)


# --------------------------------------------------------------------------- #
# I1 — mechanistic decomposition by subtraction
# --------------------------------------------------------------------------- #
def retrieval_precision_at_k(
    estimator, sequences: Sequence[Sequence[int]], k: Optional[int] = None
) -> dict:
    """Greedily peel each sequence embedding by nearest-neighbour + subtraction
    and measure how well the recovered constituents match the true ones.

    Scored as a MULTISET: an additive endpoint encodes token *counts*, so peeling
    [a,a,a,a] out of a sequence that really contains [a,a,a,b] is a genuine failure.
    Comparing bare sets, as this function used to, scores that case 1.0 precision --
    multiplicity errors could not fail. The set-based numbers are still returned under
    ``*_set`` so older results remain comparable.

    Returns mean precision@L and recall@L (L = sequence length). This is the
    interpretability claim: an additive embedding is decomposable; a black-box
    RNN state is not.
    """
    emb = estimator.model.embedding.weight.detach().cpu()
    emb_n = F.normalize(emb, dim=1)
    precisions, recalls = [], []
    precisions_set, recalls_set = [], []
    for s in sequences:
        toks = [int(t) for t in s]
        true_multiset = list(toks)
        L = k or len(toks)
        h = estimator.transform([toks], as_numpy=False)[0].cpu()
        recovered = []
        for _ in range(L):
            sims = emb_n @ F.normalize(h, dim=0)
            j = int(torch.argmax(sims).item())
            recovered.append(j)
            h = h - emb[j]
        # multiset overlap: sum_j min(count_true(j), count_recovered(j))
        ct, cr = Counter(true_multiset), Counter(recovered)
        hit_m = sum(min(ct[j], cr[j]) for j in cr)
        precisions.append(hit_m / max(1, len(recovered)))
        recalls.append(hit_m / max(1, len(true_multiset)))
        true_set, rec_set = set(true_multiset), set(recovered)
        hit_s = len(true_set & rec_set)
        precisions_set.append(hit_s / max(1, len(rec_set)))
        recalls_set.append(hit_s / max(1, len(true_set)))
    return {
        "precision_at_L": float(np.mean(precisions)),
        "recall_at_L": float(np.mean(recalls)),
        "precision_at_L_set": float(np.mean(precisions_set)),
        "recall_at_L_set": float(np.mean(recalls_set)),
    }


# --------------------------------------------------------------------------- #
# E1 — frozen linear probe (embedding utility)
# --------------------------------------------------------------------------- #
def linear_probe(X: np.ndarray, y: np.ndarray, seed: int = 0, test_size: float = 0.3) -> dict:
    """Train a single linear classifier (logistic regression) on frozen vectors
    and report held-out accuracy + macro-F1 and the majority-class floor.

    ``X`` are frozen embeddings ([N, d]); ``y`` are integer labels.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    majority = counts.max() / counts.sum()
    stratify = y if counts.min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    out = {
        "accuracy": float(acc),
        "macro_f1": float(f1_score(yte, pred, average="macro")),
        "majority": float(majority),
        "accuracy_above_majority": float(acc - majority),
        "roc_auc": None,
    }
    if len(classes) == 2:  # ROC-AUC is the MoleculeNet metric (imbalance-robust)
        try:
            proba = clf.predict_proba(Xte)[:, 1]
            out["roc_auc"] = float(roc_auc_score(yte, proba))
        except Exception as e:  # e.g. a fold with a single class present
            out["roc_auc_error"] = f"{type(e).__name__}: {e}"
    return out


def probe_event_labels(estimator, labels: dict, seed: int = 0) -> dict:
    """E1 on *event* embeddings against per-event labels (e.g. life-stage
    domain, POS tag). ``labels`` maps event-id -> label-id."""
    emb = estimator.model.embedding.weight.detach().cpu().numpy()
    ids = sorted(labels)
    X = emb[ids]
    y = np.array([labels[i] for i in ids])
    return linear_probe(X, y, seed=seed)


def probe_sequence_labels(model, record, seed: int = 0, test_size: float = 0.3,
                          label_key: str = "sequence_labels") -> dict:
    """Frozen *sequence-level* probe (the downstream-task protocol): encode each
    sequence with the frozen model, then fit one linear classifier against the
    per-sequence label. Works for any encoder exposing ``transform`` (Event2Vec
    or the baselines), so downstream accuracy can be compared per-parameter.
    Sequences with a missing label are dropped."""
    seqs = record["processed_sequences"]
    labels = record.get(label_key)
    if labels is None:
        raise ValueError(f"record has no '{label_key}'")
    idx = [i for i, l in enumerate(labels) if l is not None]
    X = model.transform([seqs[i] for i in idx], as_numpy=True)
    y = np.array([labels[i] for i in idx])
    return linear_probe(X, y, seed=seed, test_size=test_size)


# --------------------------------------------------------------------------- #
# I2 — counterfactual intervention (Euclidean closed form)
# --------------------------------------------------------------------------- #
def counterfactual_consistency(
    estimator, sequences, n_swaps_per_seq: int = 1, seed: int = 0
) -> dict:
    """Swap one event in each sequence and compare the *actual* change in
    decoded logits to the closed-form additive prediction
    ``Δlogits ≈ W_dec (e_new - e_old)`` (exact for Euclidean when no clip fires).

    Returns mean cosine(Δ, Δ̂) and relative error. High agreement = the model's
    counterfactuals are predictable in closed form (an interpretability win a
    gated RNN cannot match).
    """
    rng = random.Random(seed)
    W = estimator.model.linear_dec.weight.detach().cpu()  # [V, d]
    emb = estimator.model.embedding.weight.detach().cpu()
    V = emb.shape[0]
    cosines, rel_errs = [], []
    for s in sequences:
        toks = [int(t) for t in (s.tolist() if hasattr(s, "tolist") else s)]
        if len(toks) < 2:
            continue
        for _ in range(n_swaps_per_seq):
            t = rng.randrange(len(toks))
            old = toks[t]
            new = rng.randrange(V)
            if new == old:
                new = (new + 1) % V
            s2 = list(toks)
            s2[t] = new
            h = estimator.transform([toks], as_numpy=False)[0].cpu()
            h2 = estimator.transform([s2], as_numpy=False)[0].cpu()
            actual = (W @ h2) - (W @ h)            # [V]
            predicted = W @ (emb[new] - emb[old])  # [V]
            cosines.append(
                F.cosine_similarity(actual[None], predicted[None], dim=-1).item()
            )
            rel_errs.append((actual - predicted).norm().item() / (actual.norm().item() + EPS))
    return {
        "delta_cosine": float(np.mean(cosines)),
        "relative_error": float(np.mean(rel_errs)),
    }


# --------------------------------------------------------------------------- #
# Physics — decay-operator recovery (additivity vs known (dZ, dN) ground truth)
# --------------------------------------------------------------------------- #
def operator_recovery(estimator, record, seed: int = 0, test_size: float = 0.3) -> dict:
    """Test whether the additive endpoint faithfully encodes the conserved
    (Z, N) displacement of a decay chain.

    A chain's true net transmutation is the sum of its decay operators' (dZ, dN).
    We fit a *linear* map from the sequence embedding ``h`` to net (dZ, dN) on
    train chains and report held-out R^2: under exact additivity this is 1.0
    (the embedding linearly carries the conserved quantity); norm clipping or a
    non-additive encoder degrades it. Also returns the per-mode (dZ, dN)
    recovered by applying the fitted map to each event embedding.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    gt = record["ground_truth"]
    seqs = record["processed_sequences"]
    targets = np.array(
        [np.sum([gt[int(t)] for t in s], axis=0) for s in seqs], dtype=np.float64
    )
    H = estimator.transform(seqs, as_numpy=True)
    Htr, Hte, Ytr, Yte = train_test_split(H, targets, test_size=test_size, random_state=seed)
    reg = LinearRegression().fit(Htr, Ytr)
    r2 = r2_score(Yte, reg.predict(Hte), multioutput="uniform_average")

    # Per-mode recovered (dZ,dN) requires an event-embedding table; baselines that
    # expose one (RNN/Transformer .embedding, Event2Vec .model.embedding) work too.
    inner = getattr(estimator, "model", estimator)
    emb_w = getattr(inner, "embedding", None)
    recovered = {}
    if emb_w is not None:
        emb = emb_w.weight.detach().cpu().numpy()
        _V = getattr(estimator, "num_event_types", None)
        if _V is not None and emb.shape[0] > _V:
            emb = emb[:_V]          # drop the baselines' dedicated pad row
        for mode_id, true in gt.items():
            pred = reg.predict(emb[mode_id : mode_id + 1])[0]
            recovered[record["inv_vocab"][mode_id]] = {
                "true": tuple(true),
                "recovered": (round(float(pred[0]), 2), round(float(pred[1]), 2)),
            }
    return {"r2": float(r2), "recovered": recovered}


# --------------------------------------------------------------------------- #
# Math — abelianization gap (Euclidean additive = abelianization of the group)
# --------------------------------------------------------------------------- #
def _multiset_ceiling(records_seqs, labels, n_gen, holdout: bool = False,
                      seed: int = 0, n_splits: int = 5) -> float:
    """Best accuracy of any predictor that sees only the generator multiset
    (= count vector). This is the information ceiling for an additive endpoint,
    which is a function of the multiset (Theorem: order invariance).

    The two estimators bracket the truth and neither is exact:

    * ``holdout=False`` (default) takes the majority label on the *same* rows it
      scores, so any multiset class observed once is counted correct by
      construction. This is an **upper** bound; the optimism is negligible for
      abelian groups (the map is a bijection) but reaches +0.055 on S4.
    * ``holdout=True`` fits the majority on half the sample and scores the other
      half, averaged over ``n_splits`` shuffles. Multiset classes absent from the
      fit half are counted wrong, so this is a **lower** bound.

    For S4 at 5,000 words the bracket is [0.487, 0.542]. Quote the bracket, or the
    held-out value, rather than calling either figure exact.
    """
    from collections import Counter, defaultdict

    key = lambda s: tuple(np.bincount([int(t) for t in s], minlength=n_gen))
    if not holdout:
        by_counts = defaultdict(list)
        for seq, y in zip(records_seqs, labels):
            by_counts[key(seq)].append(y)
        correct = total = 0
        for ys in by_counts.values():
            correct += Counter(ys).most_common(1)[0][1]
            total += len(ys)
        return correct / max(1, total)

    scores = []
    for split in range(n_splits):
        rng = np.random.default_rng(seed + split)
        idx = rng.permutation(len(records_seqs))
        cut = len(idx) // 2
        maj = defaultdict(Counter)
        for i in idx[:cut]:
            maj[key(records_seqs[i])][labels[i]] += 1
        ok = tot = 0
        for i in idx[cut:]:
            k = key(records_seqs[i])
            tot += 1
            if k in maj and maj[k].most_common(1)[0][0] == labels[i]:
                ok += 1
        scores.append(ok / max(1, tot))
    return float(np.mean(scores))


def abelianization_gap(estimator, record, seed: int = 0, probe: str = "mlp") -> dict:
    """Test the claim that the additive Euclidean endpoint recovers exactly the
    abelianization G/[G,G] of the group, not the full element.

    Reports, for the full element and for the abelianization class: (i) a probe
    accuracy from the sequence embedding, and (ii) the *multiset ceiling* --- the
    best accuracy achievable from the generator counts alone, which is precisely
    what an additive endpoint can encode. The signature is: for non-abelian
    groups the abelianization ceiling is ~1 while the full-element ceiling is
    bounded well below 1; for abelian groups both ceilings are ~1.
    """
    H = estimator.transform(record["processed_sequences"], as_numpy=True)
    seqs = record["processed_sequences"]
    n_gen = len(record["vocab"])
    elem = np.array(record["sequence_labels_element"])
    abel = np.array(record["sequence_labels_abelian"])

    def _probe(y):
        if probe == "mlp":
            from sklearn.neural_network import MLPClassifier
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import train_test_split

            Xtr, Xte, ytr, yte = train_test_split(H, y, test_size=0.3, random_state=seed,
                                                  stratify=y if min(np.bincount(y)) >= 2 else None)
            clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=400, random_state=seed)
            clf.fit(Xtr, ytr)
            return float(accuracy_score(yte, clf.predict(Xte)))
        return linear_probe(H, y, seed=seed)["accuracy"]

    return {
        "element_probe": _probe(elem),
        "abelian_probe": _probe(abel),
        "element_ceiling": _multiset_ceiling(seqs, elem, n_gen),
        "abelian_ceiling": _multiset_ceiling(seqs, abel, n_gen),
        "n_elements": record["n_elements"],
        "n_abelian": record["n_abelian"],
        "is_abelian": record["is_abelian"],
    }
