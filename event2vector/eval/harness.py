"""Seeded experiment harness.

``run_full_suite`` trains each ``(spec, seed)`` model exactly once and computes
every applicable metric from it, returning a JSON-serialisable result dict
(means + std over seeds). ``build_model`` centralises training so every
experiment trains models the same way.
"""
from typing import Dict, List, Sequence

import numpy as np

from ..baselines import build_baseline
from ..metrics import (
    additivity_decay,
    counterfactual_consistency,
    distance_for,
    linear_probe,
    order_sensitivity,
    probe_event_labels,
    retrieval_precision_at_k,
)
from ..models import Event2Vec


def build_model(spec: dict, num_event_types: int, sequences, seed: int):
    """Build + fit a model from a spec dict.

    spec keys: ``kind`` in {event2vec, gru, lstm, transformer, word2vec} plus
    model kwargs (geometry, curvature, max_norm, riemannian, ...).
    """
    kind = spec["kind"]
    common = dict(
        embedding_dim=spec.get("embedding_dim", 32),
        num_epochs=spec.get("num_epochs", 15),
        batch_size=spec.get("batch_size", 64),
        learning_rate=spec.get("learning_rate", 5e-3),
        random_state=seed,
        use_gpu=spec.get("use_gpu", True),
    )
    if kind == "event2vec":
        model = Event2Vec(
            num_event_types=num_event_types,
            geometry=spec.get("geometry", "euclidean"),
            curvature=spec.get("curvature", 1.0),
            max_norm=spec.get("max_norm", 10.0),
            dropout_p=spec.get("dropout_p", 0.1),
            lambda_reconstruction=spec.get("lambda_reconstruction", 0.2),
            lambda_consistency=spec.get("lambda_consistency", 0.2),
            pad_sequences=True,
            riemannian=spec.get("riemannian", False),
            **common,
        )
    else:
        model = build_baseline(kind, num_event_types, **common)
    model.fit(sequences)
    return model


def _agg(values: List[float]) -> dict:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(arr.size)}


def _n_params(model) -> int:
    if hasattr(model, "n_params"):
        return int(model.n_params)
    return int(sum(p.numel() for p in model.model.parameters()))


def _event_embeddings(model):
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "embedding"):  # Event2Vec estimator
        return inner.embedding.weight.detach().cpu().numpy()
    if hasattr(model, "_vectors"):  # word2vec bag
        return model._vectors.numpy()
    if hasattr(model, "embedding"):  # neural baselines (GRU/LSTM/Transformer)
        E = model.embedding.weight.detach().cpu().numpy()
        V = getattr(model, "num_event_types", None)
        # baselines carry one extra row for a dedicated pad id outside the vocabulary
        return E[:V] if V is not None and E.shape[0] > V else E
    return None


def run_full_suite(record, model_specs: List[dict], seeds: Sequence[int],
                   eval_seqs, lengths=(2, 4, 8, 16, 32, 64), n_perms: int = 8) -> Dict:
    """Train each (spec, seed) once; compute A2/A1/E1/I1/I2 where applicable."""
    V = len(record["vocab"])
    seqs = record["processed_sequences"]
    labels = record["labels"]
    label_ids = sorted(labels)
    y = np.array([labels[i] for i in label_ids])
    majority = float(np.max(np.bincount(y)) / y.size)

    out = {"models": {}, "majority_label": majority}
    for spec in model_specs:
        label = spec["label"]
        acc = {"order_ps": [], "probe_acc": [], "probe_f1": [],
               "decomp_prec": [], "decomp_rec": [], "cf_cos": [], "cf_err": [],
               "decay": {L: [] for L in lengths}}
        n_params = None
        is_e2v = spec["kind"] == "event2vec"
        for seed in seeds:
            model = build_model(spec, V, seqs, seed)
            n_params = _n_params(model)

            # A2 — order sensitivity (all models)
            dist = distance_for(getattr(model, "geometry", "euclidean"),
                                getattr(model, "curvature", 1.0))
            ps = order_sensitivity(lambda s: model.transform(s, as_numpy=False),
                                   eval_seqs, n_perms=n_perms, distance=dist, seed=seed)
            acc["order_ps"].append(ps["mean"])

            # E1 — frozen probe on event labels (all models with event embeddings)
            emb = _event_embeddings(model)
            if emb is not None:
                r = linear_probe(emb[label_ids], y, seed=seed)
                acc["probe_acc"].append(r["accuracy"])
                acc["probe_f1"].append(r["macro_f1"])

            if is_e2v:
                # A1 — additivity decay
                decay = additivity_decay(model, record, lengths=lengths, seed=seed)
                for L in lengths:
                    acc["decay"][L].append(decay[L][0])
                # I1 — decomposition
                d = retrieval_precision_at_k(model, eval_seqs)
                acc["decomp_prec"].append(d["precision_at_L"])
                acc["decomp_rec"].append(d["recall_at_L"])
                # I2 — counterfactual closed form
                c = counterfactual_consistency(model, eval_seqs, seed=seed)
                acc["cf_cos"].append(c["delta_cosine"])
                acc["cf_err"].append(c["relative_error"])

        out["models"][label] = {
            "spec": spec,
            "n_params": n_params,
            "order_ps": _agg(acc["order_ps"]),
            "probe_accuracy": _agg(acc["probe_acc"]),
            "probe_macro_f1": _agg(acc["probe_f1"]),
            "probe_majority": majority,
            "decomposition_precision": _agg(acc["decomp_prec"]),
            "decomposition_recall": _agg(acc["decomp_rec"]),
            "counterfactual_cosine": _agg(acc["cf_cos"]),
            "counterfactual_rel_error": _agg(acc["cf_err"]),
            "additivity_decay": {
                "lengths": list(lengths),
                "mean": [float(np.mean(acc["decay"][L])) if acc["decay"][L] else None
                         for L in lengths],
                "std": [float(np.std(acc["decay"][L])) if acc["decay"][L] else None
                        for L in lengths],
            } if is_e2v else None,
        }
    return out
