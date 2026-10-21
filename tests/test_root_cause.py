"""Tests for the root-cause analysis API (``attribute``, ``counterfactual``,
``influence_matrix``, ``event_impact``, ``decompose``, ``explain_difference``)
and ``metrics.transition_graph_auc``. Run with `pytest tests/ -q`.
"""
import numpy as np
import pytest
import torch

from event2vector import Event2Vec
from event2vector.metrics import transition_graph_auc

V = 8
SEQS = [[0, 1, 2, 3], [0, 1, 4, 5], [6, 2, 3, 7], [0, 6, 1, 2]] * 4


def _fit(**kw):
    kw = dict(dict(num_event_types=V, embedding_dim=6, dropout_p=0.0, num_epochs=3,
                   batch_size=4, use_gpu=False, random_state=0), **kw)
    return Event2Vec(**kw).fit(SEQS)


@pytest.mark.parametrize("max_norm", [None, 10.0, 0.01])
def test_euclidean_attribution_is_exact(max_norm):
    est = _fit(max_norm=max_norm)
    seq = [0, 6, 1, 2, 1]
    out = est.attribute(seq, target=3)
    assert out["exact"] and len(out["contributions"]) == len(seq)
    assert out["bias"] + out["contributions"].sum() == pytest.approx(out["logit"], abs=1e-4)
    if max_norm is None:
        assert np.allclose(out["weights"], 1.0)
    if max_norm == 0.01:  # clipping fires: earlier events are down-weighted
        assert np.all(np.diff(out["weights"]) >= -1e-6) and out["weights"][0] < 1.0


def test_attribution_defaults_to_predicted_event():
    est = _fit()
    out = est.attribute([0, 1])
    logits = est._next_logits([0, 1])
    assert out["target"] == int(logits.argmax())


@pytest.mark.parametrize("geometry,kw", [("hyperbolic", {}), ("rotation", {"rotation_n": 3})])
def test_attribution_falls_back_to_leave_one_out(geometry, kw):
    est = _fit(geometry=geometry, **kw)
    seq = [0, 1, 2]
    out = est.attribute(seq, target=3)
    assert not out["exact"] and out["weights"] is None
    expected = out["logit"] - float(est._next_logits([0, 2])[3])
    assert out["contributions"][1] == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("geometry", ["euclidean", "hyperbolic"])
def test_counterfactual_matches_reencoding(geometry):
    est = _fit(geometry=geometry)
    out = est.counterfactual([0, 1, 2, 3], remove=[1, 3])
    q = torch.softmax(est._next_logits([0, 2]), 0).numpy()
    assert np.allclose(out["probs_counterfactual"], q, atol=1e-6)
    assert np.allclose(out["delta"], out["probs_counterfactual"] - out["probs"])
    assert out["probs"].sum() == pytest.approx(1.0, abs=1e-5)


def test_influence_matrix_is_decoder_times_embeddings_without_clipping():
    est = _fit(max_norm=None)
    A = est.influence_matrix()
    W = est.model.linear_dec.weight.detach().numpy()
    E = est.model.embedding.weight.detach().numpy()
    assert A.shape == (V, V) and np.allclose(A, W @ E.T, atol=1e-5)
    # every sequence's logits are bias + A @ counts
    counts = np.bincount([0, 1, 1, 5], minlength=V)
    b = est.model.linear_dec.bias.detach().numpy()
    assert np.allclose(est._next_logits([0, 1, 1, 5]).numpy(), b + A @ counts, atol=1e-4)


def test_event_impact_is_centered_column_norm():
    est = _fit(geometry="hyperbolic")
    A = est.influence_matrix()
    impact = est.event_impact()
    assert impact.shape == (V,)
    assert impact[2] == pytest.approx(np.linalg.norm(A[:, 2] - A[:, 2].mean()), rel=1e-5)


def _orthogonal_estimator():
    est = _fit(max_norm=None, embedding_dim=V)
    with torch.no_grad():  # orthonormal embeddings: every history is identifiable
        est.model.embedding.weight.copy_(torch.eye(V) * 2.0)
    return est


def test_decompose_recovers_counts_from_state():
    est = _orthogonal_estimator()
    seq = [3, 1, 3, 7, 3]
    expected = np.bincount(seq, minlength=V)
    assert np.array_equal(est.decompose(seq, round_counts=True), expected)
    state = est.transform([seq], as_numpy=False)[0]
    assert np.allclose(est.decompose(state), expected, atol=1e-6)


def test_explain_difference_returns_signed_counts():
    est = _orthogonal_estimator()
    diff = est.explain_difference([0, 2, 2, 5], [0, 2, 6], round_counts=True)
    expected = np.zeros(V)
    expected[[2, 5, 6]] = [1, 1, -1]
    assert np.array_equal(diff, expected)


def test_sum_based_methods_reject_other_geometries():
    est = _fit(geometry="hyperbolic")
    with pytest.raises(NotImplementedError):
        est.decompose([0, 1])
    with pytest.raises(NotImplementedError):
        est.explain_difference([0, 1], [1])


def test_transition_graph_auc():
    est = _fit()
    vocab = {f"e{i}": i for i in range(V)}
    graph = {"e0": [("e1", 0.5), ("e6", 0.5)], "e1": [("e2", 1.0)], "e2": [("e3", 1.0)]}
    A = np.zeros((V, V))
    A[1, 0] = A[6, 0] = A[2, 1] = A[3, 2] = 1.0  # a perfect score on the true edges
    out = transition_graph_auc(est, graph, vocab, influence=A)
    assert out["auc"] == 1.0 and out["n_edges"] == 4 and out["n_pairs"] == V * (V - 1)
    assert 0.0 <= transition_graph_auc(est, graph, vocab)["auc"] <= 1.0


def test_root_cause_methods_require_fit():
    est = Event2Vec(num_event_types=V, use_gpu=False)
    with pytest.raises(RuntimeError):
        est.attribute([0, 1])
