"""Tests for the root-cause methods and the Euclidean diagonal gate.

``attribute`` decomposes the next-event logit exactly (Euclidean, with or
without gate and clipping) or along the path of states (other geometries);
``rank_causes`` orders a history's events by that contribution;
``counterfactual`` re-encodes an edited history; ``event_gates`` returns the
learned gates. Run with `pytest tests/ -q`.
"""
import numpy as np
import pytest
import torch

from event2vector import Event2Vec

V = 6
SEQS = [[0, 1, 2, 3], [0, 1, 2, 4], [0, 2, 3, 5], [1, 3, 4, 5], [0, 2, 1, 4]] * 6


def _fit(**kw):
    kw.setdefault("embedding_dim", 4)
    est = Event2Vec(num_event_types=V, num_epochs=2, batch_size=8, device="cpu",
                    random_state=0, **kw)
    return est.fit(SEQS)


def _randomize(est, scale=0.5):
    """Non-trivial parameters, so gates and clipping actually act."""
    m = est.model
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        m.embedding.weight.copy_(torch.randn(m.embedding.weight.shape, generator=g) * scale)
        if getattr(m, "gate", None) == "vector":
            m.gate_embedding.weight.copy_(torch.randn(m.gate_embedding.weight.shape, generator=g))
        if getattr(m, "gate", None) == "tied":
            m.gate_linear.weight.copy_(torch.randn(m.gate_linear.weight.shape, generator=g))
    return est


SEQ = [0, 3, 1, 3, 5, 2, 4]


# ------------------------------------------------------------------ gate
def test_gate_zero_init_is_exact_sum():
    for gate in ("vector", "tied"):
        est = Event2Vec(num_event_types=V, embedding_dim=4, gate=gate, dropout_p=0.0, device="cpu")
        est._reset_parameters()
        h0 = torch.randn(2, 4)
        h, e = est.model.encoder(torch.tensor([1, 2]), h0, clipping=False)
        assert torch.allclose(h, h0 + e)


def test_gate_update_rule():
    """h_t = (1 + g_e) * h_{t-1} + e_t: savings 100 -> 70 for g = -0.3."""
    est = Event2Vec(num_event_types=V, embedding_dim=2, gate="vector", dropout_p=0.0,
                    max_norm=1e6, device="cpu")
    m = est.model
    with torch.no_grad():
        m.embedding.weight[1] = torch.tensor([0.0, 1.0])
        m.gate_embedding.weight[1] = torch.atanh(torch.tensor([-0.3, 0.0]))
    h, _ = m.encoder(torch.tensor([1]), torch.tensor([[100.0, 10.0]]))
    assert torch.allclose(h, torch.tensor([[70.0, 11.0]]))


def test_reconstruction_term_is_additivity_penalty():
    """With a gate the 0.4.1 reconstruction term equals mean((g * h_{t-1})^2)."""
    est = _randomize(_fit(gate="vector", max_norm=1e6, dropout_p=0.0))
    x, h_old = torch.tensor([1, 4]), torch.randn(2, 4)
    y1, h1, e1 = est.model(x, h_old)
    _, h2, _ = est.model(x, h_old)
    _, rec, _ = est._compute_losses(y1, h1, e1, h_old, torch.tensor([0, 0]), h2)
    expected = (est.model.gate_values(x) * h_old).pow(2).mean()
    assert torch.allclose(rec, expected)


def test_gate_rejected_for_other_geometries():
    with pytest.raises(ValueError):
        Event2Vec(num_event_types=V, geometry="hyperbolic", gate="vector")
    with pytest.raises(ValueError):
        Event2Vec(num_event_types=V, gate="scalar")


def test_default_model_has_no_gate_parameters():
    est = Event2Vec(num_event_types=V, embedding_dim=4)
    assert not any("gate" in name for name in est.model.state_dict())


# ------------------------------------------------------------------ attribute
@pytest.mark.parametrize("kw", [
    dict(max_norm=1e6),                    # exact sum
    dict(max_norm=0.5),                    # clipping fires
    dict(max_norm=1e6, gate="vector"),
    dict(max_norm=0.5, gate="vector"),
    dict(max_norm=1e6, gate="tied"),
])
def test_euclidean_attribution_is_exact(kw):
    est = _randomize(_fit(**kw))
    for target in range(V):
        out = est.attribute(SEQ, target=target)
        assert out["method"] == "decomposition"
        assert out["baseline"] == pytest.approx(float(est.model.linear_dec.bias[target].detach()), abs=1e-6)
        assert out["baseline"] + out["contributions"].sum() == pytest.approx(out["logit"], abs=1e-5)
        assert sum(out["by_event"].values()) == pytest.approx(out["contributions"].sum(), abs=1e-6)


def test_ungated_attribution_is_decoder_times_embedding():
    est = _randomize(_fit(max_norm=1e6))
    out = est.attribute(SEQ, target=2)
    E, W = est.model.embedding.weight.detach(), est.model.linear_dec.weight.detach()
    assert np.allclose(out["contributions"], (E[SEQ] @ W[2]).numpy(), atol=1e-6)


def test_gate_scales_earlier_contributions():
    """An event's contribution is its embedding times the gates of later events."""
    est = _randomize(_fit(max_norm=1e6, gate="vector"))
    E, W = est.model.embedding.weight.detach(), est.model.linear_dec.weight.detach()
    g = torch.tensor(est.event_gates())
    seq = [0, 3, 1]
    out = est.attribute(seq, target=4)
    first = (E[0] * (1 + g[3]) * (1 + g[1])) @ W[4]
    assert out["contributions"][0] == pytest.approx(float(first), abs=1e-6)


def test_attribution_defaults_to_predicted_event():
    est = _fit()
    out = est.attribute(SEQ)
    probs = est.counterfactual(SEQ)["probs"]
    assert out["target"] == int(np.argmax(probs))


@pytest.mark.parametrize("geometry,kw", [("hyperbolic", {}), ("rotation", dict(rotation_n=3))])
def test_path_attribution_sums_for_other_geometries(geometry, kw):
    est = _fit(geometry=geometry, **kw)
    out = est.attribute(SEQ, target=1)
    assert out["method"] == "path"
    assert out["baseline"] + out["contributions"].sum() == pytest.approx(out["logit"], abs=1e-4)


def test_attribute_validates_input():
    est = _fit()
    with pytest.raises(ValueError):
        est.attribute([])
    with pytest.raises(ValueError):
        est.attribute([0, V])
    with pytest.raises(ValueError):
        est.attribute([0, 1], target=V)


# ------------------------------------------------------------------ rank_causes
def test_rank_causes_orders_by_contribution_and_excludes_effect():
    est = _randomize(_fit(max_norm=1e6, gate="vector"))
    hist, effect = [0, 3, 1, 3, 5], 3
    ranked = est.rank_causes(hist, effect)
    by_event = est.attribute(hist, target=effect)["by_event"]
    assert [e for e, _ in ranked] == sorted((e for e in by_event if e != effect),
                                            key=lambda e: -by_event[e])
    assert effect not in dict(ranked)
    assert effect in dict(est.rank_causes(hist, effect, exclude_effect=False))


def test_rank_causes_finds_a_planted_cause():
    """An event whose embedding points at the effect's decoder row ranks first."""
    est = _fit(max_norm=1e6, dropout_p=0.0)
    m = est.model
    with torch.no_grad():
        m.embedding.weight.mul_(0.01)
        m.embedding.weight[2] = m.linear_dec.weight[5] / m.linear_dec.weight[5].norm()
    assert est.rank_causes([0, 1, 2, 3, 4], effect=5)[0][0] == 2


# ------------------------------------------------------------------ counterfactual
@pytest.mark.parametrize("geometry", ["euclidean", "hyperbolic"])
def test_counterfactual_matches_reencoding(geometry):
    est = _fit(geometry=geometry)
    out = est.counterfactual(SEQ, remove=[1, 4], replace={0: 5}, insert=[(2, 4), (len(SEQ), 0)])
    edited = [5, 4, 1, 3, 2, 4, 0]
    assert out["sequence_counterfactual"] == edited
    direct = est.counterfactual(edited)["probs"]
    assert np.allclose(out["probs_counterfactual"], direct, atol=1e-6)
    assert np.allclose(out["delta"], out["probs_counterfactual"] - out["probs"])
    assert out["probs"].sum() == pytest.approx(1.0, abs=1e-5)


def test_counterfactual_without_edits_is_identity():
    est = _fit(gate="vector")
    out = est.counterfactual(SEQ)
    assert out["sequence_counterfactual"] == SEQ
    assert np.allclose(out["delta"], 0.0)


def test_counterfactual_validates_positions():
    est = _fit()
    with pytest.raises(ValueError):
        est.counterfactual(SEQ, remove=len(SEQ))
    with pytest.raises(ValueError):
        est.counterfactual(SEQ, insert=[(len(SEQ) + 1, 0)])


# ------------------------------------------------------------------ event_gates
def test_event_gates_shape_and_values():
    est = _randomize(_fit(gate="tied"))
    g = est.event_gates()
    assert g.shape == (V, 4)
    E = est.model.embedding.weight.detach()
    expected = torch.tanh(E @ est.model.gate_linear.weight.detach().T).numpy()
    assert np.allclose(g, expected, atol=1e-6)
    assert np.all(np.abs(g) < 1.0)


def test_event_gates_requires_a_gate():
    with pytest.raises(ValueError):
        _fit().event_gates()


def test_root_cause_methods_require_fit():
    est = Event2Vec(num_event_types=V, gate="vector")
    for call in (lambda: est.attribute([0, 1]), lambda: est.rank_causes([0, 1], 2),
                 lambda: est.counterfactual([0, 1]), lambda: est.event_gates()):
        with pytest.raises(RuntimeError):
            call()
