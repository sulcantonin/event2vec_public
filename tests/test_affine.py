"""Tests for the affine geometry (``geometry="affine"``, :class:`AffineModel`).

The affine state is a map ``x -> A x + b``; each event applies
``x -> exp(M_e) x + v_e`` on the left. Run with `pytest tests/ -q`.
"""
import math

import numpy as np
import pytest
import torch

from event2vector import AffineModel, Event2Vec

SEQS = [[0, 1, 2, 3], [0, 1, 2, 4], [0, 2, 3, 4], [0, 1, 3, 4], [0, 2, 1, 4]] * 8


def _affine(dim=2, **kw):
    am = AffineModel(num_event_types=5, embedding_dim=dim, **kw)
    am.eval()
    with torch.no_grad():  # a non-trivial linear part for every event
        am.embedding.weight.normal_(0.0, 0.3)
    return am


def _feed(model, ids):
    h = model.zero_state(1, torch.device("cpu"))
    for i in ids:
        h = model.forward(torch.tensor([i]), h)[1]
    return h


@pytest.mark.parametrize("kw", [dict(), dict(affine_readout="full",
                                             affine_max_log_scale=1.0)])
def test_affine_fits_transforms_and_learns(kw):
    est = Event2Vec(num_event_types=5, geometry="affine", embedding_dim=2,
                    dropout_p=0.0, num_epochs=40, learning_rate=5e-3, batch_size=8,
                    pad_sequences=True, random_state=0, use_gpu=False, **kw)
    est.fit(SEQS)
    emb = est.transform(SEQS)
    assert emb.shape == (len(SEQS), est.model.state_dim)  # [vec(A), b] = 4 + 2
    assert np.isfinite(emb).all()
    assert min(est.training_loss_) < est.training_loss_[0] * 0.98
    points = est.model.point(torch.as_tensor(emb))
    assert points.shape == (len(SEQS), 2)


def test_affine_with_identity_maps_is_the_euclidean_sum():
    """M_e = 0 (the initialisation) makes the point b exactly the sum of v_e."""
    am = AffineModel(num_event_types=5, embedding_dim=3)
    am.eval()
    ids = [0, 3, 1, 3]
    v = am.embedding.weight[:, :3].detach()
    assert torch.allclose(am.point(_feed(am, ids))[0], v[ids].sum(0), atol=1e-6)


def test_affine_compose_equals_stepwise_and_is_order_sensitive():
    am = _affine()
    ids = [1, 2, 3, 4]
    a, v = am.compose(ids)
    assert torch.allclose(am.point(_feed(am, ids))[0], v, atol=1e-5)
    # applied to any state, the composed map equals feeding the events one by one
    _, b_stem = am._split(_feed(am, [0, 2]))
    after = _feed(am, [0, 2] + ids)
    assert torch.allclose(am.point(after)[0], a @ b_stem[0] + v, atol=1e-5)
    assert not torch.allclose(am.compose([1, 2])[1], am.compose([2, 1])[1], atol=1e-4)


def test_affine_recover_increment_is_exact():
    """The reconstruction loss's left-cancellation recovers each step exactly."""
    am = _affine()
    h0 = _feed(am, [0, 1, 2])
    h1, step = am.encoder(torch.tensor([3]), h0)
    assert torch.allclose(am.recover_increment(h0, h1), step, atol=1e-5)
    assert am.dist2(am.recover_increment(h0, h1), step).item() < 1e-8


def test_affine_max_log_scale_bounds_each_step():
    am = _affine(max_log_scale=0.5)
    with torch.no_grad():
        am.embedding.weight.mul_(100.0)
    s = am.override_strength([2])
    bound = math.exp(2 * 2 * 0.5)
    assert s.max().item() <= bound + 1e-4 and s.min().item() >= 1 / bound - 1e-4


def test_affine_rejects_unknown_readout():
    with pytest.raises(ValueError):
        AffineModel(num_event_types=5, readout="nope")


def test_auto_device_skips_mps_for_affine(monkeypatch):
    """Like rotation, affine needs matrix_exp: auto-selection prefers the CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    with pytest.warns(RuntimeWarning, match="geometry='affine' needs torch.matrix_exp"):
        est = Event2Vec(num_event_types=3, geometry="affine")
    assert est.device.type == "cpu"
