"""Device-selection tests: the default constructor must work on every backend.

The rest of the suite pins ``use_gpu=False``, so these are the tests that exercise
the estimator's automatic CUDA / MPS / CPU choice. Run with `pytest tests/ -q`.
"""
import numpy as np
import pytest
import torch

from event2vector import Event2Vec, ProductModel, RotationModel, StereographicModel

MPS_AVAILABLE = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

TINY_SEQS = [[0, 1, 2], [2, 1, 0, 1], [1, 2], [0, 2, 2, 1]]


@pytest.mark.filterwarnings("ignore:geometry='rotation' needs torch.matrix_exp")
def test_rotation_fits_with_default_device():
    """Event2Vec(geometry="rotation") with no device arguments must fit on any machine
    (it used to raise NotImplementedError for aten::linalg_matrix_exp on Apple Silicon)."""
    est = Event2Vec(num_event_types=3, geometry="rotation", rotation_n=3,
                    num_epochs=2, batch_size=2, random_state=0)
    assert est.device.type != "mps"
    est.fit(TINY_SEQS)

    assert est.is_fitted_
    assert np.isfinite(est.training_loss_).all()
    emb = est.transform(TINY_SEQS)
    assert emb.shape == (len(TINY_SEQS), 9)  # flattened 3x3 rotation
    assert np.isfinite(emb).all()


def test_auto_device_skips_mps_for_rotation(monkeypatch):
    """When MPS would be auto-selected, rotation falls back to CPU with a warning.
    Simulated so it runs on any platform; the model never touches MPS."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    with pytest.warns(RuntimeWarning, match="matrix_exp"):
        est = Event2Vec(num_event_types=3, geometry="rotation")
    assert est.device.type == "cpu"


@pytest.mark.skipif(not MPS_AVAILABLE, reason="needs an MPS device")
def test_rotation_runs_on_explicit_mps():
    """device="mps" is honoured: matrix_exp runs on the CPU and the result (and its
    gradient) matches a pure-CPU computation."""
    torch.manual_seed(0)
    gen = torch.randn(5, 3)
    cpu_model = RotationModel(3, n=3)
    mps_model = RotationModel(3, n=3).to("mps")

    g_cpu = gen.clone().requires_grad_()
    r_cpu = cpu_model._rot(g_cpu)
    (r_cpu ** 3).sum().backward()
    g_mps = gen.clone().to("mps").requires_grad_()
    r_mps = mps_model._rot(g_mps)
    (r_mps ** 3).sum().backward()

    assert r_mps.device.type == "mps"
    assert torch.allclose(r_cpu, r_mps.cpu(), atol=1e-5)
    assert torch.allclose(g_cpu.grad, g_mps.grad.cpu(), atol=1e-5)

    # a product manifold with a rotation factor goes through the same path
    product = ProductModel([
        RotationModel(3, n=3),
        StereographicModel(3, embedding_dim=4, kappa=-1.0),
    ]).to("mps")
    logits, _, _ = product(torch.tensor([0, 1, 2], device="mps"))
    assert torch.isfinite(logits).all()

    est = Event2Vec(num_event_types=3, geometry="rotation", device="mps",
                    num_epochs=2, batch_size=2, random_state=0)
    est.fit(TINY_SEQS)
    assert np.isfinite(est.training_loss_).all()

