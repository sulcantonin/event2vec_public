"""Tests for the Euclidean event-history reconstruction loss.

``L_recon = sum_t ||alpha* E (h_t - e_{s_t}) / (t-1) - c_{t-1} / (t-1)||^2`` with
``h_{t-1} = E^T c_{t-1}`` and the closed-form scale
``alpha* = <E E^T c, c> / ||E E^T c||^2``. Run with `pytest tests/ -q`.
"""
import pytest
import torch

from event2vector import Event2Vec

V, D = 6, 4


def _est(**kw):
    kw = dict(dict(num_event_types=V, embedding_dim=D, dropout_p=0.0,
                   use_gpu=False, random_state=0), **kw)
    est = Event2Vec(**kw)
    torch.manual_seed(0)
    with torch.no_grad():
        est.model.embedding.weight.normal_(0.0, 0.5)
    return est


def _record_counts(est, monkeypatch):
    """Capture every histogram handed to the reconstruction loss."""
    seen = []
    original = est._history_reconstruction_loss

    def wrapper(counts):
        seen.append(counts.detach().clone())
        return original(counts)

    monkeypatch.setattr(est, "_history_reconstruction_loss", wrapper)
    return seen


def _batch_loss(est, seqs, padded):
    opt = torch.optim.SGD(est.model.parameters(), lr=0.0)
    seqs = [torch.tensor(s) for s in seqs]
    if padded:
        return est._train_batch_padded(seqs, opt)
    return est._train_batch_unpadded(seqs, opt)


def test_loss_is_nonzero_for_random_embeddings():
    est = _est()
    counts = torch.randint(0, 3, (8, V)).float()
    counts[:, 0] += 1  # no empty histories
    assert est._history_reconstruction_loss(counts).item() > 1e-3


def test_loss_has_nonzero_embedding_gradient():
    est = _est()
    counts = torch.tensor([[2.0, 1.0, 0.0, 0.0, 1.0, 0.0]])
    est._history_reconstruction_loss(counts).backward()
    assert est.model.embedding.weight.grad.norm().item() > 1e-4


@pytest.mark.parametrize("padded", [False, True])
def test_histogram_counts_repeated_events(monkeypatch, padded):
    est = _est()
    seen = _record_counts(est, monkeypatch)
    _batch_loss(est, [[3, 3, 1, 3, 0]], padded)
    expected = [{3: 1}, {3: 2}, {3: 2, 1: 1}, {3: 3, 1: 1}]  # last event is only a target
    assert len(seen) == len(expected)
    for counts, exp in zip(seen, expected):
        target = torch.zeros(1, V)
        for k, v in exp.items():
            target[0, k] = v
        assert torch.equal(counts, target)


def test_loss_is_invariant_to_embedding_scale():
    est = _est()
    counts = torch.tensor([[1.0, 0.0, 2.0, 0.0, 3.0, 1.0], [0.0, 4.0, 0.0, 1.0, 0.0, 0.0]])
    before = est._history_reconstruction_loss(counts)
    with torch.no_grad():
        est.model.embedding.weight.mul_(7.0)
    assert torch.allclose(est._history_reconstruction_loss(counts), before, rtol=1e-4)


def test_empty_histories_are_excluded():
    est = _est()
    counts = torch.tensor([[1.0, 0.0, 2.0, 0.0, 0.0, 1.0]])
    with_empty = torch.cat([counts, torch.zeros(2, V)])
    assert torch.allclose(est._history_reconstruction_loss(with_empty),
                          est._history_reconstruction_loss(counts))
    assert est._history_reconstruction_loss(torch.zeros(3, V)).item() == 0.0


def test_padding_is_excluded(monkeypatch):
    # pad_value=0 is also a real event id, so a leaked pad would change the counts.
    seqs = [[0, 1, 2, 3, 4], [2, 2], [5, 0, 5]]
    padded_est, unpadded_est = _est(pad_value=0), _est(pad_value=0)
    seen = _record_counts(padded_est, monkeypatch)
    loss_padded = _batch_loss(padded_est, seqs, padded=True)
    loss_unpadded = _batch_loss(unpadded_est, seqs, padded=False)
    assert loss_padded == pytest.approx(loss_unpadded, rel=1e-5)
    # every row's histogram length equals its step index + 1, never more
    for t, counts in enumerate(seen):
        assert torch.all(counts.sum(dim=1) == t + 1)
    assert seen[-1].shape[0] == 1  # only the longest sequence is left at the end


def test_batched_loss_matches_definition():
    seqs = [[0, 1, 1, 4, 2], [3, 3, 5], [2, 0, 2, 2]]
    est = _est(lambda_consistency=0.0)
    E = est.model.embedding.weight.detach()

    # Reference: sum_{t=2}^{T} ||alpha* E (h_t - e_{s_t}) - c_{t-1}||^2 / (t-1)^2,
    # with the additive state built step by step (no clipping, no dropout).
    terms = []
    for seq in seqs:
        h = torch.zeros(D)
        c = torch.zeros(V)
        for t in range(1, len(seq)):  # s_t = seq[t] (0-based), history seq[:t]
            h = h + E[seq[t - 1]]
            c[seq[t - 1]] += 1
            h_t = h + E[seq[t]]
            c_hat = E @ (h_t - E[seq[t]])
            alpha = (c_hat @ c) / (c_hat @ c_hat)
            terms.append(((alpha * c_hat - c) / t).pow(2).sum())
    reference = torch.stack(terms).mean().item()

    for padded in (False, True):
        est.lambda_reconstruction = 0.0
        base = _batch_loss(est, seqs, padded)
        est.lambda_reconstruction = 1.0
        full = _batch_loss(est, seqs, padded)
        assert full - base == pytest.approx(reference, rel=1e-4)


def test_reconstruction_gradient_reaches_embedding_in_training():
    est = _est(lambda_consistency=0.0)
    seqs = [torch.tensor([0, 1, 2, 3])]
    grads = {}
    for lam in (0.0, 1.0):
        est.lambda_reconstruction = lam
        est.model.zero_grad()
        opt = torch.optim.SGD(est.model.parameters(), lr=0.0)
        opt.zero_grad = lambda *a, **k: None  # keep the grads for inspection
        est._train_batch_unpadded(seqs, opt)
        grads[lam] = est.model.embedding.weight.grad.clone()
    assert (grads[1.0] - grads[0.0]).norm().item() > 1e-4


@pytest.mark.parametrize("lam", [0.0, 0.2])
def test_next_event_prediction_still_learns(lam):
    seqs = [[0, 1, 2, 3, 4]] * 16
    est = Event2Vec(num_event_types=5, embedding_dim=8, dropout_p=0.0,
                    lambda_reconstruction=lam, num_epochs=60, learning_rate=1e-2,
                    batch_size=8, use_gpu=False, random_state=0)
    est.fit(seqs)
    est.model.eval()
    h = est.model.zero_state(1, est.device)
    correct = 0
    with torch.no_grad():
        for t in range(4):
            logits, h, _ = est.model(torch.tensor([t]), h)
            correct += int(logits.argmax(dim=1).item() == t + 1)
    assert correct == 4


@pytest.mark.parametrize("max_norm", [None, float("inf")])
def test_unclipped_state_is_exact_sum_and_trains(max_norm):
    est = _est(max_norm=max_norm, num_epochs=2, batch_size=2)
    E = est.model.embedding.weight.detach().clone()
    h = est.model.zero_state(1, est.device)
    for i in [3, 1, 3, 5]:
        h = est.model(torch.tensor([i]), h)[1]
    assert torch.allclose(h[0], E[[3, 1, 3, 5]].sum(0), atol=1e-5)
    est.fit([[0, 1, 2, 3], [3, 3, 1], [5, 0, 5, 2]])
    assert all(torch.isfinite(p).all() for p in est.model.parameters())


def test_difference_loss_is_nonzero_scale_invariant_and_trains():
    est = _est(lambda_difference=0.5)
    counts = torch.tensor([[2.0, 1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 3.0, 0.0, 0.0, 1.0],
                           [1.0, 0.0, 0.0, 2.0, 0.0, 0.0]])
    torch.manual_seed(0)
    loss = est._difference_reconstruction_loss(counts)
    assert loss.item() > 1e-3
    loss.backward()
    assert est.model.embedding.weight.grad.norm().item() > 1e-4
    with torch.no_grad():
        est.model.embedding.weight.mul_(5.0)
    torch.manual_seed(0)
    assert torch.allclose(est._difference_reconstruction_loss(counts), loss, rtol=1e-4)
    # identical histograms give no pair; fewer than two rows give no pair
    same = counts[:1].repeat(3, 1)
    assert est._difference_reconstruction_loss(same).item() == 0.0
    assert est._difference_reconstruction_loss(counts[:1]).item() == 0.0
    est = _est(lambda_difference=0.5, num_epochs=2, batch_size=3, pad_sequences=True)
    est.fit([[0, 1, 2, 3], [3, 3, 1, 5], [5, 0, 5, 2, 4]])
    assert all(torch.isfinite(p).all() for p in est.model.parameters())


def test_difference_loss_only_in_padded_training():
    seqs = [torch.tensor([0, 1, 2, 3]), torch.tensor([3, 3, 1, 5])]
    est = _est(lambda_difference=1.0, lambda_consistency=0.0)
    opt = torch.optim.SGD(est.model.parameters(), lr=0.0)
    torch.manual_seed(0)
    with_term = est._train_batch_padded(seqs, opt)
    est.lambda_difference = 0.0
    torch.manual_seed(0)
    without = est._train_batch_padded(seqs, opt)
    assert with_term > without
    est.lambda_difference = 1.0
    torch.manual_seed(0)
    unpadded_with = est._train_batch_unpadded(seqs, opt)
    est.lambda_difference = 0.0
    torch.manual_seed(0)
    assert unpadded_with == pytest.approx(est._train_batch_unpadded(seqs, opt))
