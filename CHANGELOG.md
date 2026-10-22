# Changelog

All notable changes to `event2vector` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-10-22

The Euclidean model gains an optional per-event gate on the history, and the
estimator gains root-cause methods that explain its next-event predictions. Both are
opt-in: with default arguments, training is identical to 0.4.1. This release builds
on 0.4.1; the history-reconstruction objective of 0.5.0 is not carried forward.

### Added

- `gate` (Euclidean, default `None`): each event can rescale the history
  coordinate-wise before its vector is added, `h_t = (1 + g_e) * h_{t-1} + e_t`
  with `g_e = tanh(a_e)`, so the factor `1 + g_e` stays in `(0, 2)`.
  `gate="vector"` learns `a_e` as a second vector per event; `gate="tied"`
  computes it from the event's own embedding, `a_e = U e`. The gate starts at zero,
  so training starts from the exact sum. With a gate, the reconstruction term
  `‖(h_t − e_t) − h_{t-1}‖²` equals `‖g_e * h_{t-1}‖²`: it becomes an additivity
  penalty, and `lambda_reconstruction` sets how much the model pays to depart
  from the pure sum.
- `attribute(sequence, target=None)`: the contribution of each earlier event to
  the logit of `target` (default: the predicted next event), also summed per event
  id. Euclidean: an exact decomposition of the state into per-event vectors, scaled
  by the gates of later events and by norm clipping, so the contributions plus the
  decoder bias equal the logit. Other geometries: the change in the target logit at
  each step, which also sums exactly but depends on the order of events.
- `rank_causes(history, effect)`: the distinct events of a history ranked by
  their contribution to `effect`.
- `counterfactual(sequence, remove=, replace=, insert=)`: the next-event
  distribution before and after editing the history by position; the edited history
  is re-encoded, so it is exact for every geometry.
- `event_gates()`: the learned gate of every event; negative coordinates shrink
  that part of the history.

These methods explain the model's predictions. They are associations learned from
the data, not causal effects.

## [0.4.1] - 2026-10-15

Documentation and metadata only; the code is unchanged from 0.4.0.

### Changed

- The paper is published in the Proceedings of Machine Learning Research
  (vol. 282): the README citation, the "Paper" link and `CITATION.cff` (GitHub's
  "Cite this repository" and Zenodo) now point to
  [PMLR](https://proceedings.mlr.press/v282/sulc26a.html); arXiv is kept as the
  preprint.
- Rewritten README: quickstart, how training works, the geometries with a figure
  of the same events composed by each geometry, a parameter and method
  reference, devices, datasets, examples, citation, and the event2vec logo.
- The Brown corpus example's extra requirements are documented
  (`pip install "event2vector[viz,manifold,nlp]" nltk`).

### Added

- `images/make_readme_figures.py`, which redraws the README's logo and geometry
  figure from the library's own model classes.

## [0.4.0] - 2026-10-15

### Added

- `geometry="affine"` (`AffineModel`): each event is an invertible affine map
  `x -> exp(M_e) x + v_e` applied on the left of an affine state, so an event can
  act on the history (shrink, stretch or rotate it) instead of only adding to it.
  With `M_e = 0`, the initialisation, the point is exactly the Euclidean sum. The
  step is recovered exactly by the reconstruction loss, and subsequences compose
  into one affine map:
  - `AffineModel.point(state)` returns the plotted point `b` from a state;
  - `AffineModel.compose(event_ids)` returns one `(A, v)` for a subsequence;
  - `AffineModel.override_strength(event_ids)` returns the singular values of its
    linear part.
- Estimator options `affine_readout` (`"point"` or `"full"`) and
  `affine_max_log_scale` (bounds how much one event can stretch or shrink the state).
- Tests for the affine geometry (`tests/test_affine.py`).

### Changed

- The MPS-safe `matrix_exp` is now a shared helper used by both the `rotation` and
  `affine` geometries, and automatic device selection falls back to the CPU for
  `affine` as it already did for `rotation`.

## [0.3.1] - 2026-10-11

### Fixed

- `Event2Vec(geometry="rotation")` no longer crashes on Apple Silicon with
  `NotImplementedError: The operator 'aten::linalg_matrix_exp' is not currently
  implemented for the MPS device`. `RotationModel` now computes the matrix
  exponential on the CPU when its input is on MPS and moves the result back, so
  `device="mps"`, a bare `RotationModel` and a `ProductModel` with a rotation
  factor all work.
- The project links on PyPI pointed to `github.com/sulcantonin/event2vector`,
  which does not exist; they now point to
  [`github.com/sulcantonin/event2vec_public`](https://github.com/sulcantonin/event2vec_public).

### Changed

- When the device is chosen automatically and MPS would be picked, the
  `rotation` geometry runs on the CPU instead and emits a `RuntimeWarning`. On
  MPS every step round-trips through the CPU, which made training roughly 8–14×
  slower than on the CPU alone for this model. Pass `device="mps"` to use MPS
  anyway. CUDA and the other geometries are unaffected.

### Added

- Tests for the automatic device selection (`tests/test_device.py`).
- This changelog, linked from the PyPI project page.

## [0.3.0] - 2026-07-21

Earlier releases are listed on [PyPI](https://pypi.org/project/event2vector/#history).

[0.4.1]: https://pypi.org/project/event2vector/0.4.1/
[0.4.0]: https://pypi.org/project/event2vector/0.4.0/
[0.3.1]: https://pypi.org/project/event2vector/0.3.1/
[0.3.0]: https://pypi.org/project/event2vector/0.3.0/
