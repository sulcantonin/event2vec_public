# Changelog

All notable changes to `event2vector` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-10-21

The Euclidean model gets a reconstruction objective that acts on whole histories
and a set of root-cause analysis methods built on the exact decomposition of its
state. Other geometries are unchanged; the API is backward compatible.

### Added

- Root-cause analysis methods on `Event2Vec`: `attribute` (exact per-event
  logit contributions for Euclidean, leave-one-out otherwise), `counterfactual`,
  `influence_matrix`, `event_impact`, and for Euclidean `decompose` and
  `explain_difference` (event counts recovered from states by non-negative least
  squares).
- `metrics.transition_graph_auc`, which scores the influence matrix against a
  known transition graph.
- `lambda_difference` (experimental, off by default): a difference-reconstruction
  term for the Euclidean model. Random pairs of histories in a padded batch must
  have a state difference that decodes (with the same scale-invariant form) to
  the events that differ. In our experiments it lowers the reconstruction error
  and improves the decoding of sums, but does not improve the decoding of
  differences; it is kept as an option for further work.
- `max_norm=None` disables the Euclidean norm clipping, so the state is the exact
  sum of its events.

### Changed

- The Euclidean reconstruction loss is upgraded to a history-level objective.
  It reconstructs the normalized event histogram `c/n` of each history from its
  additive embedding `Eᵀ c/n`, with the event-embedding matrix as a tied decoder:
  `‖α* E Eᵀ c/n − c/n‖²`, where `α* = ⟨E Eᵀ c, c⟩ / ‖E Eᵀ c‖²` is the
  closed-form best scale (no trainable parameter), so the loss does not compete
  with prediction for the embedding norm. The embedding is rebuilt without dropout
  or clipping so the gradient reaches `E`. The previous step-level term
  `‖(h_t − e_t) − h_{t-1}‖²` is satisfied by construction of the additive update
  (it acted only when `max_norm` clipping fired). The other geometries are
  unchanged.

### Fixed

- An infinite `max_norm` previously produced NaN gradients (`0 · inf` in the
  backward pass of the clip).

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
