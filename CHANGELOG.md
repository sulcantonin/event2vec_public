# Changelog

All notable changes to `event2vector` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.4.0]: https://pypi.org/project/event2vector/0.4.0/
[0.3.1]: https://pypi.org/project/event2vector/0.3.1/
[0.3.0]: https://pypi.org/project/event2vector/0.3.0/
