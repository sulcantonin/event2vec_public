<div align="center">

<a href="https://event2vec.com">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/sulcantonin/event2vec_public/main/images/logo-dark.svg">
    <img alt="event2vec" src="https://raw.githubusercontent.com/sulcantonin/event2vec_public/main/images/logo-light.svg" width="300">
  </picture>
</a>

**Interpretable, composable embeddings for sequences of discrete events.**

[![PyPI version](https://badge.fury.io/py/event2vector.svg)](https://badge.fury.io/py/event2vector)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23399003.svg)](https://doi.org/10.5281/zenodo.23399003)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PMLR](https://img.shields.io/badge/PMLR-vol.%20282-1f4e79.svg)](https://proceedings.mlr.press/v282/sulc26a.html)
[![arXiv](https://img.shields.io/badge/arXiv-2509.12188-b31b1b.svg)](https://arxiv.org/abs/2509.12188)

**[event2vec.com](https://event2vec.com)** · [Paper](https://proceedings.mlr.press/v282/sulc26a.html) · [Changelog](https://github.com/sulcantonin/event2vec_public/blob/main/CHANGELOG.md) · [Issues](https://github.com/sulcantonin/event2vec_public/issues)

![Event2Vector teaser](https://raw.githubusercontent.com/sulcantonin/event2vec_public/main/images/teaser.png)

</div>

## What it is

Event2Vector learns one vector per event type and represents a whole sequence as
the **composition of its events**. In the default Euclidean geometry that
composition is a plain sum, so a sequence's embedding is *exactly* the sum of its
events' vectors. This makes the representation interpretable by construction:

- **Decomposition**: every coordinate of a sequence is attributable to the events in it.
- **Composition**: sub-sequences combine by vector arithmetic.
- **Counterfactuals**: removing an event from a sequence is one subtraction.

Other geometries (hyperbolic, spherical, rotation, Heisenberg, affine, product)
keep this compositional structure while capturing hierarchy, cycles or order. All
of them sit behind one scikit-learn-style estimator.

## Installation

```bash
pip install event2vector
```

From source:

```bash
git clone https://github.com/sulcantonin/event2vec_public.git
cd event2vec_public
pip install .
```

Requires Python 3.9+ and PyTorch 1.13+.

## Quickstart

Sequences are lists of integer event ids (`0 … num_event_types - 1`) and can have
different lengths.

```python
import numpy as np
from event2vector import Event2Vec

vocab = ["start", "login", "search", "buy", "logout"]
sequences = [[0, 1, 2, 3, 4], [0, 1, 2, 2, 4], [0, 2, 3, 4], [0, 1, 4]] * 50

model = Event2Vec(
    num_event_types=len(vocab),
    embedding_dim=8,
    geometry="euclidean",
    pad_sequences=True,   # vectorized batches of variable-length sequences
    num_epochs=20,
    random_state=0,
)
model.fit(sequences)

embeddings = model.transform(sequences)          # (n_sequences, 8)

# A sequence is exactly the sum of its event vectors:
event_vectors = model.model.embedding.weight.detach().cpu().numpy()
assert np.allclose(embeddings[0], event_vectors[sequences[0]].sum(axis=0), atol=1e-5)

# gensim-style nearest events to an event or a whole sequence
print(model.most_similar(positive=3, topn=2))           # [(event_id, similarity), ...]
print(model.most_similar(positive=[[0, 1, 2]], topn=2))
```

## How it works

A sequence is read one event at a time. The state starts at the identity and each
event updates it, `h_t = h_{t-1} ⊕ e_t`, where `⊕` is the geometry's composition
(vector addition in the Euclidean case). Training minimizes three terms at every
step:

| term | weight | what it enforces |
|---|---|---|
| next-event prediction | 1 | a linear decoder predicts event `t+1` from `h_t` (cross-entropy) |
| reconstruction | `lambda_reconstruction` | Euclidean: the history's event counts can be read back from its embedding with the tied decoder `E`, `‖α* E Eᵀ c/n − c/n‖²` with the closed-form best scale `α*`, so it does not fix the embedding norm; other geometries: the step `e_t` is recovered from `h_{t-1}` and `h_t` |
| consistency | `lambda_consistency` | two dropout passes over the same step agree |

There is no attention and no hidden nonlinearity in the state update, which is
what keeps every embedding a readable composition of its events. See the
[paper](https://proceedings.mlr.press/v282/sulc26a.html) for details.

## Geometries

Choose the composition with `geometry=`. Every geometry uses the same estimator
and, apart from `euclidean`, a generic left-cancellation reconstruction loss.

| `geometry` | update | best for |
|---|---|---|
| `euclidean` | vector addition | order-invariant bag of events, exact decomposition |
| `hyperbolic` | Möbius addition (curvature `curvature > 0`) | hierarchical, tree-like structure |
| `stereographic` | κ-gyroaddition (signed `kappa`) | one curvature dial: `kappa < 0` hyperbolic, `0` Euclidean, `> 0` spherical |
| `spherical` | κ-gyroaddition (`kappa > 0`) | cyclic / periodic structure |
| `rotation` | left `SO(n)` product | non-commutative orientation (order matters) |
| `heisenberg` | Heisenberg group product | order-dependent signed (Lévy) area |
| `affine` | `x ↦ exp(M_e) x + v_e` on the left | events that act on the history (shrink, stretch, rotate it), not only add to it |
| `product` | per-factor update | two structure types at once, e.g. hyperbolic × spherical |

![The same three events composed by each geometry](https://raw.githubusercontent.com/sulcantonin/event2vec_public/main/images/geometries.png)

*The same three events composed by each geometry, in order a·b·c (blue) and c·b·a
(red). The Euclidean sum ends in the same place either way; where the endpoints
separate, the geometry remembers order. Drawn by the library's own model classes
with `python images/make_readme_figures.py`.*

```python
Event2Vec(num_event_types=len(vocab), geometry="hyperbolic", curvature=1.0)
Event2Vec(num_event_types=len(vocab), geometry="stereographic", kappa=-1.0)
Event2Vec(num_event_types=len(vocab), geometry="rotation", rotation_n=3)
Event2Vec(num_event_types=len(vocab), geometry="product", product_kappas=(-1.0, 1.0))
```

Hyperbolic embeddings live in the Poincaré ball; use the matching operations on them:

```python
from event2vector import HyperbolicUtils

hyp = Event2Vec(num_event_types=len(vocab), geometry="hyperbolic", curvature=1.0,
                pad_sequences=True, num_epochs=20).fit(sequences)
z = hyp.transform(sequences, as_numpy=False)
combined = HyperbolicUtils.mobius_add(z[0:1], z[1:2], hyp.curvature)
distance = HyperbolicUtils.poincare_dist_sq(z[0:1], z[1:2], hyp.curvature).sqrt()
```

### Affine geometry

In the additive geometries an event always pushes the state the same way. With
`geometry="affine"` each event is an invertible affine map `x ↦ exp(M_e) x + v_e`,
so the same event contributes differently depending on what came before, while
the model remains a group:

- with every `M_e = 0` (the initialization) the result is exactly the Euclidean sum,
  so training starts from the additive model;
- every step is invertible, and the reconstruction loss recovers it exactly;
- any sub-sequence composes into a single affine map, so its effect on *any*
  history is one `A x + v`.

The state is the whole map `[vec(A), b]`; the point to plot is `b`, the image of the
origin. The singular values of a sub-sequence's `A` measure how strongly it
overrides what came before: `1` passes the history through, values below `1`
erase it along that direction.

```python
aff = Event2Vec(num_event_types=len(vocab), geometry="affine", embedding_dim=2,
                pad_sequences=True, num_epochs=20).fit(sequences)

states = aff.transform(sequences, as_numpy=False)   # [vec(A), b] per sequence
points = aff.model.point(states)                    # (n_sequences, 2), ready to plot

A, v = aff.model.compose([1, 2, 3])                 # one map for a sub-sequence
print(aff.model.override_strength([1, 2, 3]))       # its singular values
```

`affine_readout="full"` lets the decoder read `A` as well as `b`, and
`affine_max_log_scale` bounds how strongly a single event can stretch or shrink
the state.

## API reference

### `Event2Vec` parameters

| parameter | default | meaning |
|---|---|---|
| `num_event_types` | required | size of the event vocabulary |
| `embedding_dim` | `64` | dimension of the event vectors (the plotted point for `affine`) |
| `geometry` | `"euclidean"` | see [Geometries](#geometries) |
| `curvature` | `1.0` | curvature `c > 0` for `hyperbolic` |
| `kappa` | `None` | signed curvature for `stereographic` / `spherical` |
| `rotation_n` | `3` | `n` of `SO(n)` for `rotation` |
| `heisenberg_m` | `1` | the state is the `(2m+1)`-dimensional Heisenberg group |
| `affine_readout` | `"point"` | `"point"` or `"full"`: what the decoder reads for `affine` |
| `affine_max_log_scale` | `None` | bound on how much one event can stretch the `affine` state |
| `product_kappas` | `(-1.0, 1.0)` | one curvature per factor for `product` |
| `dropout_p` | `0.2` | dropout on each step's increment |
| `max_norm` | `10.0` | norm clipping of the Euclidean state; `None` disables it, so the state is the exact sum of its events |
| `lambda_reconstruction` | `0.2` | weight of the reconstruction loss |
| `lambda_consistency` | `0.2` | weight of the consistency loss |
| `lambda_difference` | `0.0` | weight of the difference-reconstruction loss (Euclidean, needs `pad_sequences=True`): the difference of two histories' states must decode to the events that differ |
| `learning_rate` | `1e-3` | AdamW learning rate |
| `batch_size` | `32` | sequences per batch |
| `num_epochs` | `50` | training epochs |
| `pad_sequences` | `False` | vectorized padded batches (much faster on large corpora) |
| `use_gpu` / `device` | `True` / `None` | automatic CUDA / MPS / CPU choice, or an explicit device |
| `random_state` | `None` | seed for reproducible runs |
| `deterministic` | `False` | request deterministic kernels |
| `riemannian` | `False` | Riemannian Adam for `hyperbolic` (needs `geoopt`) |

### Methods

| method | returns |
|---|---|
| `fit(sequences, epochs=None, verbose=False)` | the fitted estimator; per-epoch losses in `training_loss_` |
| `transform(sequences, as_numpy=True)` | one state per sequence (NumPy array or tensor) |
| `fit_transform(sequences, ...)` | `fit`, then `transform` on the same sequences |
| `most_similar(positive, negative=None, topn=10)` | nearest event ids to events or whole sequences, with similarities |
| `attribute(sequence, target=None)` | each event's contribution to the logit of the next event `target` |
| `counterfactual(sequence, remove)` | next-event probabilities with and without the events at positions `remove` |
| `influence_matrix()` | `A[j, i]`: how much event `i` alone raises the logit of next event `j` |
| `event_impact()` | how strongly one occurrence of each event moves the next-event distribution |
| `decompose(sequence_or_state)` | event counts recovered from a state vector (Euclidean) |
| `explain_difference(sequence_a, sequence_b)` | signed counts of the events that differ, read from the two states (Euclidean) |

The underlying PyTorch module is `model.model` (event vectors in
`model.model.embedding.weight`).

### Root-cause analysis

In the Euclidean geometry the state is an exact weighted sum of the event
embeddings, `h_T = Σ_i w_i e_{s_i}`, and the decoder is linear. The weights `w_i`
are the products of the later norm-clipping factors (all `1` with
`max_norm=None`), so recent events weigh more once clipping fires. Each prediction
therefore splits exactly into per-event terms, which the methods above expose:

```python
est = Event2Vec(num_event_types=len(vocab), pad_sequences=True).fit(sequences)

out = est.attribute(history, target=vocab["job_loss"])
out["bias"] + out["contributions"].sum()   # equals out["logit"]: one term per event

est.counterfactual(history, remove=[2])["delta"]   # change in next-event probabilities
A = est.influence_matrix()                         # (V, V) event -> next-event influence
est.event_impact()                                 # one score per event
est.decompose(history, round_counts=True)          # which events a state contains
est.explain_difference(failing_run, healthy_run)   # +: more in the failing run
```

| method | geometries | exact when |
|---|---|---|
| `attribute` | all | Euclidean: always (the terms sum to the logit). Others: leave-one-out differences, which do not sum to the logit |
| `counterfactual` | all | always (the edited sequence is re-encoded) |
| `influence_matrix`, `event_impact` | all | Euclidean with `max_norm=None`: `A = W_dec Eᵀ` and every sequence's logits are `bias + A @ counts` |
| `decompose`, `explain_difference` | Euclidean | `max_norm=None` and few enough distinct events for the embeddings to separate them; with clipping they estimate the clip-weighted counts |

`event2vector.metrics.transition_graph_auc(est, graph, vocab)` scores
`influence_matrix()` against a known transition graph (ROC AUC over all ordered
event pairs), e.g. `make_lifepath()["graph"]`.

These methods explain the model's predictions. They reflect associations learned
from the data, not causal effects, and the sum-based ones cannot tell event
orders apart when clipping does not fire.

### Devices

By default the estimator uses CUDA, then Apple MPS, then the CPU. On Apple
Silicon the `rotation` and `affine` geometries default to the CPU, because MPS has
no `matrix_exp` kernel and the CPU is faster for them; pass `device="mps"` to
override. Use `use_gpu=False` to force the CPU, or `device="cuda:0"` to pick a
device.

## Datasets, metrics and baselines

The package ships domains with *exact ground truth* for the additive structure,
so interpretability can be measured rather than only visualized. Datasets live in
`event2vector.datasets` and return a uniform record (`sequences`,
`processed_sequences`, `vocab`, `labels`, `graph`):

- `make_decay()`: radioactive decay chains; each event is an exact `(ΔZ, ΔN)`, so
  additivity coincides with conservation laws (metric: `operator_recovery`).
- `make_group(name)`: Cayley-graph walks on abelian and non-abelian groups; the
  additive endpoint provably equals the abelianization (metric: `abelianization_gap`).
- `make_lifepath()`, `load_brown()`: a controlled synthetic workbench and real
  part-of-speech sequences.
- `load_gdelt()`, `load_sec_8k()`, `load_synthea()`, …: real-world event streams
  (each needs a public data download; see the loader's docstring).

Metrics are in `event2vector.metrics`, a seeded evaluation harness in
`event2vector.eval`, and GRU / LSTM / Transformer / Word2Vec baselines in
`event2vector.baselines`.

## Examples

A minimal end-to-end run (loss, embeddings, nearest events):

```bash
python3 examples/minimal_example.py
```

Part-of-speech sequences from the Brown corpus (the scripts need a few extra
packages: NLTK for the corpus, t-SNE and Word2Vec for the comparison plots):

```bash
pip install "event2vector[viz,manifold,nlp]" nltk
python3 -m examples.prepare_brown_data
python3 -m examples.train_brown_data
python3 -m examples.visualize_brown_corpus
```

Applications:

- [The Geometry of Groceries](https://sulcantonin.substack.com/p/the-geometry-of-groceries)
- [The Geometry of Language Families](https://sulcantonin.substack.com/p/the-geometry-of-language-families)

## Development

```bash
pip install -e . pytest
pytest tests/ -q
```

Changes are listed in [CHANGELOG.md](https://github.com/sulcantonin/event2vec_public/blob/main/CHANGELOG.md).
Bug reports and questions are welcome in the
[issue tracker](https://github.com/sulcantonin/event2vec_public/issues).

## Citation

If you use Event2Vector, please cite the paper, published in the
[Proceedings of Machine Learning Research](https://proceedings.mlr.press/v282/sulc26a.html) (vol. 282, NeurIPS Workshops
on Symmetry and Geometry in Neural Representations):

```bibtex
@InProceedings{pmlr-v282-sulc26a,
  title     = {Event2Vec: A Geometric Approach to Learning Composable Representations of Event Sequences},
  author    = {Sulc, Antonin},
  booktitle = {Proceedings of the 4th (2025) and 3rd (2024) NeurIPS Workshops on Symmetry and Geometry in Neural Representations},
  pages     = {492--512},
  year      = {2026},
  editor    = {Acosta, Francisco and Azeglio, Simone and Tolooshams, Bahareh and van de Geijn, Chase and Shewmake, Christian and Sanborn, Sophia and Miolane, Nina},
  volume    = {282},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v282/sulc26a.html}
}
```

A preprint is on [arXiv:2509.12188](https://arxiv.org/abs/2509.12188).

To cite the software itself, use its Zenodo DOI
[10.5281/zenodo.23399003](https://doi.org/10.5281/zenodo.23399003), which always
resolves to the latest version (each release also has its own DOI). GitHub's
"Cite this repository" button reads the same metadata from `CITATION.cff`.

## License

MIT. See the license badge above.
