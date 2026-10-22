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

**[event2vec.com](https://event2vec.com)** · [Visual guide](https://github.com/sulcantonin/event2vec_public/blob/main/docs/visual-guide.md) · [Paper](https://proceedings.mlr.press/v282/sulc26a.html) · [Changelog](https://github.com/sulcantonin/event2vec_public/blob/main/CHANGELOG.md) · [Issues](https://github.com/sulcantonin/event2vec_public/issues)

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

The [visual guide](https://github.com/sulcantonin/event2vec_public/blob/main/docs/visual-guide.md) walks through all of this picture by picture: histories
as sums, prediction, exact explanations, counterfactuals, gates and geometries.

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
| reconstruction | `lambda_reconstruction` | the step `e_t` can be recovered from `h_{t-1}` and `h_t`, so the update stays invertible |
| consistency | `lambda_consistency` | two dropout passes over the same step agree |

With `gate="vector"` (Euclidean only), each event may also rescale the history
before its vector is added, `h_t = (1 + g_e) * h_{t-1} + e_t`. The gate starts at
zero, so training starts from the exact sum, and the reconstruction term becomes
`‖g_e * h_{t-1}‖²`: the price the model pays for departing from pure addition.

There is no attention and no hidden nonlinearity in the state update, which is
what keeps every embedding a readable composition of its events. See the
[paper](https://proceedings.mlr.press/v282/sulc26a.html) for details.

## Geometries

Choose the composition with `geometry=`. Every geometry uses the same estimator
and a generic left-cancellation reconstruction loss.

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
| `max_norm` | `10.0` | norm clipping of the Euclidean state |
| `gate` | `None` | Euclidean only: `"vector"` or `"tied"` lets each event rescale the history, `h_t = (1 + g_e) * h_{t-1} + e_t` |
| `lambda_reconstruction` | `0.2` | weight of the reconstruction loss |
| `lambda_consistency` | `0.2` | weight of the consistency loss |
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
| `attribute(sequence, target=None)` | each earlier event's contribution to the logit of `target`, per position and per event id; exact for Euclidean |
| `rank_causes(history, effect)` | the distinct events of `history` ranked by their contribution to `effect` |
| `counterfactual(sequence, remove=, replace=, insert=)` | next-event distribution before and after editing the history |
| `event_gates()` | the learned gate of every event (needs `gate`) |

The underlying PyTorch module is `model.model` (event vectors in
`model.model.embedding.weight`).

### Explaining predictions

```python
import numpy as np
from event2vector import Event2Vec

sequences = [[0, 3, 7, 5], [3, 1, 7, 5], [0, 1, 2, 4], [3, 7, 2, 5]] * 50
model = Event2Vec(num_event_types=8, gate="vector", pad_sequences=True,
                  num_epochs=20).fit(sequences)

history, effect = [3, 7, 1, 7], 5
model.rank_causes(history, effect)     # [(event_id, score), ...], highest first
out = model.attribute(history, target=effect)
np.isclose(out["baseline"] + out["contributions"].sum(), out["logit"])  # True
model.counterfactual(history, remove=[1])["delta"][effect]  # change in P(effect)
model.event_gates().mean(axis=1)       # per event: < 0 shrinks the history
```

The contributions explain the model's predictions. They are associations learned
from the data, not causal effects.

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

## FAQ

**How is Event2Vec different from Word2Vec?** Both learn one vector per event, but
from different signals. Word2Vec learns from co-occurrence in a symmetric context
window: two events are similar when they appear together. Event2Vec learns from
next-event prediction: a history must predict the event that follows it, so each
event's vector is pulled towards the events that follow it. This has three
consequences:

- Its vectors encode what comes next, not what appears together. On synthetic health
  records (Synthea), the similarity of Event2Vec's vectors follows the similarity of
  the events' successors (Spearman 0.79) far more than their co-occurrence (0.32);
  for Word2Vec it is the reverse (0.30 and 0.46).
- It represents whole histories and predicts their next event. Word2Vec has neither a
  history state nor a predictive model.
- Its explanations have a direction. Asked which earlier condition caused a
  complication, Event2Vec's attribution ranks the true cause first in 84% of Synthea
  cases and Word2Vec's similarity in 52%: a complication is more similar to other
  complications than to the diabetes that causes them.

Word2Vec is the right tool when co-occurrence is the question: on intrusion-detection
alerts (AIT-ADS), where the relevant alerts are defined by occurring together, it
ranks them perfectly.

**How is Event2Vec different from an RNN?** It is a recurrent model too, but its
transition is restricted to a fixed operation selected by the event: adding the
event's vector in the Euclidean case, optionally after a per-event rescaling of the
history (`gate`), or a group product in the other geometries. An RNN such as a GRU
or LSTM learns a nonlinear function of the state and the input, with gates that
depend on the state. Two things follow:

- Event2Vec's state is a closed-form expression of its events, so histories compose
  exactly and every prediction splits exactly into one term per event. An RNN can
  only be explained after the fact, by methods that approximate this split. In our
  comparison, Event2Vec's contributions add up to the logit within 1e-6, while
  integrated gradients and gradient times input on GRU and LSTM models miss by 0.09
  to 11. On Synthea, Event2Vec ranks the known cause first more often (0.84 against
  at most 0.73 for the RNN explanations) at a similar next-event loss.
- The restriction costs expressiveness. On the AIT-ADS alert streams, where order
  matters more, a GRU predicts markedly better, and its integrated-gradients
  explanations rank the attack precursors slightly better (0.95 against 0.92).

So Event2Vec is the better choice when exact, per-event explanations matter, and an
RNN when raw predictive accuracy on strongly order-dependent data does. The numbers
come from one training run of each model.

**Why does additivity hold?** In the Euclidean model, `h_t = h_{t-1} + e_t` by
construction, so with no clipping the state is exactly the sum of its event vectors
and `h(A followed by B) = h(A) + h(B)`. Additivity is a property of the
architecture, not something training has to discover. Training decides what the
event vectors are, so that their sums predict what comes next. With
`gate="vector"`, composition becomes `h(A followed by B) = G(B) * h(A) + h(B)`,
where `G(B)` is the product of the gates `1 + g_e` of B's events. It stays exact,
and the reconstruction term keeps the gates near zero unless the data needs them.

**Why are the explanations exact?** The state is a weighted sum of event vectors,
each scaled by the gates of the events after it and by norm clipping, and the
decoder is linear. So a logit splits into one term per event plus the decoder
bias, with nothing left over. Gradient-based explanations of an RNN only
approximate this split. For the curved geometries, `attribute` uses the change in
the logit at each step, which also sums exactly but depends on the order of events.

**What does the reduced expressiveness cost?** The plain Euclidean state cannot tell
apart two histories with the same events in a different order. Predictions still use
order, because the model predicts after every prefix, but the final state does not.
The gate, or a non-Euclidean geometry, restores order sensitivity.

**When should I use the gate?** When some events should reset or fade the past, as
an acute illness, a sentence boundary or a burst of repeated alerts might.
`event_gates()` then shows which events forget. The plain sum is the default
because it keeps every guarantee above.

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
