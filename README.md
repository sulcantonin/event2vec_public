<div align="center">

# Event2Vector (event2vec)
## A Geometric Approach to Learning Composable Representations of Event Sequences

[![PyPI version](https://badge.fury.io/py/event2vector.svg)](https://badge.fury.io/py/event2vector)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2509.12188-b31b1b.svg)](https://arxiv.org/abs/2509.12188)

![](https://github.com/sulcantonin/event2vec_public/blob/main/images/teaser.png)

</div>

## Overview

**Event2Vector** is a framework for learning representations of discrete event sequences. Inspired by the geometric structures found in neural representations, this model uses a simple, additive recurrent structure to create composable and interpretable embeddings.

## Key Concepts
* **Linear Additive Hypothesis**: The core idea behind Event2Vector is that the representation of an event sequence can be modeled as the vector sum of the embeddings of its individual events. This allows for intuitive vector arithmetic, enabling the composition and decomposition of event trajectories.
* **Euclidean and Hyperbolic Models**: Event2Vector is offered in two geometric variants:
    * **Euclidean model**: Uses standard vector addition, providing a straightforward, flat geometry for event trajectories.
    * **Hyperbolic model**: Employs Möbius addition, which is better suited for hierarchical data structures, as it can embed tree-like patterns with less distortion.
* **Estimator API**: A scikit-learn style `Event2Vec` estimator exposes `fit`, `fit_transform`, and `transform`, enabling drop-in use inside pipelines while keeping the compositional recurrent loss from the paper.
* **Padded batching**: Optional padding allows entire minibatches of variable-length sequences to be processed in parallel, significantly accelerating training on large corpora without changing model behavior.

For more details, check *Sulc A., Event2Vector: A Geometric Approach to Learning Composable Representations of Event Sequences*

## Example Applications
* Substack Post: Geometry of Groceries https://sulcantonin.substack.com/p/the-geometry-of-groceries
* Substack Post: The Geometry of Language Families https://sulcantonin.substack.com/p/the-geometry-of-language-families

## Installation

Install the package directly from PyPI:

```bash
pip install event2vector
```

Or install from source:

```bash
git clone https://github.com/sulcantonin/event2vec_public.git
cd event2vec_public
pip install .
```

## Estimator API

The `Event2Vec` class mirrors scikit-learn transformers so it can slot into existing NLP pipelines:

```python
from event2vector import Event2Vec

model = Event2Vec(
    num_event_types=len(vocab),
    geometry="euclidean",
    embedding_dim=128,
    pad_sequences=True,
    num_epochs=50,
)
model.fit(train_sequences, verbose=True)
train_embeddings = model.transform(train_sequences) 
```

Hyperbolic variant (training + using trained weights):

```python
from event2vector import Event2Vec, HyperbolicUtils
import torch

hyp_model = Event2Vec(
    num_event_types=len(vocab),
    geometry="hyperbolic",
    curvature=1.0,
    embedding_dim=128,
    pad_sequences=True,
    num_epochs=50,
)
hyp_model.fit(train_sequences, verbose=True)

# Use the trained weights: encode sequences and query the decoder
seq_embeddings = hyp_model.transform(test_sequences, as_numpy=False)
torch_model = hyp_model.model

# Hyperbolic addition + distance between two datapoints (Poincaré ball)
u = seq_embeddings[0]
v = seq_embeddings[1]
uv_added = HyperbolicUtils.mobius_add(u, v, hyp_model.curvature)
uv_dist = HyperbolicUtils.poincare_dist_sq(u, v, hyp_model.curvature).sqrt()
```

Key methods:
- `fit`: optimizes embeddings with the additive loss from the paper.
- `fit_transform`: convenience helper returning the encoded sequences after fitting.
- `transform`: freezes weights and encodes arbitrary sequences, optionally returning PyTorch tensors for downstream models.
- `most_similar`: gensim-style nearest-neighbor lookup over learned event embeddings using tokens or full sequences as queries.
- `pad_sequences=True`: enables fully vectorized batches with masking for substantial throughput gains on large corpora.

Device control: set `use_gpu=False` to force CPU even if CUDA/MPS is present, or pass an explicit `device` (e.g., `"cuda:0"` or `"cpu"`). On Apple Silicon the `rotation` and `affine` geometries default to the CPU, because MPS has no `matrix_exp` kernel and the CPU is faster for them; pass `device="mps"` to override.

## Geometries (v0.2)

Beyond `euclidean` and `hyperbolic`, the estimator exposes the paper's pilot geometries as first-class `geometry=` backbones, each the natural home of a structure the additive endpoint cannot carry. They share the same estimator and a generic gyrogroup/Lie-group left-cancellation reconstruction loss.

| `geometry` | update | best for |
|---|---|---|
| `euclidean` | vector addition | order-invariant bag of events, exact decomposition |
| `hyperbolic` | Möbius addition (curvature `c>0`) | hierarchical / tree-like structure |
| `stereographic` | κ-gyroaddition (signed `kappa`) | unified curvature: `kappa<0` hyperbolic, `0` Euclidean, `>0` spherical |
| `spherical` | κ-gyroaddition (`kappa>0`) | cyclic / modular (periodic) structure |
| `rotation` | left `SO(n)` product | non-commutative orientation (order matters) |
| `heisenberg` | Heisenberg group product | order-dependent signed (Lévy) area |
| `affine` | `x -> exp(M_e) x + v_e` on the left | events that act on the history (shrink, stretch, rotate it), not only add to it |
| `product` | per-factor update | two structure types at once (e.g. hyperbolic × spherical) |

```python
from event2vector import Event2Vec

# one unified curvature dial: kappa<0 hyperbolic, 0 Euclidean, >0 spherical
model = Event2Vec(num_event_types=len(vocab), geometry="stereographic", kappa=-1.0)

# non-commutative orientation (SO(3))
model = Event2Vec(num_event_types=len(vocab), geometry="rotation", rotation_n=3)

# product of a hyperbolic and a spherical factor
model = Event2Vec(num_event_types=len(vocab), geometry="product", product_kappas=(-1.0, 1.0))
```

### Affine geometry (v0.4)

In the additive geometries an event always pushes the same way. With
`geometry="affine"` each event is an invertible affine map
`x -> exp(M_e) x + v_e`, so the same event contributes differently depending on
what came before, while the model stays a group:

- with every `M_e = 0` (the initialisation) the point is exactly the Euclidean sum,
  so training starts from the additive model;
- every step is invertible and the reconstruction loss recovers it exactly;
- any subsequence composes into a single affine map, so its effect on *any*
  history is one `A x + v`.

The state is the whole map `[vec(A), b]`; the point to plot is `b`, the image of
the origin (`model.model.point(...)`). The singular values of a subsequence's `A`
say how much it overrides what came before: `1` passes the history through,
`< 1` erases it along that direction.

```python
model = Event2Vec(num_event_types=len(vocab), geometry="affine", embedding_dim=2)
model.fit(sequences)

states = model.transform(sequences, as_numpy=False)  # [vec(A), b] per sequence
points = model.model.point(states)                   # (n, 2) points to plot

A, v = model.model.compose([3, 7, 1])                # one map for a subsequence
print(model.model.override_strength([3, 7, 1]))      # its singular values
```

Options: `affine_readout="point"` (default; the decoder reads `b`) or `"full"`
(it also reads `A`), and `affine_max_log_scale` to bound how strongly a single
event can stretch or shrink the state.

Reproduce the curvature sweep on a real knowledge graph (auto-downloads WN18RR):

```bash
python3 -m scripts.run_kg_curvature   # next-relation accuracy vs kappa
```

## Brown Corpus POS tagging example
After installation, you can try to run Brown Part-of-Speech tagging example from the paper. 

```bash
python3 -m examples.prepare_brown_data
python3 -m examples.train_brown_data
python3 -m examples.visualize_brown_corpus
```


## Minimal example script

The repository includes a runnable minimal example that trains a tiny model end-to-end and prints example outputs (loss, embeddings, and nearest tokens). Run it from the repo root:

```bash
python3 examples/minimal_example.py
```

To try a hyperbolic run, open `examples/minimal_example.py` and set `geometry="hyperbolic"` in the `Event2Vec` constructor, then rerun the script.

## Reproducing the paper experiments

The journal experiments are built around domains with *exact ground truth* for the
additive structure, so interpretability is measured rather than visualized:

```bash
# Physics (nuclear decay) + Math (group sequences): exact-ground-truth centerpiece
python3 -m scripts.run_centerpiece --seeds 5 --epochs 20 --out results/centerpiece

# Synthetic life-path workbench (order-sensitivity-vs-curvature, additivity decay)
python3 -m scripts.run_lifepath --seeds 5 --out results/lifepath
```

Datasets live in `event2vector/datasets/` and all return a uniform record
(`sequences`, `processed_sequences`, `vocab`, `labels`, `graph`):

- `make_decay()`: radioactive decay chains; each event is an exact `(ΔZ, ΔN)` so
  additivity coincides with conservation laws (metric: `operator_recovery`).
- `make_group(name)`: Cayley-graph walks on abelian/non-abelian groups; the
  additive endpoint provably equals the abelianization (metric: `abelianization_gap`).
- `make_lifepath()`, `load_brown()`: controlled synthetic + real (POS) workbench.
- `load_gdelt()`, `load_sec_8k()`, `load_synthea()`, ...: real-world event streams
  (require a public data download; see each loader's docstring).

Metrics (`event2vector.metrics`) and the seeded harness (`event2vector.eval`) are
reusable across datasets; baselines (GRU/LSTM/Transformer/Word2Vec) are in
`event2vector.baselines`. Run the test suite with `pytest tests/ -q`.

## References
For citations please use following Bibtex. 
```bibtex
@article{sulc2025event2vec,
  title={Event2Vec: A Geometric Approach to Learning Composable Representations of Event Sequences},
  author={Sulc, Antonin},
  journal={arXiv preprint arXiv:2509.12188},
  year={2025}
}
```
