<div align="center">

# Event2Vector (event2vec)
## A Geometric Approach to Learning Composable Representations of Event Sequences

[![PyPI version](https://badge.fury.io/py/event2vector.svg)](https://badge.fury.io/py/event2vector)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2509.12188-b31b1b.svg)](https://arxiv.org/abs/2509.12188)

# ![Teaser](./images/teaser.png)

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

Key methods:
- `fit`: optimizes embeddings with the additive loss from the paper.
- `fit_transform`: convenience helper returning the encoded sequences after fitting.
- `transform`: freezes weights and encodes arbitrary sequences, optionally returning PyTorch tensors for downstream models.
- `most_similar`: gensim-style nearest-neighbor lookup over learned event embeddings using tokens or full sequences as queries.
- `pad_sequences=True`: enables fully vectorized batches with masking for substantial throughput gains on large corpora.

Device control: set `use_gpu=False` to force CPU even if CUDA/MPS is present, or pass an explicit `device` (e.g., `"cuda:0"` or `"cpu"`).



## Brown Corpus POS tagging example
After installation, you can try to run Brown Part-of-Speech tagging example from the paper. 

```bash
python3 -m experiments.prepare_brown_data.py
python3 -m experiments.train_brown_data.py
python3 -m experiments.visualize_brown_corpus.py
```


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
