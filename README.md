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


## Brown Corpus POS tagging example
After installation, you can try to run Brown Part-of-Speech tagging example from the paper. 

```bash
python3 -m experiments.prepare_brown_data.py
python3 -m experiments.train_brown_data.py
python3 -m experiments.visualize_brown_corpus.py
```

## Quickstart (tiny synthetic dataset)

The snippet below trains the high-level `Event2Vec` estimator on a toy event graph for a few epochs and prints an embedding for a short sequence. It runs in seconds on CPU and mirrors the Euclidean model from the paper while exposing the new fit/transform interface.

We have 5 events: `START, A, B, C, END` and we test if we add `START + A + C ~ C`. The example remains self-contained so you can inspect the additive loss function in isolation.

```python
import random
import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reproducibility: fix all relevant seeds so results are stable across runs
SEED = 42
random.seed(SEED)                    # Python's RNG (used by get_sequences and shuffling)
np.random.seed(SEED)                 # NumPy RNG (used by PCA and any NumPy ops)
torch.manual_seed(SEED)              # Torch CPU RNG
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED) # Torch CUDA RNG (all devices)

# Optional: prefer deterministic behavior in cuDNN (may have performance impact)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Expected results (for the tiny toy graph START→A/B→C→END):
# - Training loss should decrease over epochs (with minor wobble due to randomness/dropout).
# - The printed sequence embedding for [START, A, C] is a single 8-D vector (shape (1, 8)).
# - Nearest tokens by cosine similarity to that sequence embedding should rank 'C' highest
#   (since the encoder is additive and the last token is C), with 'END' somewhat aligned.
# - The decoder's top-1 next-event probability from that state should be 'END' with high
#   confidence (≈0.9+) because the toy transitions force C→END.
# Visualization (below): a 2D PCA of token embeddings plus the sequence embedding should show
# 'SEQ(START-A-C)' lying near the point labeled 'C'.

from event2vector import Event2Vec
from event2vector.data import get_sequences

# 1) Define a tiny state-transition toy dataset
#    We model a simple Markovian process: START → (A or B) → C → END
#    The model will learn to predict the next token and produce a sequence embedding
#    by additive composition of token embeddings.
event_types = ['START', 'A', 'B', 'C', 'END']
event_transitions = {
    'START': [('A', 0.5), ('B', 0.5)],
    'A': [('C', 0.6), ('B', 0.4)],
    'B': [('C', 0.7), ('A', 0.3)],
    'C': [('END', 1.0)],
}

# 2) Generate synthetic sequences (reproducible thanks to the fixed seeds above)
#    get_sequences returns: raw sequences, tensorized sequences, and vocab mappings.
_, processed_sequences, event_2_idx, _ = get_sequences(
    event_types=event_types,
    event_transitions=event_transitions,
    initial='START',
    terminal='END',
    num_seq=200,            # generate 200 short sequences
    max_seq_length=6,       # keep them small for speed
    generate_new=True,
    prefix='tiny_quickstart'
)
inv = {v:k for k,v in event_2_idx.items()}

estimator = Event2Vec(
    num_event_types=len(event_types),
    embedding_dim=8,
    dropout_p=0.1,
    learning_rate=5e-3,
    lambda_reconstruction=0.2,
    lambda_consistency=0.2,
    batch_size=32,
    num_epochs=128,
    pad_sequences=True,   # enables padded batch processing for speed
    use_gpu=True,         # set to False to force CPU even when CUDA/MPS is present
)
estimator.fit(processed_sequences, verbose=True)
torch_model = estimator.model

# 4) Use the learned representation for a short sequence via transform()
seq = torch.tensor([
    event_2_idx['START'], event_2_idx['A'], event_2_idx['C']
], dtype=torch.long)
sequence_embedding = estimator.transform([seq], as_numpy=False)[0]
print('Sequence embedding (START-A-C):', sequence_embedding.numpy())

# 5) Nearest tokens with the built-in gensim-style helper
nearest = estimator.most_similar(positive=seq, topn=3)
print('Nearest tokens by cosine:', [(inv[i], score) for i, score in nearest])

with torch.no_grad():
    # 6) Next-event distribution from the current state
    seq_for_decoder = sequence_embedding.to(estimator.device)
    logits = torch_model.decoder(seq_for_decoder.unsqueeze(0))    # [1, V]
    probs = torch.softmax(logits, dim=-1).squeeze(0)
    top = torch.topk(probs, k=3)
    print('Top-3 next events:', [ (inv[i.item()], float(probs[i])) for i in top.indices ])

# 7) Visualization: PCA of token embeddings + sequence embedding
with torch.no_grad():
    token_emb = torch_model.embedding.weight.detach().cpu().numpy()   # [V, 8]
    seq_emb = sequence_embedding.detach().cpu().numpy()               # [1, 8]
    X = np.vstack([token_emb, seq_emb])
    pca = PCA(n_components=2, random_state=SEED)
    X2 = pca.fit_transform(X)
    tokens2, seq2 = X2[:-1], X2[-1]

plt.figure(figsize=(6, 6))
plt.scatter(tokens2[:, 0], tokens2[:, 1], c='gray', label='tokens')
for i, (x, y) in enumerate(tokens2):
    plt.text(x + 0.02, y + 0.02, inv[i], fontsize=9)
plt.scatter([seq2[0]], [seq2[1]], c='red', marker='*', s=160, label='SEQ(START-A-C)')
plt.title('PCA: token embeddings + sequence embedding (expect SEQ near C)')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
```

## Estimator API

The `Event2Vec` class mirrors scikit-learn transformers so it can slot into existing NLP pipelines:

```python
from event2vector import Event2Vec

model = Event2Vec(
    num_event_types=len(vocab),
    geometry="euclidean",          # or "hyperbolic"
    embedding_dim=128,
    pad_sequences=True,            # mini-batch speed-up
    num_epochs=50,
)
model.fit(train_sequences, verbose=True)
train_embeddings = model.transform(train_sequences)         # numpy array
test_embeddings = model.transform(test_sequences, as_numpy=False)  # PyTorch tensor
```

Key methods:
- `fit`: optimizes embeddings with the additive loss from the paper.
- `fit_transform`: convenience helper returning the encoded sequences after fitting.
- `transform`: freezes weights and encodes arbitrary sequences, optionally returning PyTorch tensors for downstream models.
- `most_similar`: gensim-style nearest-neighbor lookup over learned event embeddings using tokens or full sequences as queries.
- `pad_sequences=True`: enables fully vectorized batches with masking for substantial throughput gains on large corpora.

Device control: set `use_gpu=False` to force CPU even if CUDA/MPS is present, or pass an explicit `device` (e.g., `"cuda:0"` or `"cpu"`).


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
