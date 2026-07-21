"""Baseline sequence encoders with the same fit/transform interface as the
``Event2Vec`` estimator, so the experiment metrics can treat every model
uniformly.

- ``RNNEncoder`` (GRU/LSTM) and ``TransformerEncoder``: order-sensitive encoders
  trained as next-event predictors; ``transform`` returns the pooled hidden
  state. These are the order-sensitive reference points for A2 / E1.
- ``Word2VecBagEncoder``: an order-blind bag-of-tokens baseline (mean of learned
  token vectors), the Word2Vec comparison used throughout the paper.

All encoders expose ``n_params`` so comparisons can be framed per-parameter
rather than as a leaderboard.
"""
import random
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


def _seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def _device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if use_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _standardize(sequences) -> List[torch.Tensor]:
    out = []
    for s in sequences:
        t = s.detach().long().cpu() if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.long)
        out.append(t)
    return out


class _NextEventEncoder(nn.Module):
    """Shared next-event-prediction training loop for neural baselines."""

    def fit(self, sequences, epochs=None, verbose=False):
        seqs = [s for s in _standardize(sequences) if s.numel() >= 2]
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        ce = nn.CrossEntropyLoss(ignore_index=self.pad_value)
        self.train()
        total_epochs = epochs or self.num_epochs
        rng = random.Random(self.random_state)
        for ep in range(total_epochs):
            rng.shuffle(seqs)
            running = 0.0
            nb = 0
            for i in range(0, len(seqs), self.batch_size):
                batch = seqs[i : i + self.batch_size]
                tens = [b.to(self.device) for b in batch]
                padded = pad_sequence(tens, batch_first=True, padding_value=self.pad_value)
                inp, tgt = padded[:, :-1], padded[:, 1:]
                logits = self._logits(inp)
                loss = ce(logits.reshape(-1, self.num_event_types), tgt.reshape(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += float(loss.item())
                nb += 1
            if verbose:
                print(f"epoch {ep + 1}/{total_epochs} loss {running / max(1, nb):.4f}")
        self.eval()
        return self

    @torch.no_grad()
    def transform(self, sequences, as_numpy=True, pad_sequences=None):
        self.eval()
        seqs = _standardize(sequences)
        outs = []
        for i in range(0, len(seqs), self.batch_size):
            batch = seqs[i : i + self.batch_size]
            tens = [b.to(self.device) for b in batch]
            lengths = torch.tensor([t.size(0) for t in tens], device=self.device)
            padded = pad_sequence(tens, batch_first=True, padding_value=self.pad_value)
            pooled = self._pool(padded, lengths)
            outs.append(pooled.cpu())
        emb = torch.cat(outs, dim=0)
        return emb.numpy() if as_numpy else emb

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class RNNEncoder(_NextEventEncoder):
    def __init__(self, num_event_types, embedding_dim=32, kind="gru", num_layers=1,
                 learning_rate=1e-3, batch_size=64, num_epochs=12, pad_value=None,
                 random_state=None, use_gpu=True):
        super().__init__()
        self.num_event_types = num_event_types
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        # A dedicated pad id OUTSIDE the real vocabulary. Using 0 collides with a genuine
        # token (e.g. alpha-decay is 18% of the decay corpus, the first group generator is
        # 33% of S4 words); that token would then be frozen at zero, excluded from the loss,
        # and -- for the Transformer -- dropped from attention and pooling.
        self.pad_value = num_event_types if pad_value is None else pad_value
        self.random_state = random_state
        self.geometry = "euclidean"  # for distance_for() in metrics
        self.curvature = 1.0
        _seed(random_state)
        self.device = _device(use_gpu)
        self.embedding = nn.Embedding(num_event_types + 1, embedding_dim,
                                      padding_idx=self.pad_value)
        rnn_cls = nn.LSTM if kind.lower() == "lstm" else nn.GRU
        self.rnn = rnn_cls(embedding_dim, embedding_dim, num_layers=num_layers, batch_first=True)
        self.dec = nn.Linear(embedding_dim, num_event_types)
        self.kind = kind.lower()
        self.to(self.device)

    def _logits(self, inp):
        out, _ = self.rnn(self.embedding(inp))
        return self.dec(out)

    def _pool(self, padded, lengths):
        out, _ = self.rnn(self.embedding(padded))
        last = (lengths - 1).clamp_min(0)
        return out[torch.arange(out.size(0), device=out.device), last]


class TransformerEncoder(_NextEventEncoder):
    def __init__(self, num_event_types, embedding_dim=32, nhead=4, num_layers=2,
                 dim_feedforward=64, learning_rate=1e-3, batch_size=64, num_epochs=12,
                 pad_value=None, max_len=512, random_state=None, use_gpu=True):
        super().__init__()
        self.num_event_types = num_event_types
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.pad_value = num_event_types if pad_value is None else pad_value
        self.random_state = random_state
        self.max_len = max_len
        self.geometry = "euclidean"
        self.curvature = 1.0
        _seed(random_state)
        self.device = _device(use_gpu)
        self.embedding = nn.Embedding(num_event_types + 1, embedding_dim,
                                      padding_idx=self.pad_value)
        self.pos = nn.Embedding(max_len, embedding_dim)
        layer = nn.TransformerEncoderLayer(
            embedding_dim, nhead, dim_feedforward, batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.dec = nn.Linear(embedding_dim, num_event_types)
        self.to(self.device)

    def _embed(self, x):
        L = x.size(1)
        pos = torch.arange(L, device=x.device).clamp_max(self.max_len - 1)
        return self.embedding(x) + self.pos(pos)[None]

    def _logits(self, inp):
        L = inp.size(1)
        mask = torch.triu(torch.ones(L, L, device=inp.device), diagonal=1).bool()
        key_pad = inp == self.pad_value
        h = self.encoder(self._embed(inp), mask=mask, src_key_padding_mask=key_pad)
        # fully-masked (all-padding) query rows can emit NaN; targets there are
        # ignored in the loss, but NaN would still poison the backward pass.
        return torch.nan_to_num(self.dec(h))

    def _pool(self, padded, lengths):
        key_pad = padded == self.pad_value
        h = self.encoder(self._embed(padded), src_key_padding_mask=key_pad)
        h = torch.nan_to_num(h)  # padded rows may be NaN; they are zeroed by the mask below
        mask = (~key_pad).float()[..., None]
        return (h * mask).sum(1) / mask.sum(1).clamp_min(1.0)


class Word2VecBagEncoder:
    """Order-blind baseline: mean of gensim Word2Vec token vectors."""

    def __init__(self, num_event_types, embedding_dim=32, window=5, min_count=1,
                 num_epochs=12, random_state=0, pooling="mean", **_):
        self.num_event_types = num_event_types
        self.embedding_dim = embedding_dim
        self.window = window
        self.min_count = min_count
        self.num_epochs = num_epochs
        self.random_state = random_state
        self.pooling = pooling
        self.geometry = "euclidean"
        self.curvature = 1.0
        self._vectors = None

    def fit(self, sequences, **_):
        from gensim.models import Word2Vec

        toks = [[str(int(t)) for t in (s.tolist() if isinstance(s, torch.Tensor) else s)]
                for s in sequences]
        model = Word2Vec(
            sentences=toks, vector_size=self.embedding_dim, window=self.window,
            min_count=self.min_count, workers=1, seed=self.random_state, epochs=self.num_epochs,
        )
        vecs = np.zeros((self.num_event_types, self.embedding_dim), dtype=np.float32)
        for i in range(self.num_event_types):
            key = str(i)
            if key in model.wv:
                vecs[i] = model.wv[key]
        self._vectors = torch.tensor(vecs)
        return self

    def transform(self, sequences, as_numpy=True, **_):
        out = []
        for s in sequences:
            toks = [int(t) for t in (s.tolist() if isinstance(s, torch.Tensor) else s)]
            if not toks:
                out.append(torch.zeros(self.embedding_dim))
                continue
            v = self._vectors[toks]
            out.append(v.sum(0) if self.pooling == "sum" else v.mean(0))
        emb = torch.stack(out)
        return emb.numpy() if as_numpy else emb

    @property
    def n_params(self):
        # Skip-Gram trains TWO V x d matrices (input `wv` and output `syn1neg`); gensim
        # discards the output one after fitting. Reporting only the retained matrix would
        # make this look half the size of Event2Vec at equal capacity, so count both and
        # note that only V x d is kept at inference time.
        return 2 * self.num_event_types * self.embedding_dim


class GraphWalkEncoder:
    """Order-blind graph baselines: DeepWalk and Node2Vec.

    These methods do not embed sequences directly; they embed the *nodes* of a
    graph learned from random walks, exactly the ``walks as unordered context''
    setting discussed in the Related Work. We build the observed event-transition
    graph (nodes = event types, edge weight = count of ``i \\to j`` transitions in
    the training sequences), learn node vectors by DeepWalk (uniform walks +
    Skip-Gram) or Node2Vec (biased $p,q$ walks), and pool the node vectors of a
    sequence's tokens into a sequence embedding. Pooling is a *sum* by default so
    the comparison to Event2Vec's additive endpoint isolates the learned
    representation rather than the pooling rule (``mean`` is also available).
    """

    def __init__(self, num_event_types, embedding_dim=32, method="deepwalk",
                 walk_length=20, num_walks=10, window=5, p=1.0, q=1.0,
                 pooling="sum", num_epochs=5, random_state=0, **_):
        self.num_event_types = num_event_types
        self.embedding_dim = embedding_dim
        self.method = method.lower()
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window = window
        self.p = p
        self.q = q
        self.pooling = pooling
        self.num_epochs = num_epochs
        self.random_state = random_state
        self.geometry = "euclidean"
        self.curvature = 1.0
        self._vectors = None

    def _graph(self, sequences):
        import networkx as nx

        g = nx.DiGraph()
        g.add_nodes_from(range(self.num_event_types))
        for s in sequences:
            toks = [int(t) for t in (s.tolist() if isinstance(s, torch.Tensor) else s)]
            for a, b in zip(toks[:-1], toks[1:]):
                w = g[a][b]["weight"] + 1.0 if g.has_edge(a, b) else 1.0
                g.add_edge(a, b, weight=w)
        return g

    def _deepwalk_walks(self, g):
        rng = random.Random(self.random_state)
        nodes = [n for n in g.nodes() if g.out_degree(n) > 0]
        walks = []
        for _ in range(self.num_walks):
            rng.shuffle(nodes)
            for start in nodes:
                walk = [start]
                while len(walk) < self.walk_length:
                    cur = walk[-1]
                    nbrs = list(g.successors(cur))
                    if not nbrs:
                        break
                    wts = np.array([g[cur][n]["weight"] for n in nbrs], dtype=np.float64)
                    walk.append(nbrs[int(rng.choices(range(len(nbrs)), weights=wts)[0])])
                walks.append([str(n) for n in walk])
        return walks

    def fit(self, sequences, **_):
        g = self._graph(sequences)
        if self.method == "node2vec":
            from node2vec import Node2Vec

            n2v = Node2Vec(g, dimensions=self.embedding_dim, walk_length=self.walk_length,
                           num_walks=self.num_walks, p=self.p, q=self.q, weight_key="weight",
                           workers=1, quiet=True, seed=self.random_state)
            wv = n2v.fit(window=self.window, min_count=0, sg=1, seed=self.random_state,
                         epochs=self.num_epochs).wv
        else:  # deepwalk = uniform random walks + Skip-Gram
            from gensim.models import Word2Vec

            walks = self._deepwalk_walks(g)
            wv = Word2Vec(walks, vector_size=self.embedding_dim, window=self.window,
                          min_count=0, sg=1, workers=1, seed=self.random_state,
                          epochs=self.num_epochs).wv
        vecs = np.zeros((self.num_event_types, self.embedding_dim), dtype=np.float32)
        for i in range(self.num_event_types):
            if str(i) in wv:
                vecs[i] = wv[str(i)]
        self._vectors = torch.tensor(vecs)
        return self

    def transform(self, sequences, as_numpy=True, **_):
        out = []
        for s in sequences:
            toks = [int(t) for t in (s.tolist() if isinstance(s, torch.Tensor) else s)]
            if not toks:
                out.append(torch.zeros(self.embedding_dim))
                continue
            v = self._vectors[toks]
            out.append(v.sum(0) if self.pooling == "sum" else v.mean(0))
        emb = torch.stack(out)
        return emb.numpy() if as_numpy else emb

    @property
    def n_params(self):
        return 2 * self.num_event_types * self.embedding_dim  # Skip-Gram: input + output matrices


def build_baseline(name: str, num_event_types: int, **kw):
    """Construct a baseline by name.

    Note on pooling: ``Word2VecBagEncoder``/``GraphWalkEncoder`` default to *sum*
    pooling here, matching \methodtitle{}'s endpoint (which is a sum, not a mean).
    Mean pooling rescales every sequence by 1/L and so measures something different;
    it stays available via ``pooling="mean"`` but is no longer what a caller gets by
    forgetting to pass the argument -- the paper's tables are all sum-pooled.
    """
    name = name.lower()
    if name in ("word2vec", "word2vec_bag", "bag", "deepwalk", "node2vec"):
        kw.setdefault("pooling", "sum")
    if name in ("gru", "lstm"):
        return RNNEncoder(num_event_types, kind=name, **kw)
    if name == "transformer":
        return TransformerEncoder(num_event_types, **kw)
    if name in ("word2vec", "word2vec_bag", "bag"):
        return Word2VecBagEncoder(num_event_types, **kw)
    if name in ("deepwalk", "node2vec"):
        method = "node2vec" if name == "node2vec" else "deepwalk"
        return GraphWalkEncoder(num_event_types, method=method, **kw)
    raise ValueError(f"unknown baseline: {name}")
