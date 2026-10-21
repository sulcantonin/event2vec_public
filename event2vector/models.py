import random
import warnings
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from .geometries import (
    AffineModel,
    HeisenbergModel,
    ProductModel,
    RotationModel,
    StereographicModel,
)

EPS = 1e-5  # Small constant for numerical stability of hyperbolic space


class EuclideanModel(nn.Module):
    """
    A recurrent model for learning event sequence representations in Euclidean space.
    The model uses a simple additive update rule, where the representation of a
    sequence is the sum of its constituent event embeddings.
    """

    geometry = "euclidean"

    def __init__(self, num_event_types, embedding_dim, dropout_p=0.2, max_norm=10.0):
        super(EuclideanModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.state_dim = embedding_dim
        self.max_norm = max_norm
        self.embedding = nn.Embedding(num_event_types, embedding_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.linear_dec = nn.Linear(embedding_dim, num_event_types)

    def zero_state(self, batch, device):
        return torch.zeros((batch, self.embedding_dim), device=device)

    def reset_parameters(self):
        """Geometry-aware (re)initialization used by the estimator."""
        nn.init.xavier_uniform_(self.linear_dec.weight)
        if self.linear_dec.bias is not None:
            nn.init.zeros_(self.linear_dec.bias)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def encoder(self, x, h=None, clipping=True):
        if h is None:
            h = torch.zeros((x.shape[0], self.embedding_dim), device=x.device)
        # Dropout the *increment* (not the accumulated state) so the additive
        # recurrence h_t = h_{t-1} + e_t stays a clean sum of (possibly dropped)
        # increments; this keeps the additive semantics and lets the two-pass
        # consistency loss still see distinct masks.
        e = self.dropout(self.embedding(x))
        h_next = e + h

        # max_norm=None (or inf) disables clipping, keeping h_t an exact sum; an
        # infinite coefficient would otherwise turn the gradient into 0 * inf = NaN.
        if clipping and self.max_norm is not None and self.max_norm != float("inf"):
            norm = torch.norm(h_next, p=2, dim=1, keepdim=True)
            clip_coef = self.max_norm / (norm + 1e-6)
            clip_coef = torch.min(clip_coef, torch.ones_like(clip_coef))
            h_next = h_next * clip_coef

        return h_next, e

    def decoder(self, h):
        return self.linear_dec(h)

    def forward(self, x, h=None):
        # Returns (logits, next_state, increment). `increment` is the vector
        # actually added at this step (post-dropout).
        h, e = self.encoder(x, h)
        return self.decoder(h), h, e


class HyperbolicUtils:
    """A collection of static methods for operations in the Poincaré Ball model."""

    @staticmethod
    def mobius_add(x, y, c):
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        den = 1 + 2 * c * xy + c**2 * x2 * y2
        # clamp_min, not den + EPS: the denominator is ~1 for points inside the ball, so
        # adding EPS biased *every* update by ~1e-5 and floored the hyperbolic
        # reconstruction loss. This also makes this path bit-identical to
        # StereographicUtils.add at kappa = -c, which was not previously the case.
        return num / den.clamp_min(EPS)

    @staticmethod
    def log_map_origin(y, c):
        sqrt_c = c**0.5
        y_norm = torch.norm(y, p=2, dim=-1, keepdim=True).clamp_min(EPS)
        artanh_arg = (sqrt_c * y_norm).clamp(max=1.0 - EPS)
        log_map = (1. / sqrt_c) * (0.5 * torch.log((1 + artanh_arg) / (1 - artanh_arg))) * (y / y_norm)
        return log_map

    @staticmethod
    def poincare_dist_sq(x, y, c):
        sqrt_c = c**0.5
        mob_add_res = HyperbolicUtils.mobius_add(-x, y, c)
        mob_add_norm = torch.norm(mob_add_res, p=2, dim=-1, keepdim=True).clamp_min(EPS)
        artanh_arg = (sqrt_c * mob_add_norm).clamp(max=1.0 - EPS)
        dist = (2. / sqrt_c) * (0.5 * torch.log((1 + artanh_arg) / (1 - artanh_arg)))
        return dist.pow(2)

    @staticmethod
    def project_to_ball(x, c):
        max_norm = 1.0 / (c**0.5)
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        cond = norm >= max_norm
        projected_x = torch.where(cond, x / (norm + EPS) * (max_norm - EPS), x)
        return projected_x


class HyperbolicModel(nn.Module):
    """
    A recurrent model for learning event sequence representations in Hyperbolic space.
    This model is well-suited for hierarchical data, as the geometry of hyperbolic
    space can embed tree-like structures with low distortion.
    """

    geometry = "hyperbolic"

    def __init__(self, num_event_types, embedding_dim, dropout_p, c=1.0):
        super(HyperbolicModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.state_dim = embedding_dim
        self.c = c
        self.embedding = nn.Embedding(num_event_types, embedding_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.linear_dec = nn.Linear(embedding_dim, num_event_types)
        self.reset_parameters()

    def zero_state(self, batch, device):
        return torch.zeros((batch, self.embedding_dim), device=device)

    def reset_parameters(self):
        """Geometry-aware (re)initialization: tiny near-origin embeddings so the
        Poincare-ball tangent-space linearization is well conditioned, then
        project inside the ball. (Previously this was clobbered by a generic
        std=0.02 normal init, pushing points far from the origin.)"""
        nn.init.xavier_uniform_(self.linear_dec.weight)
        if self.linear_dec.bias is not None:
            nn.init.zeros_(self.linear_dec.bias)
        self.embedding.weight.data.uniform_(-0.001, 0.001)
        self.project_embeddings()

    def project_embeddings(self):
        with torch.no_grad():
            self.embedding.weight.data = HyperbolicUtils.project_to_ball(
                self.embedding.weight.data, self.c
            )

    def encoder(self, x, h):
        e = self.embedding(x)
        e_dropped = self.dropout(e)
        h_next = HyperbolicUtils.mobius_add(h, e_dropped, self.c)
        h_next = HyperbolicUtils.project_to_ball(h_next, self.c)
        # Return the dropped increment that was actually Mobius-added, so the
        # reconstruction loss can recover it exactly via left-cancellation.
        return h_next, e_dropped

    def decoder(self, h):
        h_tangent = HyperbolicUtils.log_map_origin(h, self.c)
        return self.linear_dec(h_tangent)

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros((x.shape[0], self.embedding_dim), device=x.device)
        h_next, e = self.encoder(x, h)
        return self.decoder(h_next), h_next, e


SequenceType = Union[torch.Tensor, Sequence[int]]


def _nonnegative_lstsq(A, b):
    """``argmin_{x >= 0} ||A x - b||``.

    The active-set NNLS returns a sparse minimizer, which is what decoding event
    counts needs when ``A`` has more columns than rows. Its default iteration cap
    (3x the columns) is too small for some embeddings, so it gets a larger one and,
    if that still fails, falls back to bounded-variable least squares.
    """
    from scipy.optimize import lsq_linear, nnls

    try:
        return nnls(A, b, maxiter=50 * A.shape[1])[0]
    except RuntimeError:
        return lsq_linear(A, b, bounds=(0.0, np.inf), method="bvls").x


class Event2Vec:
    """
    High-level estimator that follows the fit/transform conventions of scikit-learn
    while implementing the additive training objective described in the Event2Vec
    paper (Sulc, 2025). Supports both Euclidean and Hyperbolic geometries.
    """

    def __init__(
        self,
        num_event_types: int,
        embedding_dim: int = 64,
        geometry: str = "euclidean",
        dropout_p: float = 0.2,
        max_norm: float = 10.0,
        curvature: float = 1.0,
        kappa: Optional[float] = None,
        rotation_n: int = 3,
        heisenberg_m: int = 1,
        affine_readout: str = "point",
        affine_max_log_scale: Optional[float] = None,
        product_kappas: Sequence[float] = (-1.0, 1.0),
        lambda_reconstruction: float = 0.2,
        lambda_consistency: float = 0.2,
        lambda_difference: float = 0.0,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        num_epochs: int = 50,
        pad_sequences: bool = False,
        pad_value: int = 0,
        use_gpu: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        random_state: Optional[int] = None,
        deterministic: bool = False,
        riemannian: bool = False,
    ):
        if num_event_types <= 0:
            raise ValueError("num_event_types must be positive.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if num_epochs <= 0:
            raise ValueError("num_epochs must be positive.")
        if geometry.lower() == "hyperbolic" and curvature <= 0:
            raise ValueError(
                "curvature must be positive (c>0 corresponds to sectional curvature -c)."
            )

        self.num_event_types = num_event_types
        self.embedding_dim = embedding_dim
        self.geometry = geometry.lower()
        self.dropout_p = dropout_p
        self.max_norm = max_norm
        self.curvature = curvature
        self.kappa = kappa
        self.rotation_n = rotation_n
        self.heisenberg_m = heisenberg_m
        self.affine_readout = affine_readout
        self.affine_max_log_scale = affine_max_log_scale
        self.product_kappas = tuple(product_kappas)
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_consistency = lambda_consistency
        self.lambda_difference = lambda_difference
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.pad_sequences = pad_sequences
        self.pad_value = pad_value
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.deterministic = deterministic
        self.riemannian = riemannian
        self._rng = random.Random(random_state)
        if device is None:
            if use_gpu and torch.cuda.is_available():
                device = torch.device("cuda")
            elif use_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                if self.geometry in ("rotation", "affine"):
                    # MPS has no matrix_exp, so every step would round-trip through
                    # the CPU; for a model this small the CPU is much faster.
                    warnings.warn(
                        f"geometry='{self.geometry}' needs torch.matrix_exp, which MPS does not "
                        "implement; using device='cpu'. Pass device='mps' to run on MPS "
                        "anyway (matrix_exp is then computed on the CPU).",
                        RuntimeWarning,
                    )
                    device = torch.device("cpu")
                else:
                    device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device)
        self.device = device

        self.model = self._build_model()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.training_loss_: List[float] = []
        self.is_fitted_: bool = False

    def _build_model(self):
        if self.geometry == "euclidean":
            model = EuclideanModel(
                num_event_types=self.num_event_types,
                embedding_dim=self.embedding_dim,
                dropout_p=self.dropout_p,
                max_norm=self.max_norm,
            )
        elif self.geometry == "hyperbolic":
            model = HyperbolicModel(
                num_event_types=self.num_event_types,
                embedding_dim=self.embedding_dim,
                dropout_p=self.dropout_p,
                c=self.curvature,
            )
        elif self.geometry in ("spherical", "stereographic"):
            if self.geometry == "spherical":
                kappa = self.kappa if self.kappa is not None else abs(self.curvature)
                if kappa <= 0:
                    raise ValueError("spherical geometry needs positive curvature (kappa>0).")
            else:
                if self.kappa is None:
                    raise ValueError(
                        "stereographic geometry requires `kappa` "
                        "(kappa<0 hyperbolic, 0 Euclidean, >0 spherical)."
                    )
                kappa = self.kappa
            model = StereographicModel(
                num_event_types=self.num_event_types,
                embedding_dim=self.embedding_dim,
                dropout_p=self.dropout_p,
                kappa=kappa,
            )
        elif self.geometry == "rotation":
            model = RotationModel(
                num_event_types=self.num_event_types,
                n=self.rotation_n,
                dropout_p=self.dropout_p,
            )
        elif self.geometry == "heisenberg":
            model = HeisenbergModel(
                num_event_types=self.num_event_types,
                m=self.heisenberg_m,
                dropout_p=self.dropout_p,
            )
        elif self.geometry == "affine":
            model = AffineModel(
                num_event_types=self.num_event_types,
                embedding_dim=self.embedding_dim,
                dropout_p=self.dropout_p,
                readout=self.affine_readout,
                max_log_scale=self.affine_max_log_scale,
            )
        elif self.geometry == "product":
            factors = [
                StereographicModel(
                    num_event_types=self.num_event_types,
                    embedding_dim=self.embedding_dim,
                    dropout_p=self.dropout_p,
                    kappa=k,
                )
                for k in self.product_kappas
            ]
            model = ProductModel(factors)
        else:
            raise ValueError(
                "geometry must be one of: 'euclidean', 'hyperbolic', 'spherical', "
                "'stereographic', 'rotation', 'heisenberg', 'affine', 'product'."
            )
        return model.to(self.device)

    def _reset_parameters(self):
        # Delegate to the model's geometry-aware initializer. The previous
        # generic `apply(_init_weights)` overwrote the hyperbolic model's tiny
        # near-origin embedding init with a std=0.02 normal (~160x too large for
        # the unit ball), which degraded the hyperbolic geometry.
        self.model.reset_parameters()

    def _seed_everything(self):
        """Seed the global RNGs and (optionally) enforce deterministic kernels."""
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)
            self._rng = random.Random(self.random_state)
        if self.deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if str(self.device).startswith("mps"):
                warnings.warn(
                    "deterministic=True requested but device is MPS, whose kernels "
                    "are not guaranteed reproducible; use device='cpu' for exact "
                    "determinism.",
                    RuntimeWarning,
                )

    def _make_optimizer(self):
        """AdamW for Euclidean; optionally RiemannianAdam (geoopt) for hyperbolic."""
        if self.geometry == "hyperbolic" and self.riemannian:
            try:
                import geoopt
            except ImportError:
                warnings.warn(
                    "riemannian=True but geoopt is not installed; falling back to "
                    "AdamW with manifold projection. `pip install geoopt` to enable "
                    "true Riemannian optimization.",
                    RuntimeWarning,
                )
                self.riemannian = False
                return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

            ball = geoopt.PoincareBall(c=self.curvature)
            emb = self.model.embedding
            if not isinstance(emb.weight, geoopt.ManifoldParameter):
                with torch.no_grad():
                    projected = HyperbolicUtils.project_to_ball(
                        emb.weight.data, self.curvature
                    )
                emb.weight = geoopt.ManifoldParameter(projected, manifold=ball)
            return geoopt.optim.RiemannianAdam(
                self.model.parameters(), lr=self.learning_rate
            )
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def fit(
        self,
        sequences: Iterable[SequenceType],
        y=None,
        epochs: Optional[int] = None,
        pad_sequences: Optional[bool] = None,
        verbose: bool = False,
    ):
        """
        Fits the model on the provided sequences.
        """
        processed = self._standardize_sequences(sequences)
        use_padding = self.pad_sequences if pad_sequences is None else pad_sequences
        total_epochs = epochs or self.num_epochs

        self._seed_everything()
        self._reset_parameters()
        optimizer = self._make_optimizer()
        self.training_loss_ = []

        for epoch in range(total_epochs):
            epoch_loss = self._train_epoch(processed, optimizer, use_padding)
            self.training_loss_.append(epoch_loss)
            if verbose:
                print(f"Epoch {epoch + 1}/{total_epochs} - loss: {epoch_loss:.4f}")

        self.is_fitted_ = True
        return self

    def fit_transform(
        self,
        sequences: Iterable[SequenceType],
        y=None,
        epochs: Optional[int] = None,
        pad_sequences: Optional[bool] = None,
        verbose: bool = False,
        as_numpy: bool = True,
    ):
        """
        Convenience method that fits the model and returns the transformed sequences.
        """
        self.fit(
            sequences,
            y=y,
            epochs=epochs,
            pad_sequences=pad_sequences,
            verbose=verbose,
        )
        return self.transform(
            sequences,
            pad_sequences=pad_sequences,
            as_numpy=as_numpy,
        )

    def transform(
        self,
        sequences: Iterable[SequenceType],
        pad_sequences: Optional[bool] = None,
        as_numpy: bool = True,
        batch_size: Optional[int] = None,
    ):
        """
        Encodes each sequence into the additive Event2Vec representation.
        """
        self._check_is_fitted()
        processed = self._standardize_sequences(sequences)
        use_padding = self.pad_sequences if pad_sequences is None else pad_sequences
        batch_size = batch_size or self.batch_size

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            if use_padding:
                embeddings = self._embed_padded(processed, batch_size)
            else:
                embeddings = self._embed_unpadded(processed)
        if was_training:
            self.model.train()

        if as_numpy:
            return embeddings.cpu().numpy()
        return embeddings

    def most_similar(
        self,
        positive: Union[
            int,
            SequenceType,
            torch.Tensor,
            Sequence[Union[int, SequenceType, torch.Tensor]],
        ],
        negative: Optional[
            Union[
                int,
                SequenceType,
                torch.Tensor,
                Sequence[Union[int, SequenceType, torch.Tensor]],
            ]
        ] = None,
        topn: int = 10,
        restrict_vocab: Optional[int] = None,
        normalize: bool = True,
        exclude: Optional[Sequence[int]] = None,
    ):
        """
        Returns the most similar event ids to a target combination of tokens.
        Mirrors gensim's `most_similar`: positive/negative can be token ids,
        1-D int tensors or Python sequences (treated as event sequences and
        encoded via the trained encoder), or pre-computed embedding vectors.

        Note: the arithmetic (sum of `positive`, minus `negative`) and the cosine
        ranking are computed in the *ambient Euclidean* space of the embedding
        table for both geometries. This matches the analogy queries reported in
        the paper. For the hyperbolic model this is an approximation (it ignores
        the Poincaré metric); use Möbius operations / Poincaré distance directly
        (see `HyperbolicUtils`) for geometry-faithful queries.
        """
        self._check_is_fitted()
        if not hasattr(self.model, "embedding"):
            raise NotImplementedError(
                "most_similar is undefined for the 'product' geometry, which has no "
                "single event-embedding table (query each factor's model instead)."
            )
        if topn <= 0:
            raise ValueError("topn must be positive.")
        if positive is None and negative is None:
            raise ValueError("At least one positive or negative item is required.")

        pos_items = positive if isinstance(positive, (list, tuple)) else [positive]
        neg_items = (
            []
            if negative is None
            else negative
            if isinstance(negative, (list, tuple))
            else [negative]
        )

        def _extract_index(item):
            if isinstance(item, int):
                return item
            if isinstance(item, torch.Tensor) and item.ndim == 0:
                return int(item.item())
            return None

        exclude_set = set(exclude or [])
        for item in pos_items + neg_items:
            idx = _extract_index(item)
            if idx is not None:
                exclude_set.add(idx)

        with torch.no_grad():
            pos_vecs = [self._vector_from_input(item) for item in pos_items]
            target_vec = torch.stack(pos_vecs).sum(dim=0)
            if neg_items:
                neg_vecs = [self._vector_from_input(item) for item in neg_items]
                target_vec = target_vec - torch.stack(neg_vecs).sum(dim=0)

            emb_matrix = self.model.embedding.weight.detach()
            if restrict_vocab is not None:
                emb_matrix = emb_matrix[:restrict_vocab]
            target_vec = target_vec.to(emb_matrix.device)
            if normalize:
                target_vec = torch.nn.functional.normalize(target_vec, dim=0)
                emb_matrix = torch.nn.functional.normalize(emb_matrix, dim=1)
            sims = emb_matrix @ target_vec

            if exclude_set:
                mask_indices = torch.tensor(
                    sorted(exclude_set), device=sims.device, dtype=torch.long
                )
                mask_indices = mask_indices[
                    (mask_indices >= 0) & (mask_indices < sims.numel())
                ]
                sims[mask_indices] = -float("inf")

            k = min(topn, emb_matrix.size(0))
            values, indices = torch.topk(sims, k=k)
        return [(idx.item(), float(val)) for idx, val in zip(indices, values)]

    # ------------------------------------------------------------------ #
    # Root-cause analysis
    #
    # For the Euclidean geometry the state is a (clip-weighted) sum of event
    # embeddings and the decoder is linear, so every quantity below is an exact
    # decomposition of the model's own logits. For the other geometries the
    # generic methods fall back to re-encoding (leave-one-out / forward passes),
    # which is exact for the model but not additive. All of them explain the
    # model's predictions; they are associations learned from data, not causal
    # effects.
    # ------------------------------------------------------------------ #
    def _as_ids(self, sequence: SequenceType) -> List[int]:
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.tolist()
        return [int(i) for i in sequence]

    def _next_logits(self, sequence: SequenceType) -> torch.Tensor:
        """Decoder logits for the event following ``sequence`` (dropout off)."""
        ids = self._as_ids(sequence)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            h = self.model.zero_state(1, self.device)
            logits = self.model.decoder(h)
            for i in ids:
                logits, h, _ = self.model(torch.tensor([i], device=self.device), h)
        if was_training:
            self.model.train()
        return logits[0]

    def _clip_weights(self, ids: List[int]) -> torch.Tensor:
        """Euclidean weights ``w_i`` with ``h_T = sum_i w_i e_{s_i}`` exactly.

        Step ``k`` scales ``h_{k-1} + e_k`` by ``gamma_k = min(1, max_norm / ||.||)``,
        so ``w_i = prod_{k >= i} gamma_k``; all ones when clipping never fires.
        """
        E = self.model.embedding.weight.detach()
        max_norm = self.model.max_norm
        h = torch.zeros(E.shape[1], device=E.device)
        gammas = []
        for i in ids:
            v = h + E[i]
            gamma = 1.0
            if max_norm is not None and max_norm != float("inf"):
                gamma = min(1.0, max_norm / (v.norm().item() + 1e-6))
            gammas.append(gamma)
            h = v * gamma
        g = torch.tensor(gammas, device=E.device)
        return torch.flip(torch.cumprod(torch.flip(g, [0]), 0), [0])

    def _require_euclidean(self, method: str):
        if self.geometry != "euclidean":
            raise NotImplementedError(
                f"{method} needs the state to be a sum of event embeddings, which only "
                "holds for geometry='euclidean'."
            )

    def attribute(self, sequence: SequenceType, target: Optional[int] = None) -> dict:
        """Contribution of each event in ``sequence`` to the logit of the next event ``target``.

        ``target`` defaults to the most likely next event. Euclidean: the logit is
        decomposed exactly, ``logit = bias + sum_i contributions[i]`` with
        ``contributions[i] = weights[i] * (W_dec e_{s_i})[target]``, where ``weights``
        are the norm-clipping weights (all ones with ``max_norm=None``), so recent
        events weigh more when clipping fires. Other geometries: leave-one-out,
        ``contributions[i] = logit(sequence) - logit(sequence without position i)``,
        which does not sum to the logit.

        Returns a dict with ``target``, ``logit``, ``contributions`` (one per
        position), ``weights`` (Euclidean only, else ``None``), ``bias`` and
        ``exact`` (whether the contributions sum to ``logit - bias``).
        """
        self._check_is_fitted()
        ids = self._as_ids(sequence)
        if not ids:
            raise ValueError("sequence must be non-empty.")
        logits = self._next_logits(ids)
        if target is None:
            target = int(logits.argmax())
        logit = float(logits[target])
        if self.geometry == "euclidean":
            E = self.model.embedding.weight.detach()
            W = self.model.linear_dec.weight.detach()
            b = self.model.linear_dec.bias.detach()
            weights = self._clip_weights(ids)
            contributions = weights * (E[ids] @ W[target])
            return {
                "target": target,
                "logit": logit,
                "contributions": contributions.cpu().numpy(),
                "weights": weights.cpu().numpy(),
                "bias": float(b[target]) if b is not None else 0.0,
                "exact": True,
            }
        contributions = np.array([
            logit - float(self._next_logits(ids[:i] + ids[i + 1:])[target])
            for i in range(len(ids))
        ])
        return {"target": target, "logit": logit, "contributions": contributions,
                "weights": None, "bias": None, "exact": False}

    def counterfactual(self, sequence: SequenceType, remove: Union[int, Sequence[int]]) -> dict:
        """Next-event distribution with and without the events at positions ``remove``.

        Re-encodes the edited sequence, so it is exact for every geometry. Returns
        ``probs`` (factual), ``probs_counterfactual`` and ``delta`` (their
        difference), each of length ``num_event_types``.
        """
        self._check_is_fitted()
        ids = self._as_ids(sequence)
        drop = {remove} if isinstance(remove, int) else set(remove)
        kept = [i for p, i in enumerate(ids) if p not in drop]
        p = torch.softmax(self._next_logits(ids), dim=0).cpu().numpy()
        q = torch.softmax(self._next_logits(kept), dim=0).cpu().numpy()
        return {"probs": p, "probs_counterfactual": q, "delta": q - p}

    def influence_matrix(self) -> np.ndarray:
        """``A[j, i]``: change in the logit of next event ``j`` caused by event ``i``
        alone, ``A[j, i] = logit_j([i]) - logit_j([])``.

        Euclidean without clipping this is exactly ``W_dec E^T``, and every
        sequence's logits are ``bias + sum_i count_i A[:, i]``; with clipping it is
        the single-event effect. Shape ``(num_event_types, num_event_types)``.
        """
        self._check_is_fitted()
        base = self._next_logits([])
        cols = [self._next_logits([i]) - base for i in range(self.num_event_types)]
        return torch.stack(cols, dim=1).cpu().numpy()

    def event_impact(self) -> np.ndarray:
        """How strongly one occurrence of each event moves the next-event
        distribution: the norm of its centered influence column
        ``||A[:, i] - mean(A[:, i])||`` (centering removes the softmax-invariant
        shift). Shape ``(num_event_types,)``.
        """
        A = self.influence_matrix()
        return np.linalg.norm(A - A.mean(axis=0, keepdims=True), axis=0)

    def decompose(self, item: Union[SequenceType, torch.Tensor], round_counts: bool = False) -> np.ndarray:
        """Recover event counts from a Euclidean state by non-negative least squares,
        ``min_{x >= 0} ||E^T x - h||``.

        ``item`` is a sequence (encoded first) or a state vector. Without clipping
        the true counts are the unique solution whenever the history is short
        enough relative to the coherence of the embeddings; with clipping the
        result estimates the clip-weighted counts. ``round_counts`` rounds to
        integers. Shape ``(num_event_types,)``.
        """
        self._check_is_fitted()
        self._require_euclidean("decompose")
        h = self._state_of(item)
        E = self.model.embedding.weight.detach().cpu().double().numpy()
        x = _nonnegative_lstsq(E.T, h)
        return np.round(x) if round_counts else x

    def explain_difference(self, sequence_a: SequenceType, sequence_b: SequenceType,
                           round_counts: bool = False) -> np.ndarray:
        """Which events differ between two histories, read from their states alone.

        Decodes ``h_a - h_b`` into signed counts (positive: more in ``a``) by
        non-negative least squares on ``[E^T, -E^T]``. Without clipping
        ``h_a - h_b = E^T (c_a - c_b)`` exactly, so a failing run can be compared
        with a healthy one. Shape ``(num_event_types,)``.
        """
        self._check_is_fitted()
        self._require_euclidean("explain_difference")
        d = self._state_of(sequence_a) - self._state_of(sequence_b)
        E = self.model.embedding.weight.detach().cpu().double().numpy()
        x = _nonnegative_lstsq(np.concatenate([E.T, -E.T], axis=1), d)
        diff = x[: self.num_event_types] - x[self.num_event_types:]
        return np.round(diff) if round_counts else diff

    def _state_of(self, item) -> np.ndarray:
        if isinstance(item, torch.Tensor) and item.dtype.is_floating_point:
            return item.detach().cpu().double().numpy()
        if isinstance(item, np.ndarray) and np.issubdtype(item.dtype, np.floating):
            return item.astype(np.float64)
        ids = self._as_ids(item)
        return self.transform([ids], as_numpy=True)[0].astype(np.float64)

    def _standardize_sequences(
        self, sequences: Iterable[SequenceType]
    ) -> List[torch.Tensor]:
        processed = []
        for seq in sequences:
            if isinstance(seq, torch.Tensor):
                tensor = seq.detach().long().cpu()
            else:
                tensor = torch.tensor(seq, dtype=torch.long)
            if tensor.ndim != 1:
                raise ValueError("Each sequence must be 1-D.")
            processed.append(tensor)
        if not processed:
            raise ValueError("At least one sequence is required.")
        return processed

    def _history_reconstruction_loss(self, counts):
        """Euclidean event-history reconstruction ``||alpha* E E^T c / n - c / n||^2``.

        ``counts`` is a ``(B, V)`` histogram of the events in each row's history
        and ``n`` its length. The history embedding ``h = E^T c / n`` is rebuilt
        from the counts (no dropout, no clipping, not detached) and decoded with
        the same embedding matrix, so the gradient reaches ``E`` through both the
        encoder and the decoder. Rows with an empty history are excluded; the
        result is the mean over the remaining rows.

        ``alpha* = <E E^T c, c> / ||E E^T c||^2`` is the closed-form best scale
        per row (no trainable parameter). It makes the loss invariant to the scale
        of ``E``, so it scores only how well the embedding preserves the histogram
        and does not pin the embedding norm (``||c||^2 sin^2`` of the angle between
        ``E E^T c`` and ``c``).
        """
        n = counts.sum(dim=1, keepdim=True)
        valid = n.squeeze(1) > 0
        if not torch.any(valid):
            return counts.new_zeros(())
        c = counts[valid] / n[valid]
        E = self.model.embedding.weight
        c_hat = (c @ E) @ E.t()
        alpha = (c_hat * c).sum(dim=1, keepdim=True) / (
            c_hat.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-12)
        )
        return (alpha * c_hat - c).pow(2).sum(dim=1).mean()

    def _difference_reconstruction_loss(self, counts):
        """Euclidean difference reconstruction, the same scale-invariant form as
        ``_history_reconstruction_loss`` applied to differences of histories.

        ``counts`` is a ``(B, V)`` batch of histograms. Each row is paired with
        another row of the batch (a random cyclic shift), both normalized by
        their length, and the unit-normalized difference ``d`` must decode to
        itself: ``min_alpha ||alpha E E^T d - d||^2``, which equals the squared
        sine of the angle between ``E E^T d`` and ``d``. Pairs with identical
        histograms are skipped. This trains ``h_a - h_b`` to be decodable into
        the events that differ, which the history term alone does not.
        """
        n = counts.sum(dim=1, keepdim=True)
        valid = n.squeeze(1) > 0
        if valid.sum() < 2:
            return counts.new_zeros(())
        c = counts[valid] / n[valid]
        shift = int(torch.randint(1, c.shape[0], (1,)).item())
        d = c - torch.roll(c, shifts=shift, dims=0)
        norm = d.norm(dim=1, keepdim=True)
        keep = norm.squeeze(1) > 1e-8
        if not torch.any(keep):
            return counts.new_zeros(())
        d = d[keep] / norm[keep]
        E = self.model.embedding.weight
        d_hat = (d @ E) @ E.t()
        alpha = (d_hat * d).sum(dim=1, keepdim=True) / (
            d_hat.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-12)
        )
        return (alpha * d_hat - d).pow(2).sum(dim=1).mean()

    def _compute_losses(self, y1, h1, e_curr1, h_old, target, h2, counts=None):
        """Per-step prediction / reconstruction / consistency losses.

        Reconstruction:
        - Euclidean: reconstructs the normalized event histogram ``counts`` of
          the history from its additive embedding with the tied decoder ``E``
          (see ``_history_reconstruction_loss``). The previous
          ``(h_t - e_t) == h_{t-1}`` form was identically zero by construction
          of ``h_t = h_{t-1} + e_t`` (non-zero only when norm clipping fired).
        - Hyperbolic: the gyrogroup *left-cancellation* law recovers the
          increment exactly, ``(-h_{t-1}) ⊕_c h_t == e_t``. (The previous
          ``(h_t ⊕_c -e_t)`` form is a right-cancellation and carries a
          non-vanishing gyration residual, so it could not reach zero even at
          the ideal Möbius update.)
        """
        prediction_loss = self.ce_loss(y1, target)
        if self.geometry == "euclidean":
            reconstruction_loss = self._history_reconstruction_loss(counts)
            consistency_loss = self.mse_loss(h1, h2)
        elif self.geometry == "hyperbolic":
            e_recovered = HyperbolicUtils.mobius_add(-h_old, h1, self.curvature)
            reconstruction_loss = HyperbolicUtils.poincare_dist_sq(
                e_recovered, e_curr1, self.curvature
            ).mean()
            consistency_loss = HyperbolicUtils.poincare_dist_sq(
                h1, h2, self.curvature
            ).mean()
        else:
            # Generic gyrogroup / Lie-group reconstruction: the geometry's own
            # left-cancellation recovers the increment, and its squared distance
            # scores the residual and the two-pass consistency. Exactly zero at the
            # ideal update for every geometry implementing this interface.
            e_recovered = self.model.recover_increment(h_old, h1)
            reconstruction_loss = self.model.dist2(e_recovered, e_curr1).mean()
            consistency_loss = self.model.dist2(h1, h2).mean()
        return prediction_loss, reconstruction_loss, consistency_loss

    def _zero_counts(self, batch):
        """Empty event histograms for the Euclidean reconstruction loss (else None)."""
        if self.geometry != "euclidean":
            return None
        return torch.zeros((batch, self.num_event_types), device=self.device)

    def _train_epoch(self, sequences, optimizer, use_padding: bool) -> float:
        shuffled = sequences.copy()
        self._rng.shuffle(shuffled)
        total_loss = 0.0
        total_batches = 0
        for idx in range(0, len(shuffled), self.batch_size):
            batch = shuffled[idx : idx + self.batch_size]
            if use_padding:
                batch_loss = self._train_batch_padded(batch, optimizer)
            else:
                batch_loss = self._train_batch_unpadded(batch, optimizer)
            if batch_loss is not None:
                total_loss += batch_loss
                total_batches += 1
        return total_loss / max(1, total_batches)

    def _train_batch_unpadded(self, batch, optimizer):
        # Transition-weighted mean (every transition contributes equally),
        # matching `_train_batch_padded` so switching `pad_sequences` does not
        # change the optimized objective.
        total_loss = torch.tensor(0.0, device=self.device)
        total_transitions = 0
        for seq in batch:
            if seq.numel() < 2:
                continue
            seq = seq.to(self.device)
            h = self.model.zero_state(1, self.device)
            counts = self._zero_counts(1)
            for t in range(seq.numel() - 1):
                x = seq[t].unsqueeze(0)
                target = seq[t + 1].unsqueeze(0)
                h_old = h.detach()
                y1, h1, e_curr1 = self.model(x, h_old)
                _, h2, _ = self.model(x, h_old)
                if counts is not None:
                    # Histogram of seq[: t + 1], the events summed into h1.
                    counts = counts + nn.functional.one_hot(x, self.num_event_types)

                prediction_loss, reconstruction_loss, consistency_loss = (
                    self._compute_losses(y1, h1, e_curr1, h_old, target, h2, counts)
                )
                combined_loss = (
                    prediction_loss
                    + self.lambda_reconstruction * reconstruction_loss
                    + self.lambda_consistency * consistency_loss
                )
                total_loss = total_loss + combined_loss
                total_transitions += 1
                h = h1.detach()

        if total_transitions == 0:
            return None

        avg_loss = total_loss / total_transitions
        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()
        self._project_if_needed()
        return float(avg_loss.item())

    def _train_batch_padded(self, batch, optimizer):
        sequences = [seq for seq in batch if seq.numel() >= 2]
        if not sequences:
            return None
        tensors = [seq.to(self.device) for seq in sequences]
        lengths = torch.tensor([tensor.size(0) for tensor in tensors], device=self.device)
        padded = pad_sequence(
            tensors, batch_first=True, padding_value=self.pad_value
        )

        h = self.model.zero_state(len(sequences), self.device)
        counts = self._zero_counts(len(sequences))
        total_loss = torch.tensor(0.0, device=self.device)
        total_transitions = 0

        for t in range(padded.size(1) - 1):
            valid_mask = lengths > (t + 1)
            if not torch.any(valid_mask):
                continue
            idx = torch.nonzero(valid_mask, as_tuple=True)[0]
            x = padded[idx, t]
            target = padded[idx, t + 1]
            h_old = h[idx].detach()

            y1, h1, e_curr1 = self.model(x, h_old)
            _, h2, _ = self.model(x, h_old)
            step_counts = None
            if counts is not None:
                # Only rows still inside their sequence are updated, so padding
                # never enters a histogram.
                counts = counts.index_add(
                    0, idx, nn.functional.one_hot(x, self.num_event_types).to(counts.dtype)
                )
                step_counts = counts[idx]

            prediction_loss, reconstruction_loss, consistency_loss = (
                self._compute_losses(y1, h1, e_curr1, h_old, target, h2, step_counts)
            )
            combined_loss = (
                prediction_loss
                + self.lambda_reconstruction * reconstruction_loss
                + self.lambda_consistency * consistency_loss
            )
            if self.lambda_difference > 0 and step_counts is not None:
                # Differences need at least two histories, which only the padded
                # loop (several sequences per step) provides.
                combined_loss = combined_loss + self.lambda_difference * (
                    self._difference_reconstruction_loss(step_counts)
                )
            step_transitions = idx.numel()
            total_loss = total_loss + combined_loss * step_transitions
            total_transitions += step_transitions

            h[idx] = h1.detach()

        if total_transitions == 0:
            return None

        avg_loss = total_loss / total_transitions
        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()
        self._project_if_needed()
        return float(avg_loss.item())

    def _embed_unpadded(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        outputs = []
        for seq in sequences:
            if seq.numel() == 0:
                outputs.append(self.model.zero_state(1, self.device).squeeze(0).cpu())
                continue
            seq = seq.to(self.device)
            h = self.model.zero_state(1, self.device)
            for token in seq:
                _, h, _ = self.model(token.unsqueeze(0), h)
            outputs.append(h.squeeze(0).cpu())
        return torch.stack(outputs)

    def _embed_padded(
        self, sequences: List[torch.Tensor], batch_size: int
    ) -> torch.Tensor:
        embeddings = []
        for idx in range(0, len(sequences), batch_size):
            batch = sequences[idx : idx + batch_size]
            tensors = [seq.to(self.device) for seq in batch]
            lengths = torch.tensor(
                [tensor.size(0) for tensor in tensors], device=self.device
            )
            padded = pad_sequence(
                tensors, batch_first=True, padding_value=self.pad_value
            )
            h = self.model.zero_state(len(batch), self.device)
            for t in range(padded.size(1)):
                valid_mask = lengths > t
                if not torch.any(valid_mask):
                    break
                active_idx = torch.nonzero(valid_mask, as_tuple=True)[0]
                x = padded[active_idx, t]
                h_old = h[active_idx]
                _, h_next, _ = self.model(x, h_old)
                h[active_idx] = h_next
            embeddings.append(h.detach().cpu())
        return torch.cat(embeddings, dim=0)

    def _project_if_needed(self):
        # When using a Riemannian optimizer the embedding is a geoopt
        # ManifoldParameter kept inside the ball by the optimizer's retraction,
        # so an extra hard projection is unnecessary (and would interfere).
        if self.riemannian:
            return
        if hasattr(self.model, "project_embeddings"):
            self.model.project_embeddings()

    def _check_is_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Event2Vec instance is not fitted yet.")

    def _vector_from_input(
        self, item: Union[int, SequenceType, torch.Tensor]
    ) -> torch.Tensor:
        """
        Converts an event id, a sequence of event ids, or an embedding vector
        into a single embedding on the model device.
        """
        if isinstance(item, torch.Tensor):
            if item.ndim == 0:
                idx = item.long().to(self.device)
                return self.model.embedding(idx).squeeze(0)
            if item.dtype.is_floating_point and item.ndim == 1:
                return item.to(self.device)
            if item.dtype in (
                torch.int64,
                torch.int32,
                torch.int16,
                torch.int8,
                torch.uint8,
            ) and item.ndim == 1:
                seq_tensor = item.long().cpu()
                return self.transform([seq_tensor], as_numpy=False)[0].to(self.device)
        if isinstance(item, int):
            idx_tensor = torch.tensor(item, device=self.device)
            return self.model.embedding(idx_tensor).squeeze(0)
        if isinstance(item, (list, tuple)):
            if len(item) == 0:
                raise ValueError("Sequence inputs must be non-empty.")
            seq_tensor = torch.tensor(item, dtype=torch.long)
            return self.transform([seq_tensor], as_numpy=False)[0].to(self.device)
        raise TypeError(
            "Inputs must be event indices, 1-D int tensors, sequences, or embedding vectors."
        )

