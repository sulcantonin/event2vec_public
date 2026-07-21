"""Additional geometric backbones for Event2Vec.

The Euclidean and hyperbolic models in :mod:`event2vector.models` cover a flat
(order-invariant) and a negatively-curved (order-sensitive) additive recurrence.
This module promotes the paper's *pilot* geometries to first-class trained models
that plug into the same :class:`~event2vector.models.Event2Vec` estimator, each the
natural home of a structure the additive Euclidean endpoint cannot carry:

================  ===========================  ===================================
geometry           update (state <- state, e)  what it captures
================  ===========================  ===================================
stereographic      kappa-gyroaddition          one signed curvature kappa unifies
                                                Euclidean (kappa=0), spherical
                                                (kappa>0) and hyperbolic (kappa<0)
rotation (SO(n))   left matrix product          non-commutative orientation;
                                                Euclidean sum sees only the
                                                abelianization
heisenberg         nilpotent group product      the signed (Levy) area / commutator,
                                                a purely order-dependent quantity
product            per-factor update            two structure types at once
                                                (e.g. hierarchy H x cycle S)
================  ===========================  ===================================

Every model exposes a small, uniform interface the estimator drives generically:
``state_dim``, ``zero_state(batch, device)``, ``forward(x, h) -> (logits, h_next,
increment)``, ``recover_increment(h_old, h_new)`` (the geometry's left-cancellation,
used by the reconstruction loss) and ``dist2(a, b)`` (squared geodesic-ish distance,
used by the reconstruction and consistency losses).
"""
import math

import torch
import torch.nn as nn

EPS = 1e-5


# --------------------------------------------------------------------------- #
# kappa-stereographic model: one signed curvature for E / S / H
# --------------------------------------------------------------------------- #
class StereographicUtils:
    """Operations of the kappa-stereographic model (Bachmann et al., 2020).

    A single signed curvature ``kappa`` unifies the three constant-curvature
    geometries. With ``kappa < 0`` these reduce exactly to the Poincare-ball
    operations of :class:`~event2vector.models.HyperbolicUtils` with ``c = -kappa``;
    ``kappa = 0`` degenerates to ordinary vector addition; ``kappa > 0`` gives the
    spherical (stereographically projected) model.
    """

    @staticmethod
    def _tan_k(x, kappa):
        if kappa > 0:
            sk = kappa ** 0.5
            x = torch.clamp(x, max=(math.pi / 2 - 1e-3) / sk)
            return torch.tan(sk * x) / sk
        if kappa < 0:
            sk = (-kappa) ** 0.5
            return torch.tanh(sk * x) / sk
        return x

    @staticmethod
    def _artan_k(x, kappa):
        if kappa > 0:
            sk = kappa ** 0.5
            return torch.atan(sk * x) / sk
        if kappa < 0:
            sk = (-kappa) ** 0.5
            return torch.atanh(torch.clamp(sk * x, max=1.0 - EPS)) / sk
        return x

    @staticmethod
    def add(x, y, kappa):
        """kappa-gyroaddition. Reduces to Mobius add (c=-kappa) for kappa<0, to + at 0."""
        x2 = (x * x).sum(-1, keepdim=True)
        y2 = (y * y).sum(-1, keepdim=True)
        xy = (x * y).sum(-1, keepdim=True)
        num = (1 - 2 * kappa * xy - kappa * y2) * x + (1 + kappa * x2) * y
        den = 1 - 2 * kappa * xy + (kappa ** 2) * x2 * y2
        return num / den.clamp_min(EPS)

    @staticmethod
    def expmap0(v, kappa):
        vn = v.norm(dim=-1, keepdim=True).clamp_min(EPS)
        return StereographicUtils._tan_k(vn, kappa) * v / vn

    @staticmethod
    def logmap0(y, kappa):
        yn = y.norm(dim=-1, keepdim=True).clamp_min(EPS)
        return StereographicUtils._artan_k(yn, kappa) * y / yn

    @staticmethod
    def dist2(x, y, kappa):
        d = StereographicUtils.add(-x, y, kappa).norm(dim=-1, keepdim=True).clamp_min(EPS)
        return (2 * StereographicUtils._artan_k(d, kappa)) ** 2

    @staticmethod
    def project(x, kappa):
        """Keep points in a single chart: inside the ball for kappa<0, and below the
        tan blow-up radius for kappa>0 (a no-op for kappa=0)."""
        if kappa == 0:
            return x
        if kappa < 0:
            max_norm = (1.0 / (-kappa) ** 0.5) - EPS
        else:
            max_norm = (math.pi / 2 - 1e-2) / (kappa ** 0.5)
        norm = x.norm(dim=-1, keepdim=True)
        cond = norm > max_norm
        return torch.where(cond, x / (norm + EPS) * max_norm, x)


class StereographicModel(nn.Module):
    """Additive recurrence on a constant-curvature space of signed curvature ``kappa``.

    ``kappa < 0`` is hyperbolic (equivalent to :class:`HyperbolicModel` with
    ``c = -kappa``), ``kappa = 0`` is Euclidean, ``kappa > 0`` is spherical. The
    reconstruction target uses the gyrogroup left-cancellation, so it is exactly
    minimized at the ideal update for every kappa.
    """

    geometry = "stereographic"

    def __init__(self, num_event_types, embedding_dim, dropout_p=0.2, kappa=-1.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.state_dim = embedding_dim
        self.kappa = float(kappa)
        self.embedding = nn.Embedding(num_event_types, embedding_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.linear_dec = nn.Linear(embedding_dim, num_event_types)
        self.reset_parameters()

    def reset_parameters(self):
        if self.linear_dec is not None:  # None when used as a ProductModel factor
            nn.init.xavier_uniform_(self.linear_dec.weight)
            if self.linear_dec.bias is not None:
                nn.init.zeros_(self.linear_dec.bias)
        # tiny near-origin init so the tangent-space linearization is well conditioned
        self.embedding.weight.data.uniform_(-0.001, 0.001)
        self.project_embeddings()

    def project_embeddings(self):
        with torch.no_grad():
            self.embedding.weight.data = StereographicUtils.project(
                self.embedding.weight.data, self.kappa
            )

    def zero_state(self, batch, device):
        return torch.zeros((batch, self.state_dim), device=device)

    def to_tangent(self, h):
        return StereographicUtils.logmap0(h, self.kappa)

    def encoder(self, x, h):
        e = self.dropout(self.embedding(x))
        h_next = StereographicUtils.project(
            StereographicUtils.add(h, e, self.kappa), self.kappa
        )
        return h_next, e

    def decoder(self, h):
        if self.linear_dec is None:
            raise RuntimeError(
                "this model is a ProductModel factor; decode through the "
                "product's shared decoder, not the factor's")
        return self.linear_dec(self.to_tangent(h))

    def forward(self, x, h=None):
        if h is None:
            h = self.zero_state(x.shape[0], x.device)
        h_next, e = self.encoder(x, h)
        return self.decoder(h_next), h_next, e

    def recover_increment(self, h_old, h_new):
        return StereographicUtils.add(-h_old, h_new, self.kappa)

    def dist2(self, a, b):
        return StereographicUtils.dist2(a, b, self.kappa)


# --------------------------------------------------------------------------- #
# Rotation / SO(n) Lie-group model: non-commutative orientation
# --------------------------------------------------------------------------- #
class RotationModel(nn.Module):
    """State is an orientation in SO(n); each event is a rotation applied on the
    left. Rotations do not commute, so (unlike the additive sum of axis counts)
    the state pins down the full orientation, not just its abelianization.
    """

    geometry = "rotation"

    def __init__(self, num_event_types, n=3, dropout_p=0.0):
        super().__init__()
        self.n = n
        self.n_gen = n * (n - 1) // 2          # dim of so(n)
        self.embedding_dim = self.n_gen
        self.state_dim = n * n                  # flattened rotation matrix
        self.embedding = nn.Embedding(num_event_types, self.n_gen)
        self.dropout = nn.Dropout(p=dropout_p)
        self.linear_dec = nn.Linear(self.state_dim, num_event_types)
        idx = torch.triu_indices(n, n, offset=1)
        self.register_buffer("triu_i", idx[0])
        self.register_buffer("triu_j", idx[1])
        self.reset_parameters()

    def reset_parameters(self):
        if self.linear_dec is not None:  # None when used as a ProductModel factor
            nn.init.xavier_uniform_(self.linear_dec.weight)
            if self.linear_dec.bias is not None:
                nn.init.zeros_(self.linear_dec.bias)
        self.embedding.weight.data.normal_(0.0, 0.1)

    def _rot(self, gen):
        """(B, n_gen) so(n) coordinates -> (B, n, n) rotation via matrix exp."""
        b = gen.shape[0]
        a = torch.zeros(b, self.n, self.n, device=gen.device, dtype=gen.dtype)
        a[:, self.triu_i, self.triu_j] = gen
        a = a - a.transpose(1, 2)
        return torch.matrix_exp(a)

    def zero_state(self, batch, device):
        return torch.eye(self.n, device=device).reshape(1, -1).repeat(batch, 1)

    def to_tangent(self, h):
        return h  # the flattened matrix is already a Euclidean feature for the decoder

    def encoder(self, x, h):
        step = self._rot(self.dropout(self.embedding(x)))      # (B, n, n)
        r_old = h.reshape(-1, self.n, self.n)
        r_new = torch.bmm(step, r_old)
        return r_new.reshape(h.shape[0], -1), step.reshape(h.shape[0], -1)

    def decoder(self, h):
        if self.linear_dec is None:
            raise RuntimeError(
                "this model is a ProductModel factor; decode through the "
                "product's shared decoder, not the factor's")
        return self.linear_dec(h)

    def forward(self, x, h=None):
        if h is None:
            h = self.zero_state(x.shape[0], x.device)
        h_next, e = self.encoder(x, h)
        return self.decoder(h_next), h_next, e

    def recover_increment(self, h_old, h_new):
        r_old = h_old.reshape(-1, self.n, self.n)
        r_new = h_new.reshape(-1, self.n, self.n)
        step = torch.bmm(r_new, r_old.transpose(1, 2))         # R_step = R_new R_old^-1
        return step.reshape(h_old.shape[0], -1)

    def dist2(self, a, b):
        return ((a - b) ** 2).sum(-1, keepdim=True)            # Frobenius^2 surrogate


# --------------------------------------------------------------------------- #
# Heisenberg group model: the signed (Levy) area / commutator
# --------------------------------------------------------------------------- #
class HeisenbergModel(nn.Module):
    """State is a point of the (2m+1)-dim Heisenberg group; the group product
    carries the symplectic area swept by the path in its central coordinate, a
    purely order-dependent quantity that the abelian sum discards.

    Product: ``(v1, z1) . (v2, z2) = (v1+v2, z1+z2 + 1/2 v1^T J v2)`` with ``J`` the
    canonical symplectic form; inverse ``(v, z)^-1 = (-v, -z)``.
    """

    geometry = "heisenberg"

    def __init__(self, num_event_types, m=1, dropout_p=0.0):
        super().__init__()
        self.m = m
        self.dim_v = 2 * m
        self.embedding_dim = 2 * m + 1
        self.state_dim = 2 * m + 1
        self.embedding = nn.Embedding(num_event_types, self.state_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.linear_dec = nn.Linear(self.state_dim, num_event_types)
        j = torch.zeros(self.dim_v, self.dim_v)
        for i in range(m):
            j[2 * i, 2 * i + 1] = 1.0
            j[2 * i + 1, 2 * i] = -1.0
        self.register_buffer("J", j)
        self.reset_parameters()

    def reset_parameters(self):
        if self.linear_dec is not None:  # None when used as a ProductModel factor
            nn.init.xavier_uniform_(self.linear_dec.weight)
            if self.linear_dec.bias is not None:
                nn.init.zeros_(self.linear_dec.bias)
        self.embedding.weight.data.normal_(0.0, 0.1)

    def _split(self, h):
        return h[:, : self.dim_v], h[:, self.dim_v:]

    def _prod(self, a, b):
        va, za = self._split(a)
        vb, zb = self._split(b)
        cross = 0.5 * ((va @ self.J) * vb).sum(-1, keepdim=True)
        return torch.cat([va + vb, za + zb + cross], dim=-1)

    def zero_state(self, batch, device):
        return torch.zeros((batch, self.state_dim), device=device)

    def to_tangent(self, h):
        return h

    def encoder(self, x, h):
        e = self.dropout(self.embedding(x))
        return self._prod(h, e), e

    def decoder(self, h):
        if self.linear_dec is None:
            raise RuntimeError(
                "this model is a ProductModel factor; decode through the "
                "product's shared decoder, not the factor's")
        return self.linear_dec(h)

    def forward(self, x, h=None):
        if h is None:
            h = self.zero_state(x.shape[0], x.device)
        h_next, e = self.encoder(x, h)
        return self.decoder(h_next), h_next, e

    def recover_increment(self, h_old, h_new):
        return self._prod(-h_old, h_new)                       # h_old^-1 . h_new = e

    def dist2(self, a, b):
        return ((a - b) ** 2).sum(-1, keepdim=True)


# --------------------------------------------------------------------------- #
# Product manifold: two structure types at once (e.g. hierarchy H x cycle S)
# --------------------------------------------------------------------------- #
class ProductModel(nn.Module):
    """A product of factor geometries. Each factor keeps its own event embedding
    and updates its own slice of the state independently; a single linear decoder
    reads the concatenated tangent representations. The reconstruction and
    consistency distances are the sum of the per-factor distances.
    """

    geometry = "product"

    def __init__(self, factors):
        super().__init__()
        self.factors = nn.ModuleList(factors)
        self.state_dim = sum(f.state_dim for f in factors)
        self._slices, off = [], 0
        for f in factors:
            self._slices.append((off, off + f.state_dim))
            off += f.state_dim
        num_events = factors[0].linear_dec.out_features
        self.linear_dec = nn.Linear(self.state_dim, num_events)
        # The product reads the concatenated tangents through ONE shared decoder, so each
        # factor's own decoder is dead weight: it never sees a gradient, yet it would be
        # counted in n_params and carried in the optimizer state. Drop it.
        for f in self.factors:
            f.linear_dec = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.linear_dec is not None:  # None when used as a ProductModel factor
            nn.init.xavier_uniform_(self.linear_dec.weight)
            if self.linear_dec.bias is not None:
                nn.init.zeros_(self.linear_dec.bias)
        for f in self.factors:
            f.reset_parameters()

    def _slice(self, h, k):
        a, b = self._slices[k]
        return h[:, a:b]

    def zero_state(self, batch, device):
        return torch.cat([f.zero_state(batch, device) for f in self.factors], dim=-1)

    def project_embeddings(self):
        for f in self.factors:
            if hasattr(f, "project_embeddings"):
                f.project_embeddings()

    def encoder(self, x, h):
        outs, incs = [], []
        for k, f in enumerate(self.factors):
            hn, e = f.encoder(x, self._slice(h, k))
            outs.append(hn)
            incs.append(e)
        return torch.cat(outs, dim=-1), torch.cat(incs, dim=-1)

    def decoder(self, h):
        tan = torch.cat(
            [f.to_tangent(self._slice(h, k)) for k, f in enumerate(self.factors)], dim=-1
        )
        return self.linear_dec(tan)

    def forward(self, x, h=None):
        if h is None:
            h = self.zero_state(x.shape[0], x.device)
        h_next, e = self.encoder(x, h)
        return self.decoder(h_next), h_next, e

    def recover_increment(self, h_old, h_new):
        return torch.cat(
            [f.recover_increment(self._slice(h_old, k), self._slice(h_new, k))
             for k, f in enumerate(self.factors)],
            dim=-1,
        )

    def dist2(self, a, b):
        total = 0.0
        for k, f in enumerate(self.factors):
            total = total + f.dist2(self._slice(a, k), self._slice(b, k))
        return total
