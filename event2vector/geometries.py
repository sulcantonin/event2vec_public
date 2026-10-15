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
affine (Aff+(d))   x -> exp(M_e) x + v_e        events that act on the history
                   on the left                  (shrink / stretch / rotate it) as
                                                well as add to it; M_e = 0 is
                                                exactly the Euclidean sum
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


def _matrix_exp(a):
    """torch.matrix_exp that also works for MPS tensors: MPS has no matrix_exp
    kernel (pytorch/pytorch#141287), so it is computed on the CPU and moved back.
    Both transfers are differentiable."""
    if a.device.type == "mps":
        return torch.matrix_exp(a.cpu()).to(a.device)
    return torch.matrix_exp(a)


def _solve(a, b):
    """torch.linalg.solve, computed on the CPU for MPS tensors (as above)."""
    if a.device.type == "mps":
        return torch.linalg.solve(a.cpu(), b.cpu()).to(a.device)
    return torch.linalg.solve(a, b)


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
        return _matrix_exp(a)

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
# Affine group model: events that act on the history, not only add to it
# --------------------------------------------------------------------------- #
class AffineModel(nn.Module):
    """State is an orientation-preserving affine map ``x -> A x + b`` of R^d,
    stored flat as ``[vec(A), b]`` and starting at the identity. Each event is the
    affine map ``x -> exp(M_e) x + v_e`` applied on the left::

        A_t = exp(M_e) A_{t-1}            b_t = exp(M_e) b_{t-1} + v_e

    ``b`` is the sequence's *point* (the image of the origin). With every
    ``M_e = 0`` it is exactly the Euclidean sum of the ``v_e``, so this geometry
    contains event2vec's additive model and starts training from it.

    The linear part lets an event act on what came before -- shrink, stretch or
    rotate it -- so the same event contributes differently after different
    histories, while the model stays a group: every step is invertible (``exp``
    of any matrix is), the reconstruction loss recovers it exactly, and any
    subsequence composes into one affine map (:meth:`compose`). Its singular
    values (:meth:`override_strength`) say how strongly that subsequence
    overrides the history: values below 1 erase the history along a direction.

    Args:
        embedding_dim: dimension ``d`` of the point ``b`` (2 for a planar map).
            Each event has ``d + d*d`` parameters (``v_e`` and ``M_e``).
        readout: ``"point"`` decodes the next event from ``b`` only (the plotted
            point); ``"full"`` also reads ``A``.
        max_log_scale: if set, ``M_e`` entries are squashed to
            ``(-max_log_scale, max_log_scale)``, bounding how much one event can
            stretch or shrink the state (each step's condition number is at most
            ``exp(2 * d * max_log_scale)``). ``None`` leaves ``M_e`` free.
    """

    geometry = "affine"

    def __init__(self, num_event_types, embedding_dim=2, dropout_p=0.0,
                 readout="point", max_log_scale=None):
        super().__init__()
        if readout not in ("point", "full"):
            raise ValueError("readout must be 'point' or 'full'.")
        d = embedding_dim
        self.d = d
        self.point_dim = d
        self.embedding_dim = d + d * d        # generator: [v_e, vec(M_e)]
        self.state_dim = d * d + d            # state: [vec(A), b]
        self.readout = readout
        self.max_log_scale = max_log_scale
        self.embedding = nn.Embedding(num_event_types, self.embedding_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.linear_dec = nn.Linear(d if readout == "point" else self.state_dim,
                                    num_event_types)
        self.reset_parameters()

    def reset_parameters(self):
        if self.linear_dec is not None:  # None when used as a ProductModel factor
            nn.init.xavier_uniform_(self.linear_dec.weight)
            if self.linear_dec.bias is not None:
                nn.init.zeros_(self.linear_dec.bias)
        d = self.d
        with torch.no_grad():
            nn.init.normal_(self.embedding.weight[:, :d], 0.0, 0.02)  # v_e as Euclidean
            self.embedding.weight[:, d:].zero_()                      # M_e = 0: A_e = I

    # -- group elements ----------------------------------------------------- #
    def _maps(self, gen):
        """(B, d + d*d) generators -> (A_e (B, d, d), v_e (B, d))."""
        d = self.d
        v, m = gen[:, :d], gen[:, d:].reshape(-1, d, d)
        if self.max_log_scale is not None:
            c = self.max_log_scale
            m = c * torch.tanh(m / c)
        return _matrix_exp(m), v

    def _split(self, h):
        d = self.d
        return h[:, : d * d].reshape(-1, d, d), h[:, d * d:]

    @staticmethod
    def _join(a, b):
        return torch.cat([a.reshape(a.shape[0], -1), b], dim=-1)

    def zero_state(self, batch, device):
        a = torch.eye(self.d, device=device).expand(batch, -1, -1)
        return self._join(a, torch.zeros(batch, self.d, device=device))

    def point(self, h):
        """The plotted point ``b`` of a state (or of ``transform``'s output)."""
        return h[..., self.d * self.d:]

    def to_tangent(self, h):
        return self.point(h) if self.readout == "point" else h

    # -- recurrence ---------------------------------------------------------- #
    def encoder(self, x, h):
        a_e, v_e = self._maps(self.dropout(self.embedding(x)))
        a_old, b_old = self._split(h)
        a_new = torch.bmm(a_e, a_old)
        b_new = torch.bmm(a_e, b_old.unsqueeze(-1)).squeeze(-1) + v_e
        return self._join(a_new, b_new), self._join(a_e, v_e)

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
        """g_e = g_new . g_old^-1:  A_e = A_new A_old^-1,  v_e = b_new - A_e b_old."""
        a_old, b_old = self._split(h_old)
        a_new, b_new = self._split(h_new)
        # A_e A_old = A_new  <=>  A_old^T A_e^T = A_new^T (solve, don't invert)
        a_e = _solve(a_old.transpose(1, 2), a_new.transpose(1, 2)).transpose(1, 2)
        v_e = b_new - torch.bmm(a_e, b_old.unsqueeze(-1)).squeeze(-1)
        return self._join(a_e, v_e)

    def dist2(self, a, b):
        return ((a - b) ** 2).sum(-1, keepdim=True)            # Frobenius^2 on [A, b]

    # -- interpretation -------------------------------------------------------- #
    @torch.no_grad()
    def compose(self, event_ids):
        """One affine map (A, v) for a whole subsequence: applying it to any state
        equals feeding the events one by one (exact; it is a group product)."""
        dev = self.embedding.weight.device
        a, v = torch.eye(self.d, device=dev), torch.zeros(self.d, device=dev)
        for i in event_ids:
            a_e, v_e = self._maps(self.embedding.weight[int(i)].unsqueeze(0))
            a, v = a_e[0] @ a, a_e[0] @ v + v_e[0]
        return a, v

    @torch.no_grad()
    def override_strength(self, event_ids):
        """Singular values of the subsequence's linear part, largest first:
        1 = the history passes through unchanged, < 1 = erased along that
        direction, > 1 = amplified."""
        a, _ = self.compose(event_ids)
        return torch.linalg.svdvals(a)


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
