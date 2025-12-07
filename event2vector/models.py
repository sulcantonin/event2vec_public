import random
from typing import Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

EPS = 1e-5  # Small constant for numerical stability of hyperbolic space


class EuclideanModel(nn.Module):
    """
    A recurrent model for learning event sequence representations in Euclidean space.
    The model uses a simple additive update rule, where the representation of a
    sequence is the sum of its constituent event embeddings.
    """

    def __init__(self, num_event_types, embedding_dim, dropout_p=0.2, max_norm=10.0):
        super(EuclideanModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.embedding = nn.Embedding(num_event_types, embedding_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.linear_dec = nn.Linear(embedding_dim, num_event_types)

    def encoder(self, x, h=None, clipping=True):
        if h is None:
            h = torch.zeros((x.shape[0], self.embedding_dim), device=x.device)
        e = self.embedding(x)
        h_next = self.dropout(e + h)

        if clipping:
            norm = torch.norm(h_next, p=2, dim=1, keepdim=True)
            clip_coef = self.max_norm / (norm + 1e-6)
            clip_coef = torch.min(clip_coef, torch.ones_like(clip_coef))
            h_next = h_next * clip_coef

        return h_next, e

    def decoder(self, h):
        return self.linear_dec(h)

    def forward(self, x, h=None):
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
        return num / (den + EPS)

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

    def __init__(self, num_event_types, embedding_dim, dropout_p, c=1.0):
        super(HyperbolicModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.c = c
        self.embedding = nn.Embedding(num_event_types, embedding_dim)
        self.embedding.weight.data.uniform_(-0.001, 0.001)
        self.dropout = nn.Dropout(p=dropout_p)
        self.linear_dec = nn.Linear(embedding_dim, num_event_types)

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
        return h_next, e

    def decoder(self, h):
        h_tangent = HyperbolicUtils.log_map_origin(h, self.c)
        return self.linear_dec(h_tangent)

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros((x.shape[0], self.embedding_dim), device=x.device)
        h_next, e = self.encoder(x, h)
        return self.decoder(h_next), h_next, e


SequenceType = Union[torch.Tensor, Sequence[int]]


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
        lambda_reconstruction: float = 0.2,
        lambda_consistency: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        num_epochs: int = 50,
        pad_sequences: bool = False,
        pad_value: int = 0,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if num_event_types <= 0:
            raise ValueError("num_event_types must be positive.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if num_epochs <= 0:
            raise ValueError("num_epochs must be positive.")

        self.num_event_types = num_event_types
        self.embedding_dim = embedding_dim
        self.geometry = geometry.lower()
        self.dropout_p = dropout_p
        self.max_norm = max_norm
        self.curvature = curvature
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_consistency = lambda_consistency
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.pad_sequences = pad_sequences
        self.pad_value = pad_value
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        else:
            raise ValueError("geometry must be either 'euclidean' or 'hyperbolic'.")
        return model.to(self.device)

    def _reset_parameters(self):
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        self.model.apply(_init_weights)
        if hasattr(self.model, "project_embeddings"):
            self.model.project_embeddings()

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

        self._reset_parameters()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
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

    def _train_epoch(self, sequences, optimizer, use_padding: bool) -> float:
        shuffled = sequences.copy()
        random.shuffle(shuffled)
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
        losses = []
        for seq in batch:
            if seq.numel() < 2:
                continue
            seq = seq.to(self.device)
            h = torch.zeros((1, self.embedding_dim), device=self.device)
            sequence_loss = torch.tensor(0.0, device=self.device)
            valid_steps = 0
            for t in range(seq.numel() - 1):
                x = seq[t].unsqueeze(0)
                target = seq[t + 1].unsqueeze(0)
                h_old = h.detach()
                y1, h1, e_curr1 = self.model(x, h_old)
                y2, h2, e_curr2 = self.model(x, h_old)

                prediction_loss = self.ce_loss(y1, target)
                h_reconstructed = h1 - e_curr1
                reconstruction_loss = self.mse_loss(h_reconstructed, h_old)
                consistency_loss = self.mse_loss(h1, h2)

                combined_loss = (
                    prediction_loss
                    + self.lambda_reconstruction * reconstruction_loss
                    + self.lambda_consistency * consistency_loss
                )
                sequence_loss = sequence_loss + combined_loss
                h = h1.detach()
                valid_steps += 1

            if valid_steps:
                losses.append(sequence_loss / valid_steps)

        if not losses:
            return None

        batch_loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        self._project_if_needed()
        return float(batch_loss.item())

    def _train_batch_padded(self, batch, optimizer):
        sequences = [seq for seq in batch if seq.numel() >= 2]
        if not sequences:
            return None
        tensors = [seq.to(self.device) for seq in sequences]
        lengths = torch.tensor([tensor.size(0) for tensor in tensors], device=self.device)
        padded = pad_sequence(
            tensors, batch_first=True, padding_value=self.pad_value
        )

        h = torch.zeros((len(sequences), self.embedding_dim), device=self.device)
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
            y2, h2, e_curr2 = self.model(x, h_old)

            prediction_loss = self.ce_loss(y1, target)
            h_reconstructed = h1 - e_curr1
            reconstruction_loss = self.mse_loss(h_reconstructed, h_old)
            consistency_loss = self.mse_loss(h1, h2)

            combined_loss = (
                prediction_loss
                + self.lambda_reconstruction * reconstruction_loss
                + self.lambda_consistency * consistency_loss
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
                outputs.append(torch.zeros(self.embedding_dim))
                continue
            seq = seq.to(self.device)
            h = torch.zeros((1, self.embedding_dim), device=self.device)
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
            h = torch.zeros((len(batch), self.embedding_dim), device=self.device)
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
        if hasattr(self.model, "project_embeddings"):
            self.model.project_embeddings()

    def _check_is_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Event2Vec instance is not fitted yet.")

