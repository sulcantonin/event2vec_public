import torch
import torch.nn as nn
import random
import pickle

# --- Models ---

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


# --- Data Generation ---

def generate_sequences(event_types, event_transitions, initial, terminal, num_seq, max_length):
    """
    Generates synthetic sequences of events based on a state transition graph.
    This function simulates realistic life trajectories by performing a guided
    random walk on the graph.
    """
    sequences = []
    for _ in range(num_seq):
        seq = [initial]
        while seq[-1] != terminal and len(seq) < max_length:
            current_event = seq[-1]
            transitions = event_transitions.get(current_event, [])
            if not transitions or random.random() < 0.1:
                next_event = random.choice([e for e in event_types if e != current_event])
            else:
                events, probs = zip(*transitions)
                next_event = random.choices(events, weights=probs, k=1)[0]
            seq.append(next_event)
        if seq[-1] != terminal:
            seq.append(terminal)
        sequences.append(seq)
    return sequences


def get_sequences(event_types, event_transitions, initial, terminal, num_seq, max_seq_length, generate_new, prefix):
    """
    Generates and preprocesses sequences, or loads them from a file.
    """
    if generate_new:
        event_2_idx = {event: idx for idx, event in enumerate(event_types)}
        idx_2_event = {idx: event for idx, event in enumerate(event_types)}
        sequences = generate_sequences(event_types, event_transitions, initial, terminal, num_seq, max_seq_length)
        processed_sequences = [torch.tensor([event_2_idx[s] for s in seq], dtype=torch.long) for seq in sequences]
        with open(f'{prefix}_training_data.pkl', 'wb') as f:
            pickle.dump({'sequences': sequences,
                         'processed_sequences': processed_sequences,
                         'event_2_idx': event_2_idx,
                         'idx_2_event': idx_2_event}, f)
    else:
        with open(f'{prefix}_training_data.pkl', 'rb') as f:
            data = pickle.load(f)
        sequences = data['sequences']
        processed_sequences = data['processed_sequences']
        event_2_idx = data['event_2_idx']
        idx_2_event = data['idx_2_event']
    return sequences, processed_sequences, event_2_idx, idx_2_event

