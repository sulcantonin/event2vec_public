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
inv = {v: k for k, v in event_2_idx.items()}

# 3) Initialize and train the scikit-style estimator
#    Setting pad_sequences=True lets us vectorize across padded batches for speed.
estimator = Event2Vec(
    num_event_types=len(event_types),
    embedding_dim=8,
    dropout_p=0.1,
    learning_rate=5e-3,
    lambda_reconstruction=0.2,
    lambda_consistency=0.2,
    batch_size=32,
    num_epochs=128,
    pad_sequences=True,
)
estimator.fit(processed_sequences, verbose=True)
torch_model = estimator.model

# 4) Use the learned representation for a short sequence via the new transform API
seq = torch.tensor([
    event_2_idx['START'], event_2_idx['A'], event_2_idx['C']
], dtype=torch.long)
sequence_embedding = estimator.transform([seq], as_numpy=False)[0]
print('Sequence embedding (START-A-C):', sequence_embedding.numpy())

# 5) Nearest tokens using the built-in most_similar helper (gensim-style)
nearest = estimator.most_similar(positive=seq, topn=3)
print('Most similar tokens:', [(inv[i], score) for i, score in nearest])

with torch.no_grad():
    # 6) Next-event distribution from the current state (decoder matches the paper)
    seq_for_decoder = sequence_embedding.to(estimator.device)
    logits = torch_model.decoder(seq_for_decoder.unsqueeze(0))    # [1, V]
    probs = torch.softmax(logits, dim=-1).squeeze(0)
    top = torch.topk(probs, k=3)
    print('Top-3 next events:', [ (inv[i.item()], float(probs[i])) for i in top.indices ])

# 7) Visualization: PCA of token embeddings + sequence embedding
#    Expect the red star (sequence) to lie near the point labeled 'C'.
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