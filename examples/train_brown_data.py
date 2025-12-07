import torch
import pickle
import os
from event2vector import Event2Vec

# --- Hyperparameters ---
num_epochs = 128
embedding_dim = 64
lambda_reconstruction = 0.2
lambda_consistency = 0.2
dropout_p = 0.2
batch_size = 64
learning_rate = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Correctly locate the data file relative to the project root ---
# Assumes this script is in 'examples/' and the data file is in the root.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_file = os.path.join(project_root, 'brown_corpus_data.pkl')


# --- Data Preparation ---
print("Loading preprocessed Brown Corpus data...")
try:
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Data file '{data_file}' not found.")
    print("Please run 'python -m examples.prepare_brown_data' from the project's root directory first.")
    exit()

processed_sequences = data['processed_sequences']
event_2_idx = data['event_2_idx']
idx_2_event = data['idx_2_event']
num_event_types = len(event_2_idx)

model = Event2Vec(
    num_event_types=num_event_types,
    embedding_dim=embedding_dim,
    dropout_p=dropout_p,
    learning_rate=learning_rate,
    lambda_reconstruction=lambda_reconstruction,
    lambda_consistency=lambda_consistency,
    batch_size=batch_size,
    num_epochs=num_epochs,
    pad_sequences=True,
    device=device,
)

print(f"Training Event2Vec model on Brown Corpus ({len(processed_sequences)} sequences)...")
model.fit(processed_sequences, verbose=True)
print("Training complete.")

# --- Save the trained model ---
output_model_path = os.path.join(project_root, 'event2vec_brown_corpus.model')
torch.save({
    'state_dict': model.model.state_dict(),
    'event_2_idx': event_2_idx,
    'idx_2_event': idx_2_event,
    'embedding_dim': embedding_dim,
}, output_model_path)

print(f"Brown Corpus model saved to {output_model_path}")


