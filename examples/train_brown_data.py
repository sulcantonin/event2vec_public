import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
import pickle
import os
from event2vector import EuclideanModel

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

# --- Model Initialization ---
model = EuclideanModel(num_event_types, embedding_dim, dropout_p=dropout_p).to(device)
loss_function = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# --- Training Loop ---
print(f"Training Euclidean Event2Vec Model on Brown Corpus ({len(processed_sequences)} sequences)...")
for epoch in range(num_epochs):
    # Shuffle the data for each epoch
    shuffled_sequences = random.sample(processed_sequences, len(processed_sequences))
    total_loss = 0
    num_batches = 0
    
    # Create an iterator for the batches, and wrap it with tqdm
    batch_iterator = range(0, len(shuffled_sequences), batch_size)
    pbar = tqdm(batch_iterator, desc=f"Epoch {epoch+1}/{num_epochs}")

    for i in pbar:
        # Create the batch by slicing the shuffled sequence list
        batch = shuffled_sequences[i:i+batch_size]
        batch_loss = 0
        
        for seq_indices in batch:
            if len(seq_indices) < 2:
                continue
            
            seq_indices = seq_indices.to(device)
            h = torch.zeros((1, embedding_dim), device=device)
            sequence_loss = 0

            for j in range(len(seq_indices) - 1):
                x = seq_indices[j].unsqueeze(0)
                target = seq_indices[j + 1].unsqueeze(0)
                h_old = h.detach()

                model.train()
                y1, h1, e_curr1 = model.forward(x, h_old)
                y2, h2, e_curr2 = model.forward(x, h_old)

                prediction_loss = loss_function(y1.view(1, -1), target)
                h_reconstructed = h1 - e_curr1
                reconstruction_loss = mse_loss(h_reconstructed, h_old)
                consistency_loss = mse_loss(h1, h2)

                combined_loss = (prediction_loss +
                                 (lambda_reconstruction * reconstruction_loss) +
                                 (lambda_consistency * consistency_loss))
                sequence_loss += combined_loss
                h = h1.detach()

            # Make sure to handle the case where a sequence has only one valid step
            if (len(seq_indices) - 1) > 0:
                batch_loss += sequence_loss / (len(seq_indices) - 1)

        if len(batch) > 0:
            avg_batch_loss = batch_loss / len(batch)
            optimizer.zero_grad()
            avg_batch_loss.backward()
            optimizer.step()
            total_loss += avg_batch_loss.item()
            num_batches += 1
            pbar.set_postfix(avg_loss=total_loss/num_batches)

print("Training complete.")

# --- Save the trained model ---
output_model_path = os.path.join(project_root, 'event2vec_brown_corpus.model')
torch.save({
    'state_dict': model.state_dict(),
    'event_2_idx': event_2_idx,
    'idx_2_event': idx_2_event,
    'embedding_dim': embedding_dim,
}, output_model_path)

print(f"Brown Corpus model saved to {output_model_path}")


