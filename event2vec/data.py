import random
import torch
import pickle

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
    Generates and preprocesses sequences for synthetic data, or loads them from a file.
    """
    data_file = f'{prefix}_training_data.pkl'
    if generate_new:
        event_2_idx = {event: idx for idx, event in enumerate(event_types)}
        idx_2_event = {idx: event for idx, event in enumerate(event_types)}
        sequences = generate_sequences(event_types, event_transitions, initial, terminal, num_seq, max_seq_length)
        processed_sequences = [torch.tensor([event_2_idx[s] for s in seq], dtype=torch.long) for seq in sequences]
        with open(data_file, 'wb') as f:
            pickle.dump({'sequences': sequences,
                         'processed_sequences': processed_sequences,
                         'event_2_idx': event_2_idx,
                         'idx_2_event': idx_2_event}, f)
    else:
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        sequences = data['sequences']
        processed_sequences = data['processed_sequences']
        event_2_idx = data['event_2_idx']
        idx_2_event = data['idx_2_event']
    return sequences, processed_sequences, event_2_idx, idx_2_event

