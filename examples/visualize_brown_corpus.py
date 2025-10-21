import torch
import pickle
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from openTSNE import TSNE
from gensim.models import Word2Vec
from sklearn.metrics import silhouette_score, davies_bouldin_score
from tqdm import tqdm

from event2vec import EuclideanModel

# --- Configuration ---
EVENT2VEC_MODEL_FILE = 'event2vec_brown_corpus.model'
DATA_FILE = 'brown_corpus_data.pkl'
NUM_SAMPLES_PER_STRUCTURE = 1000
TSNE_PERPLEXITY = 30
TSNE_METRIC = 'euclidean'
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Define the POS tag sequences to analyze, as in the paper
SEQUENCES_TO_PLOT = {
    "AT-JJ-NN": ['DET', 'ADJ', 'NOUN'], # Article(Determiner)-Adjective-Noun
    "NP-VB-RB": ['NOUN', 'VERB', 'ADV'], # Noun-Verb-Adverb
    "PP-MD-VB": ['PRON', 'VERB', 'VERB'], # Pronoun-ModalVerb-Verb (approximated)
    "IN-AT-NN": ['ADP', 'DET', 'NOUN'], # Preposition(Adposition)-Article-Noun
}

def generate_sequence_vectors(pos_sequences, embeddings, words_by_pos, event_2_idx, num_samples):
    """
    Generates embedded vectors for random phrases matching specific POS structures.
    """
    all_vectors = []
    all_labels = []

    print("Generating composite sequence vectors...")
    for label, sequence in tqdm(pos_sequences.items()):
        sequence_vectors = []
        for _ in range(num_samples):
            h_cumulative = np.zeros(embeddings.shape[1], dtype=np.float32)
            possible = True
            for pos_tag in sequence:
                if not words_by_pos[pos_tag]:
                    print(f"Warning: No vocabulary words found for POS tag '{pos_tag}'. Skipping structure.")
                    possible = False
                    break
                random_word = random.choice(words_by_pos[pos_tag])
                word_idx = event_2_idx[random_word]
                h_cumulative += embeddings[word_idx]
            
            if possible:
                sequence_vectors.append(h_cumulative)
        
        if sequence_vectors:
            all_vectors.append(np.array(sequence_vectors))
            all_labels.extend([label] * len(sequence_vectors))

    return np.vstack(all_vectors), all_labels


def evaluate_clusters(vectors, labels, model_name):
    """Calculates and prints clustering evaluation metrics."""
    print(f"\n--- Quantitative Evaluation for {model_name} ---")
    silhouette = silhouette_score(vectors, labels, metric=TSNE_METRIC)
    db_score = davies_bouldin_score(vectors, labels)
    print(f"Silhouette Score: {silhouette:.4f} (Higher is better)")
    print(f"Davies-Bouldin Score: {db_score:.4f} (Lower is better)")


def plot_tsne(tsne_results, labels, title, filename):
    """Creates and saves a high-quality t-SNE plot."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 16))
    
    palette = sns.color_palette("husl", n_colors=len(np.unique(labels)))
    
    sns.scatterplot(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        hue=labels,
        palette=palette,
        alpha=0.8,
        s=50,
        edgecolor=None,
        ax=ax
    )
    
    ax.set_title(title, fontsize=24, pad=20)
    ax.tick_params(axis='both', labelsize=14)
    legend = ax.legend(title="POS Sequence", fontsize=14, title_fontsize=16)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {filename}")
    plt.close()


def main():
    """Main execution function."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # --- Load Data ---
    data_path = os.path.join(project_root, DATA_FILE)
    print(f"Loading data from {data_path}...")
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Data file not found. Please run 'prepare_brown_data.py' first.")
        return
        
    sequences = data['sequences']
    event_labels = data['event_labels']
    event_2_idx = data['event_2_idx']

    # --- Load Event2Vec Model ---
    model_path = os.path.join(project_root, EVENT2VEC_MODEL_FILE)
    print(f"Loading Event2Vec model from {model_path}...")
    try:
        e2v_data = torch.load(model_path, map_location='cpu')
    except FileNotFoundError:
        print(f"Error: Model file not found. Please run 'train_brown_corpus.py' first.")
        return

    e2v_model = EuclideanModel(len(event_2_idx), e2v_data['embedding_dim'])
    e2v_model.load_state_dict(e2v_data['state_dict'])
    e2v_model.eval()
    e2v_embeddings = e2v_model.embedding.weight.detach().numpy()

    # --- Train Word2Vec Baseline ---
    print("\nTraining Word2Vec baseline model...")
    w2v_model = Word2Vec(sentences=sequences, vector_size=e2v_data['embedding_dim'], window=5, min_count=1, workers=4)
    w2v_embeddings = np.zeros_like(e2v_embeddings)
    for word, idx in event_2_idx.items():
        if word in w2v_model.wv:
            w2v_embeddings[idx] = w2v_model.wv[word]
    
    # --- Prepare word lookup by POS tag ---
    words_by_pos = defaultdict(list)
    for word, pos_tag in event_labels.items():
        if word in event_2_idx:
            words_by_pos[pos_tag].append(word)

    # --- Process and Evaluate Event2Vec ---
    e2v_vectors, e2v_labels = generate_sequence_vectors(SEQUENCES_TO_PLOT, e2v_embeddings, words_by_pos, event_2_idx, NUM_SAMPLES_PER_STRUCTURE)
    evaluate_clusters(e2v_vectors, e2v_labels, "Event2Vec")
    
    print("\nRunning t-SNE for Event2Vec...")
    tsne_e2v = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, metric=TSNE_METRIC, random_state=RANDOM_SEED, n_jobs=-1)
    e2v_tsne_results = tsne_e2v.fit(e2v_vectors)
    plot_tsne(e2v_tsne_results, e2v_labels, "Event2Vec: Grammatical Structure Embeddings", "event2vec_composition_viz.png")

    # --- Process and Evaluate Word2Vec ---
    w2v_vectors, w2v_labels = generate_sequence_vectors(SEQUENCES_TO_PLOT, w2v_embeddings, words_by_pos, event_2_idx, NUM_SAMPLES_PER_STRUCTURE)
    evaluate_clusters(w2v_vectors, w2v_labels, "Word2Vec")

    print("\nRunning t-SNE for Word2Vec...")
    tsne_w2v = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, metric=TSNE_METRIC, random_state=RANDOM_SEED, n_jobs=-1)
    w2v_tsne_results = tsne_w2v.fit(w2v_vectors)
    plot_tsne(w2v_tsne_results, w2v_labels, "Word2Vec: Grammatical Structure Embeddings", "word2vec_composition_viz.png")

if __name__ == '__main__':
    main()

