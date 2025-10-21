import nltk
import pickle
import os
from collections import Counter, defaultdict
import torch
from tqdm import tqdm

# --- Configuration ---
MIN_TOKEN_FREQUENCY = 128
MIN_SEQUENCE_LENGTH = 5

def download_nltk_resources():
    """Downloads necessary NLTK resources if they are not already present."""
    resources = {
        'corpora/brown': 'brown',
        'tokenizers/punkt': 'punkt',
        'corpora/stopwords': 'stopwords',
        'taggers/universal_tagset': 'universal_tagset'
    }
    for path, resource_id in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"NLTK resource '{resource_id}' not found. Downloading...")
            nltk.download(resource_id)

def main():
    """
    Preprocesses the Brown Corpus by filtering for frequent words, removing stopwords,
    and creating the final data structures for training.
    """
    print("Ensuring NLTK resources are available...")
    download_nltk_resources()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_file = os.path.join(project_root, 'brown_corpus_data.pkl')
    
    # --- 1. Vocabulary Filtering (based on frequency) ---
    print("Step 1: Building vocabulary from Brown Corpus...")
    stopwords_en = set(nltk.corpus.stopwords.words('english'))
    all_words = [
        w.lower() for w in tqdm(nltk.corpus.brown.words(), desc="Counting words")
        if w.isalpha()
    ]
    
    word_counts = Counter(all_words)
    
    # Keep words that meet the minimum frequency threshold
    vocabulary = {
        word for word, count in word_counts.items() 
        if count >= MIN_TOKEN_FREQUENCY and word not in stopwords_en
    }
    print(f"Vocabulary size after frequency filtering (>= {MIN_TOKEN_FREQUENCY}) and stopword removal: {len(vocabulary)} words.")

    # --- 2. Determine Most Frequent POS Tag for each word ---
    print("\nStep 2: Determining the most frequent POS tag for each word...")
    word_pos_tags = defaultdict(list)
    tagged_words = nltk.corpus.brown.tagged_words(tagset='universal')

    for word, tag in tqdm(tagged_words, desc="Mapping words to POS tags"):
        lower_word = word.lower()
        if lower_word in vocabulary:
            word_pos_tags[lower_word].append(tag)
            
    event_labels = {}
    for word, tags in tqdm(word_pos_tags.items(), desc="Finding most common tag"):
        most_common_tag = Counter(tags).most_common(1)[0][0]
        event_labels[word] = most_common_tag

    # --- 3. Filter Sentences ---
    print("\nStep 3: Filtering sentences...")
    all_sents = nltk.corpus.brown.sents()
    sequences = []
    for sent in tqdm(all_sents, desc="Processing sentences"):
        filtered_sent = [
            w.lower() for w in sent 
            if w.lower() in vocabulary
        ]
        # Keep sentences that are long enough after filtering
        if len(filtered_sent) >= MIN_SEQUENCE_LENGTH:
            sequences.append(filtered_sent)
            
    print(f"Kept {len(sequences)} sentences with length >= {MIN_SEQUENCE_LENGTH}.")

    # --- 4. Create Final Data Structures ---
    print("\nStep 4: Creating final data structures...")
    # Final vocabulary might be smaller if some words never appeared in long-enough sentences
    final_vocabulary = sorted(list({word for seq in sequences for word in seq}))
    event_2_idx = {word: i for i, word in enumerate(final_vocabulary)}
    idx_2_event = {i: word for i, word in enumerate(final_vocabulary)}

    # Filter event_labels to only include words in the final vocabulary
    final_event_labels = {k: v for k, v in event_labels.items() if k in final_vocabulary}

    processed_sequences = [
        torch.tensor([event_2_idx[word] for word in seq], dtype=torch.long)
        for seq in tqdm(sequences, desc="Converting sequences to tensors")
    ]

    # --- 5. Save to File ---
    data_to_save = {
        'sequences': sequences,
        'processed_sequences': processed_sequences,
        'event_2_idx': event_2_idx,
        'idx_2_event': idx_2_event,
        'event_labels': final_event_labels
    }

    with open(output_file, 'wb') as f:
        pickle.dump(data_to_save, f)
        
    print(f"\nPreprocessed Brown Corpus data saved to '{output_file}'")
    print(f"   - Total sequences: {len(sequences)}")
    print(f"   - Vocabulary size: {len(final_vocabulary)}")

if __name__ == '__main__':
    main()

