"""Brown Corpus loader returning the uniform dataset record.

Refactors the logic of ``examples/prepare_brown_data.py`` into a function so the
Brown experiment (real, credential-free, with free Part-of-Speech labels) runs
through the same harness as every other dataset. POS tags become the per-event
label used by the E1 frozen probe.
"""
import os
import pickle
from collections import Counter, defaultdict

import torch

MIN_TOKEN_FREQUENCY = 128
MIN_SEQUENCE_LENGTH = 5


def _download_nltk():
    import nltk

    for path, rid in {
        "corpora/brown": "brown",
        "corpora/stopwords": "stopwords",
        "taggers/universal_tagset": "universal_tagset",
    }.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(rid)


def load_brown(min_freq: int = MIN_TOKEN_FREQUENCY, min_len: int = MIN_SEQUENCE_LENGTH,
               cache_path: str = "brown_corpus_data.pkl", rebuild: bool = False) -> dict:
    """Load (or build) the Brown dataset record.

    Returns ``{sequences, processed_sequences, vocab, inv_vocab, labels,
    label_names, graph=None}`` where ``labels`` maps event-id -> POS-tag-id.
    """
    if os.path.exists(cache_path) and not rebuild:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
    else:
        import nltk

        _download_nltk()
        stop = set(nltk.corpus.stopwords.words("english"))
        counts = Counter(w.lower() for w in nltk.corpus.brown.words() if w.isalpha())
        vocab_words = {w for w, c in counts.items() if c >= min_freq and w not in stop}

        word_tags = defaultdict(list)
        for w, tag in nltk.corpus.brown.tagged_words(tagset="universal"):
            lw = w.lower()
            if lw in vocab_words:
                word_tags[lw].append(tag)
        event_labels = {w: Counter(t).most_common(1)[0][0] for w, t in word_tags.items()}

        sequences = []
        for sent in nltk.corpus.brown.sents():
            filt = [w.lower() for w in sent if w.lower() in vocab_words]
            if len(filt) >= min_len:
                sequences.append(filt)

        final_vocab = sorted({w for s in sequences for w in s})
        event_2_idx = {w: i for i, w in enumerate(final_vocab)}
        data = {
            "sequences": sequences,
            "event_2_idx": event_2_idx,
            "idx_2_event": {i: w for w, i in event_2_idx.items()},
            "event_labels": {w: t for w, t in event_labels.items() if w in event_2_idx},
        }
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    vocab = data["event_2_idx"]
    inv_vocab = data["idx_2_event"]
    sequences = data["sequences"]
    processed = [torch.tensor([vocab[w] for w in s], dtype=torch.long) for s in sequences]

    pos_names = sorted(set(data["event_labels"].values()))
    pos_id = {p: i for i, p in enumerate(pos_names)}
    labels = {vocab[w]: pos_id[t] for w, t in data["event_labels"].items() if w in vocab}

    return {
        "sequences": sequences,
        "processed_sequences": processed,
        "vocab": vocab,
        "inv_vocab": inv_vocab,
        "labels": labels,
        "label_names": pos_names,
        "graph": None,
    }
