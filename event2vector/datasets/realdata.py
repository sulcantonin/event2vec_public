"""Loaders for the public real-data benchmarks (R2 RetailRocket, R3 Instacart,
R4 StackOverflow/MOOC, R5 Synthea).

These reduce to a common shape: a table of (group_id, ordered mark, optional
per-group label) becomes per-group event-mark sequences. ``load_event_log``
implements that reduction; the dataset adapters wire up column names and point
to the public download when the data is absent (several require Kaggle access).
"""
import os
import re
import sys
from collections import defaultdict
from typing import Optional

import torch

_DOWNLOAD_HINTS = {
    "retailrocket": "Kaggle: retailrocket/ecommerce-dataset (events.csv). "
                    "Place under <data_dir>/retailrocket/events.csv.",
    "instacart": "Kaggle: instacart-market-basket-analysis (order_products__*.csv, "
                 "orders.csv, products.csv). Place under <data_dir>/instacart/.",
    "stackoverflow": "StackOverflow badge sequences (e.g. the RMTPP/Du et al. release). "
                     "Place a CSV with user_id,badge,timestamp under <data_dir>/stackoverflow/.",
    "synthea": "Generate locally with Synthea (https://github.com/synthetichealth/synthea); "
               "point at the output/csv directory (conditions.csv).",
}


def _require(path: str, name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} data not found at {path}. {_DOWNLOAD_HINTS.get(name, '')}"
        )


def load_event_log(
    rows,
    group_key,
    mark_key,
    time_key,
    label_key: Optional[str] = None,
    min_len: int = 2,
    max_vocab: Optional[int] = None,
    min_count: int = 1,
) -> dict:
    """Turn an iterable of dict-like rows into the uniform dataset record.

    ``rows`` are grouped by ``group_key``, ordered by ``time_key``, and each
    group becomes a sequence of ``mark_key`` tokens. If ``label_key`` is given,
    its (first-seen) value per group becomes a per-sequence label.
    ``max_vocab``/``min_count`` cap the vocabulary (rare marks are dropped).
    """
    from collections import Counter

    groups = defaultdict(list)
    group_label = {}
    counts = Counter()
    for r in rows:
        g = r[group_key]
        groups[g].append((r[time_key], r[mark_key]))
        counts[r[mark_key]] += 1
        if label_key is not None and g not in group_label:
            group_label[g] = r[label_key]

    keep = {m for m, c in counts.items() if c >= min_count}
    if max_vocab is not None:
        keep = {m for m, _ in counts.most_common(max_vocab)} & keep
    vocab = {m: i for i, m in enumerate(sorted(keep))}

    sequences, seq_labels = [], []
    for g, evs in groups.items():
        evs.sort(key=lambda t: t[0])
        seq = [m for _, m in evs if m in vocab]
        if len(seq) >= min_len:
            sequences.append(seq)
            if label_key is not None:
                seq_labels.append(group_label.get(g))

    processed = [torch.tensor([vocab[m] for m in s], dtype=torch.long) for s in sequences]
    record = {
        "sequences": sequences,
        "processed_sequences": processed,
        "vocab": vocab,
        "inv_vocab": {i: m for m, i in vocab.items()},
        "labels": None,
        "graph": None,
    }
    if label_key is not None:
        lab_names = sorted(set(l for l in seq_labels if l is not None))
        lab_id = {l: i for i, l in enumerate(lab_names)}
        record["sequence_labels"] = [lab_id.get(l) for l in seq_labels]
        record["label_names"] = lab_names
    return record


def _read_csv(path):
    import csv

    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            yield row


def load_retailrocket(data_dir: str = "data", max_vocab: int = 5000, **kw) -> dict:
    """R2: per-visitor clickstreams (mark = 'event:itemid'); funnel/counterfactual."""
    path = os.path.join(data_dir, "retailrocket", "events.csv")
    _require(path, "retailrocket")
    rows = ({"visitorid": r["visitorid"],
             "mark": f'{r["event"]}:{r["itemid"]}',
             "ts": int(r["timestamp"]),
             "buy": 1 if r["event"] == "transaction" else 0}
            for r in _read_csv(path))
    return load_event_log(rows, "visitorid", "mark", "ts", label_key="buy",
                          max_vocab=max_vocab, min_count=kw.get("min_count", 5))


def load_synthea(data_dir: str = "data", max_vocab: int = 2000, **kw) -> dict:
    """R5: per-patient ordered clinical conditions (SNOMED descriptions)."""
    path = os.path.join(data_dir, "synthea", "conditions.csv")
    _require(path, "synthea")
    rows = ({"patient": r["PATIENT"], "mark": r["DESCRIPTION"], "ts": r["START"]}
            for r in _read_csv(path))
    return load_event_log(rows, "patient", "mark", "ts",
                          max_vocab=max_vocab, min_count=kw.get("min_count", 5))


def load_stackoverflow(data_dir: str = "data", **kw) -> dict:
    """R4: per-user badge-award sequences."""
    path = os.path.join(data_dir, "stackoverflow", "badges.csv")
    _require(path, "stackoverflow")
    rows = ({"user": r["user_id"], "mark": r["badge"], "ts": r["timestamp"]}
            for r in _read_csv(path))
    return load_event_log(rows, "user", "mark", "ts", min_count=kw.get("min_count", 1))


# CAMEO root event types (01--20) with representative Goldstein conflict-cooperation
# scores (-10 hostile .. +10 cooperative). Used as ground-truth ordering for the
# escalation probe (does the learned geometry recover the escalation scale?).
CAMEO_ROOT_GOLDSTEIN = {
    "01": 0.0, "02": 3.0, "03": 4.0, "04": 1.0, "05": 3.5, "06": 6.0, "07": 7.0,
    "08": 5.0, "09": -2.0, "10": -5.0, "11": -2.0, "12": -4.0, "13": -6.5,
    "14": -6.5, "15": -7.2, "16": -4.0, "17": -7.0, "18": -9.0, "19": -10.0,
    "20": -10.0,
}
_DOWNLOAD_HINTS["gdelt"] = (
    "GDELT 1.0 event files (tab-separated, no header). Download from "
    "http://data.gdeltproject.org/events/ and place one or more *.export.CSV "
    "under <data_dir>/gdelt/.")
_DOWNLOAD_HINTS["sec8k"] = (
    "SEC 8-K item codes per filing as cik,item,date CSV (build from EDGAR "
    "full-index + header parsing). Place under <data_dir>/sec8k/filings.csv.")


def _goldstein_class(score: float) -> int:
    return 0 if score < -2.0 else (2 if score > 2.0 else 1)  # conflict / neutral / cooperate


def load_gdelt(data_dir: str = "data", min_count: int = 1,
               max_files: Optional[int] = None, subdir: Optional[str] = None,
               **kw) -> dict:
    """R3 (politics): dyadic CAMEO event sequences with the Goldstein escalation
    scale as per-event-type ground truth.

    Sequences are events between a country dyad over time; the mark is the CAMEO
    root code (01--20). ``labels`` maps each event-id to a 3-way Goldstein class
    (conflict / neutral / cooperation), the ground truth for the escalation probe.

    Accepts GDELT 1.0 daily exports either unpacked (``*.CSV``) or still zipped
    (``*.CSV.zip``, read in place -- a day is ~7 MB zipped but ~23 MB unpacked, so
    unzipping a long range is a needless several GB), and searches ``<data_dir>/gdelt``
    recursively so a nested download directory works.

    ``max_files`` caps how many days are read, in chronological (filename) order.
    Use it to fix the window: the recovered correlation depends strongly on which days
    are included (7 days gives rho~0.35, 60 days rho~0.66), so the window is part of the
    protocol rather than an incidental detail.

    ``subdir`` restricts the search to one directory under ``<data_dir>/gdelt``. Give it
    whenever more than one download lives there: the recursive search would otherwise
    merge e.g. ``gdelt_2025/`` and ``gdelt_2026/`` into a single pile, and since files
    sort by basename the ``max_files`` cut would land somewhere non-obvious.
    """
    import glob
    import zipfile

    folder = os.path.join(data_dir, "gdelt")
    if subdir:
        folder = os.path.join(folder, subdir)
    plain = sorted(glob.glob(os.path.join(folder, "**", "*.CSV"), recursive=True) +
                   glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True))
    zipped = sorted(glob.glob(os.path.join(folder, "**", "*.zip"), recursive=True))
    files = sorted(plain + zipped, key=lambda p: os.path.basename(p))
    if not files:
        _require(folder, "gdelt")
    if max_files is not None:
        files = files[:max_files]
    # GDELT 1.0 column indices: 1 SQLDATE, 7 Actor1CountryCode,
    # 17 Actor2CountryCode, 28 EventRootCode, 30 GoldsteinScale.
    def _lines(fp):
        if fp.lower().endswith(".zip"):
            with zipfile.ZipFile(fp) as z:
                for member in z.namelist():
                    if not member.lower().endswith((".csv",)):
                        continue
                    with z.open(member) as f:
                        for raw in f:
                            yield raw.decode("utf8", "replace")
        else:
            with open(fp, encoding="utf8", errors="replace") as f:
                yield from f

    # Intern the three retained fields. They are drawn from tiny alphabets -- 20 CAMEO
    # roots, one date per file, ~50k dyads -- but a naive parse allocates a fresh string
    # per row, so a full year (~9.5M rows) costs ~1.5 GB resident instead of ~0.6 GB.
    intern = sys.intern

    def rows():
        for fp in files:
            for line in _lines(fp):
                c = line.rstrip("\n").split("\t")
                if len(c) < 31 or not c[7] or not c[17] or not c[28]:
                    continue
                yield {"dyad": intern(f"{c[7]}->{c[17]}"),
                       "mark": intern(c[28].zfill(2)),
                       "ts": intern(c[1])}

    record = load_event_log(rows(), "dyad", "mark", "ts", min_count=min_count)
    record["labels"] = {
        i: _goldstein_class(CAMEO_ROOT_GOLDSTEIN.get(code, 0.0))
        for code, i in record["vocab"].items()
    }
    record["label_names"] = ["conflict", "neutral", "cooperation"]
    return record


def _sic_division(sic) -> Optional[str]:
    """Map a 4-digit SIC code to its broad industry division."""
    try:
        s = int(sic)
    except (ValueError, TypeError):
        return None
    if s <= 999:
        return "Agriculture"
    if s <= 1499:
        return "Mining"
    if s <= 1799:
        return "Construction"
    if s <= 3999:
        return "Manufacturing"
    if s <= 4999:
        return "Transport/Utilities"
    if s <= 5199:
        return "Wholesale"
    if s <= 5999:
        return "Retail"
    if s <= 6799:
        return "Finance"
    if s <= 8999:
        return "Services"
    return "Other"


def load_sec_8k(data_dir: str = "data", min_count: int = 20, label: str = "sector",
                **kw) -> dict:
    """Finance: per-company 8-K item-code trajectories (a real corporate
    life-path). Expects `sec8k/filings.csv` (cik,item,date) and optionally
    `sec8k/labels.csv` (cik,distress,sic) — both from `scripts/build_sec8k.py`.

    ``label='sector'`` (default) gives each company its SIC industry division as a
    non-leaky downstream-probe target (does the pattern of material events reflect
    the industry?); ``label='distress'`` gives the binary bankruptcy/delisting
    flag (rare; note it leaks if the defining items remain in the sequence)."""
    path = os.path.join(data_dir, "sec8k", "filings.csv")
    _require(path, "sec8k")
    labels_path = os.path.join(data_dir, "sec8k", "labels.csv")
    meta = {}
    if os.path.exists(labels_path):
        for r in _read_csv(labels_path):
            meta[r["cik"]] = {"distress": int(r.get("distress", 0) or 0),
                              "sic": r.get("sic", "")}

    def _label_for(cik):
        m = meta.get(cik)
        if not m:
            return None
        return m["distress"] if label == "distress" else _sic_division(m["sic"])

    def rows():
        for r in _read_csv(path):
            row = {"cik": r["cik"], "mark": r["item"], "ts": r["date"]}
            if meta:
                row["label"] = _label_for(r["cik"])
            yield row

    return load_event_log(rows(), "cik", "mark", "ts",
                          label_key="label" if meta else None, min_count=min_count)


# --------------------------------------------------------------------------- #
# Chemistry: MoleculeNet SMILES property classification (competitiveness benchmark)
# --------------------------------------------------------------------------- #
# Standard SMILES atom/bond tokenizer (two-char atoms, bracketed atoms, ring
# numbers, bonds, branches) — Schwaller et al. 2018.
_SMILES_TOKEN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
)
_MOLNET = {  # name -> (csv filename, smiles column, label column)
    "bbbp": ("BBBP.csv", "smiles", "p_np"),
    "bace": ("bace.csv", "mol", "Class"),
    "clintox": ("clintox.csv", "smiles", "CT_TOX"),
    "hiv": ("HIV.csv", "smiles", "HIV_active"),
}
_DOWNLOAD_HINTS["moleculenet"] = (
    "MoleculeNet CSVs from DeepChem: "
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/<FILE> "
    "(BBBP.csv, bace.csv, clintox.csv, HIV.csv). Place under <data_dir>/moleculenet/.")


def tokenize_smiles(smiles: str) -> list:
    """Tokenize a SMILES string into atom/bond tokens."""
    return _SMILES_TOKEN.findall(smiles)


# --------------------------------------------------------------------------- #
# Knowledge graph: WN18RR relation-path walks (the kappa-stereographic showcase)
# --------------------------------------------------------------------------- #
_WN18RR_BASE = ("https://raw.githubusercontent.com/villmow/"
                "datasets_knowledge_embedding/master/WN18RR/text")
_DOWNLOAD_HINTS["wn18rr"] = (
    "WN18RR triples (head\\trelation\\ttail). Auto-downloaded from "
    f"{_WN18RR_BASE}/<split>.txt to <data_dir>/wn18rr/; or place them there yourself.")


def _maybe_download(url: str, path: str) -> None:
    import urllib.request

    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(url, path)


def load_wn18rr(data_dir: str = "data", num_walks: int = 8000, walk_len: int = 8,
                seed: int = 0, splits=("train",), download: bool = True) -> dict:
    """Knowledge-graph relation walks: the tokens are the 11 WordNet relations,
    a sequence is the relation path of a random walk on the entity graph, and its
    composition is the net relation between the walk's endpoints.

    This is the curvature showcase. WN18RR is strongly hierarchical (hypernym
    trees), so it is a domain where the curvature of the recurrence should matter:
    the additive endpoint is an order-invariant bag of relations, while a curved
    (kappa<0, hyperbolic) recurrence can respect the tree structure. Sweep
    ``geometry='stereographic'`` over kappa to read the best-fitting curvature off
    real data (see ``scripts/run_kg_curvature.py``)."""
    import random
    from collections import defaultdict

    folder = os.path.join(data_dir, "wn18rr")
    triples = []
    for sp in splits:
        path = os.path.join(folder, f"{sp}.txt")
        if not os.path.exists(path):
            if download:
                _maybe_download(f"{_WN18RR_BASE}/{sp}.txt", path)
            else:
                _require(path, "wn18rr")
        with open(path) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 3:
                    triples.append((parts[0], parts[1], parts[2]))

    adj = defaultdict(list)
    rels = set()
    for h, r, t in triples:
        adj[h].append((r, t))
        rels.add(r)
    vocab = {r: i for i, r in enumerate(sorted(rels))}
    entities = [e for e in adj if adj[e]]

    rng = random.Random(seed)
    sequences = []
    for _ in range(num_walks):
        e = rng.choice(entities)
        walk = []
        for _ in range(walk_len):
            if not adj[e]:
                break
            r, t = rng.choice(adj[e])
            walk.append(r)
            e = t
        if len(walk) >= 2:
            sequences.append(walk)

    processed = [torch.tensor([vocab[r] for r in s], dtype=torch.long) for s in sequences]
    return {
        "sequences": sequences,
        "processed_sequences": processed,
        "vocab": vocab,
        "inv_vocab": {i: r for r, i in vocab.items()},
        "labels": None,
        "graph": None,
        "n_relations": len(vocab),
        "n_triples": len(triples),
    }


def load_moleculenet(name: str = "bbbp", data_dir: str = "data",
                     max_vocab: Optional[int] = None, **kw) -> dict:
    """Chemistry: a molecule is a SMILES token sequence; the per-molecule binary
    property (e.g. blood-brain-barrier penetration) is the sequence label. This
    is the competitiveness benchmark (frozen-probe ROC-AUC vs. mol2vec/Word2Vec,
    GRU/LSTM/Transformer, per-parameter)."""
    from collections import Counter

    name = name.lower()
    if name not in _MOLNET:
        raise ValueError(f"unknown MoleculeNet dataset {name}; choose from {list(_MOLNET)}")
    fname, smi_col, lab_col = _MOLNET[name]
    path = os.path.join(data_dir, "moleculenet", fname)
    if not os.path.exists(path) and kw.get("download", True):
        try:
            _maybe_download(
                f"https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{fname}", path
            )
        except Exception:
            pass
    _require(path, "moleculenet")

    smiles, labels = [], []
    for r in _read_csv(path):
        s, lab = r.get(smi_col, ""), r.get(lab_col, "")
        if not s or lab == "" or lab is None:
            continue
        toks = tokenize_smiles(s)
        if len(toks) < 2:
            continue
        smiles.append(toks)
        labels.append(int(float(lab)))

    counts = Counter(t for toks in smiles for t in toks)
    keep = ({t for t, _ in counts.most_common(max_vocab)} if max_vocab else set(counts))
    vocab = {t: i for i, t in enumerate(sorted(keep))}
    seqs, seq_labels = [], []
    for toks, y in zip(smiles, labels):
        kept = [t for t in toks if t in vocab]
        if len(kept) >= 2:
            seqs.append(kept)
            seq_labels.append(y)
    processed = [torch.tensor([vocab[t] for t in s], dtype=torch.long) for s in seqs]
    return {
        "sequences": seqs,
        "processed_sequences": processed,
        "vocab": vocab,
        "inv_vocab": {i: t for t, i in vocab.items()},
        "labels": None,
        "sequence_labels": seq_labels,
        "label_names": [0, 1],
        "graph": None,
    }
