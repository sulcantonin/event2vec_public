"""Dataset loaders returning a uniform record.

Each loader returns a dict with at least ``sequences`` (raw token sequences),
``processed_sequences`` (1-D ``LongTensor``s), ``vocab``/``inv_vocab``, and,
where available, ``labels`` (per-event-id ground-truth label) and ``graph``
(a transition graph for ground-truth structural evaluation).
"""
from .arithmetic import make_arithmetic
from .brown import load_brown
from .decay import make_decay
from .groups import make_group
from .lifepath import (
    EVENT_TYPES,
    INITIAL,
    LIFEPATH_DOMAINS,
    LIFEPATH_TRANSITIONS,
    TERMINAL,
    make_lifepath,
)
from .realdata import (
    load_event_log,
    load_gdelt,
    load_moleculenet,
    load_retailrocket,
    load_sec_8k,
    load_stackoverflow,
    load_synthea,
    load_wn18rr,
    tokenize_smiles,
)

__all__ = [
    "make_arithmetic",
    "make_lifepath",
    "LIFEPATH_TRANSITIONS",
    "LIFEPATH_DOMAINS",
    "EVENT_TYPES",
    "INITIAL",
    "TERMINAL",
    "make_decay",
    "make_group",
    "load_brown",
    "load_event_log",
    "load_gdelt",
    "load_moleculenet",
    "tokenize_smiles",
    "load_retailrocket",
    "load_sec_8k",
    "load_synthea",
    "load_stackoverflow",
    "load_wn18rr",
]
