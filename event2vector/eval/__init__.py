"""Evaluation harness: train models (Event2Vec + baselines) across seeds and
compute the experiment metrics into JSON-able result dicts."""
from .harness import build_model, run_full_suite

__all__ = ["build_model", "run_full_suite"]
