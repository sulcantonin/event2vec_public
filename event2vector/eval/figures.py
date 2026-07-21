"""Matplotlib figure generation for the Event2Vec experiments.

Functions consume the ``out["models"]`` dict from ``harness.run_full_suite``
and save vector PDFs (paper-ready).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _ensure_dir(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def plot_order_sensitivity(models: dict, path: str, title="Order sensitivity vs. curvature"):
    """A2 hero figure: hyperbolic PS as a curve over curvature; Euclidean and
    neural/bag baselines as horizontal reference lines."""
    _ensure_dir(path)
    hyper, refs = [], {}
    for label, m in models.items():
        spec = m["spec"]
        mean, std = m["order_ps"]["mean"], m["order_ps"]["std"]
        if mean is None:
            continue
        if spec.get("geometry") == "hyperbolic":
            hyper.append((spec.get("curvature"), mean, std))
        else:
            refs[label] = (spec, mean, std)
    hyper.sort(key=lambda t: t[0])

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    if hyper:
        cs, ms, ss = zip(*hyper)
        ax.errorbar(cs, ms, yerr=ss, marker="o", capsize=3, lw=2.2,
                    color="#7b1fa2", label="Hyperbolic Event2Vec")
        ax.set_xscale("log")
        xlim = (min(cs) * 0.6, max(cs) * 1.6)
    else:
        xlim = (0.05, 3.0)
    ax.set_xlim(xlim)

    colors = {"euclidean": "#1565c0", "gru": "#2e7d32", "lstm": "#558b2f",
              "transformer": "#ef6c00", "word2vec": "#9e9e9e"}
    for label, (spec, mean, std) in refs.items():
        key = spec.get("geometry") if spec["kind"] == "event2vec" else spec["kind"]
        col = colors.get(key, "#444")
        ax.axhline(mean, ls="--", color=col, lw=1.6, label=f"{label} = {mean:.2f}")
        ax.fill_between(xlim, mean - std, mean + std, color=col, alpha=0.07)

    ax.axhline(0.0, color="black", lw=0.8, alpha=0.4)
    ax.set_xlabel("curvature $c$  (hyperbolic Event2Vec)")
    ax.set_ylabel("Permutation Sensitivity  PS")
    ax.set_title(title)
    ax.legend(fontsize=7.5, loc="best", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_additivity_decay(models: dict, path: str,
                          title="Additivity: cos(recurrent state, $\\Sigma$ embeddings)"):
    """A1 figure: additivity cosine vs sequence length for each Event2Vec variant."""
    _ensure_dir(path)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    plotted = False
    for label, m in models.items():
        d = m.get("additivity_decay")
        if not d:
            continue
        ys = d["mean"]
        es = d["std"]
        ax.errorbar(d["lengths"], ys, yerr=es, marker="o", capsize=3, lw=2, label=label)
        plotted = True
    if plotted:
        ax.set_xscale("log", base=2)
    ax.set_xlabel("sequence length $L$")
    ax.set_ylabel("additivity cosine")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(1.0, color="black", lw=0.8, alpha=0.4)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
