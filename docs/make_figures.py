"""Figures for docs/visual-guide.md, computed by the library itself.

    python docs/make_figures.py

Trains Event2Vec on a small synthetic patient data set with known rules and writes
every picture of the guide to docs/figures/<topic>-{light,dark}.svg. The numbers
quoted in the guide are printed at the end and saved to docs/figures/facts.json.
"""
import json, os, random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from event2vector import Event2Vec

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

THEMES = {
    "light": dict(ink="#1f2328", muted="#59636e", grid="#d1d9e0", panel="#f6f8fa",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"], other="#a3a29c"),
    "dark": dict(ink="#e6edf3", muted="#9198a1", grid="#3d444d", panel="#151b23",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#2ea043"], other="#7d8590"),
}

# ------------------------------------------------------------------ toy data
EVENTS = ["checkup", "flu", "sprain", "recovery", "prediabetes", "diabetes", "retinopathy", "kidney disease"]
ID = {e: i for i, e in enumerate(EVENTS)}
GROUP = {"checkup": 0, "flu": 1, "sprain": 1, "recovery": 1, "prediabetes": 2, "diabetes": 2,
         "retinopathy": 3, "kidney disease": 3}
GROUP_NAMES = ["routine", "acute episode", "metabolic", "complication"]


def patient(rng):
    """A history with known rules: acute episodes resolve, diabetes leads to complications."""
    seq = ["checkup"]
    chronic = rng.random() < 0.45
    stage = 0
    for _ in range(rng.randint(5, 10)):
        r = rng.random()
        if r < 0.25:
            seq += [rng.choice(["flu", "sprain"]), "recovery"]
        elif r < 0.45:
            seq.append("checkup")
        elif chronic:
            if stage == 0:
                seq.append("prediabetes"); stage = 1
            elif stage == 1:
                seq.append("diabetes"); stage = 2
            elif stage == 2:
                seq.append(rng.choice(["retinopathy", "kidney disease"])); stage = 3
    return [ID[e] for e in seq]


rng = random.Random(0)
data = [patient(rng) for _ in range(3000)]
torch.manual_seed(0)
model = Event2Vec(num_event_types=len(EVENTS), embedding_dim=16, gate="vector", max_norm=1e6,
                  pad_sequences=True, num_epochs=40, random_state=0, device="cpu").fit(data)
plain2d = Event2Vec(num_event_types=len(EVENTS), embedding_dim=2, max_norm=1e6,
                    pad_sequences=True, num_epochs=40, random_state=0, device="cpu").fit(data)
facts = {}

HISTORY = ["checkup", "flu", "recovery", "prediabetes", "diabetes", "checkup"]
H_IDS = [ID[e] for e in HISTORY]
TARGET = ID["retinopathy"]


def theme_rc(t):
    return {"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": t["muted"], "axes.labelcolor": t["ink"],
            "text.color": t["ink"], "xtick.color": t["muted"], "ytick.color": t["muted"],
            "axes.spines.top": False, "axes.spines.right": False, "axes.facecolor": "none", "figure.facecolor": "none",
            "svg.fonttype": "path", "legend.frameon": False}


def save(name, draw):
    for theme, t in THEMES.items():
        with plt.rc_context(theme_rc(t)):
            fig = draw(t)
            fig.savefig(os.path.join(OUT, f"{name}-{theme}.svg"), transparent=True, bbox_inches="tight", pad_inches=0.05)
            plt.close(fig)


# ------------------------------------------------------------------ 0. the big picture
def big_picture(t):
    fig, ax = plt.subplots(figsize=(9.5, 2.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    def box(x, y, w, h, title, sub, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=t["panel"], ec=col, lw=1.4))
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=11, fontweight="bold", color=t["ink"])
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=8.5, color=t["muted"])
    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=t["muted"], lw=1.2))
    s = t["series"]
    box(0.1, 1.0, 1.9, 1.0, "sequences", "lists of event ids", t["muted"])
    box(2.7, 1.0, 1.9, 1.0, "Event2Vec.fit", "one vector per event", s[0])
    box(5.4, 2.05, 2.0, 0.85, "transform", "a state per history", s[2])
    box(5.4, 1.08, 2.0, 0.85, "predict", "next-event probabilities", s[3])
    box(5.4, 0.1, 2.0, 0.85, "explain", "exact contributions", s[1])
    box(8.1, 1.08, 1.8, 0.85, "what if", "counterfactuals", s[4])
    arrow(2.0, 1.5, 2.7, 1.5); arrow(4.6, 1.6, 5.4, 2.45); arrow(4.6, 1.5, 5.4, 1.5); arrow(4.6, 1.4, 5.4, 0.55)
    arrow(7.4, 1.5, 8.1, 1.5)
    return fig


# ------------------------------------------------------------------ 1. a history is a sum
E2 = plain2d.model.embedding.weight.detach().numpy()
A_SEQ, B_SEQ = ["checkup", "prediabetes", "diabetes"], ["flu", "recovery"]


def history_sum(t):
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    s = t["series"]
    OFF = {"checkup": (-30, -16), "prediabetes": (8, 0), "diabetes": (-58, -4), "flu": (8, -2), "recovery": (6, -14)}
    def arrow(p0, p1, col, lw=2.0, ls="-", alpha=1.0):
        ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.28",
                    color=col, lw=lw, ls=ls, alpha=alpha, shrinkA=0, shrinkB=0))
    pts, cur = [np.zeros(2)], np.zeros(2)
    for seq, col in ((A_SEQ, s[0]), (B_SEQ, s[1])):
        for e in seq:
            nxt = cur + E2[ID[e]]
            arrow(cur, nxt, col)
            ax.annotate(e, (cur + nxt) / 2, xytext=OFF[e], textcoords="offset points", fontsize=9.5, color=col)
            cur = nxt; pts.append(cur.copy())
        if seq is A_SEQ:
            hA = cur.copy()
    hAB = cur.copy(); hB = E2[[ID[e] for e in B_SEQ]].sum(0)
    arrow(np.zeros(2), hB, t["muted"], lw=1.6)                 # H(B) from the start
    arrow(hA, hA + hB, t["muted"], lw=1.2, ls=(0, (3, 2)))       # the same vector, appended to H(A)
    pts += [hB]
    for p, lab, off in ((np.zeros(2), "start", (8, 0)), (hA, "H(A)", (-34, -8)), (hB, "H(B)", (6, 2)),
                        (hAB, "H(A then B)\n= H(A) + H(B)", (8, -24))):
        ax.scatter(*p, s=30, color=t["ink"], zorder=5)
        ax.annotate(lab, p, xytext=off, textcoords="offset points", fontsize=9.5, color=t["ink"])
    P = np.array(pts); lo, hi = P.min(0), P.max(0); span = hi - lo
    ax.set_xlim(lo[0] - 0.35 * span[0], hi[0] + 0.45 * span[0]); ax.set_ylim(lo[1] - 0.1 * span[1], hi[1] + 0.12 * span[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
    facts["sum_error"] = float(np.abs(plain2d.transform([A_SEQ_IDS + B_SEQ_IDS])[0] - hAB).max())
    facts["flu_recovery_net"] = float(np.linalg.norm(hB)); facts["flu_norm"] = float(np.linalg.norm(E2[ID["flu"]]))
    return fig


A_SEQ_IDS, B_SEQ_IDS = [ID[e] for e in A_SEQ], [ID[e] for e in B_SEQ]

# ------------------------------------------------------------------ 2. predicting
probs = model.counterfactual(H_IDS)["probs"]
top = np.argsort(-probs)[:6]
facts["next_event"] = [(EVENTS[i], float(probs[i])) for i in top]


def predict(t):
    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    ys = np.arange(len(top))[::-1]
    ax.barh(ys, probs[top], color=[t["series"][GROUP[EVENTS[i]]] for i in top], height=0.62)
    for y, i in zip(ys, top):
        ax.text(probs[i] + 0.008, y, f"{probs[i]:.2f}", va="center", fontsize=9, color=t["muted"])
    ax.set_yticks(ys); ax.set_yticklabels([EVENTS[i] for i in top]); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False); ax.set_xlabel("P(next event | history)")
    ax.grid(axis="x", color=t["grid"], lw=0.6); ax.set_axisbelow(True)
    return fig


# ------------------------------------------------------------------ 3. explaining
att = model.attribute(H_IDS, target=TARGET)
facts["attribution"] = dict(bias=float(att["baseline"]), logit=float(att["logit"]),
                            contributions=[(HISTORY[k], float(c)) for k, c in enumerate(att["contributions"])],
                            error=float(abs(att["baseline"] + att["contributions"].sum() - att["logit"])))
facts["rank_causes"] = [(EVENTS[e], float(v)) for e, v in model.rank_causes(H_IDS, TARGET)]


def explain(t):
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    lanes = np.arange(len(HISTORY))[::-1]; x = att["baseline"]
    lo = min(x, x + np.cumsum(att["contributions"]).min()); hi = max(att["logit"], x + np.cumsum(att["contributions"]).max())
    top_, bot = lanes[0] + 0.42, lanes[-1] - 0.42
    ax.plot([x, x], [bot, top_], color=t["muted"], ls=(0, (3, 3)), lw=1.0)
    for k, (y, phi) in enumerate(zip(lanes, att["contributions"])):
        ax.barh(y, phi, left=x, height=0.62, color=t["series"][GROUP[HISTORY[k]]], zorder=3)
        if k + 1 < len(HISTORY):
            ax.plot([x + phi] * 2, [y - 0.31, lanes[k + 1] + 0.31], color=t["muted"], lw=0.6)
        ax.text(hi + 0.25 * (hi - lo) + 0.4, y, f"{phi:+.2f}", va="center", ha="right", fontsize=9, color=t["muted"])
        x += phi
    ax.plot([att["logit"]] * 2, [bot, top_], color=t["ink"], lw=1.4)
    ax.text(att["baseline"], top_ + 0.1, "bias", ha="center", va="bottom", fontsize=9)
    ax.text(att["logit"], top_ + 0.1, "logit", ha="center", va="bottom", fontsize=9)
    ax.set_xlim(lo - 0.5, hi + 0.25 * (hi - lo) + 0.45); ax.set_ylim(-0.6, lanes[0] + 1.1)
    ax.set_yticks(lanes); ax.set_yticklabels(HISTORY); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False); ax.set_xlabel("logit of retinopathy")
    ax.grid(axis="x", color=t["grid"], lw=0.6); ax.set_axisbelow(True)
    return fig


# ------------------------------------------------------------------ 4. what if
REMOVE = HISTORY.index("diabetes")
cf = model.counterfactual(H_IDS, remove=[REMOVE])
show = [ID[e] for e in ("retinopathy", "kidney disease", "diabetes", "checkup", "flu")]
facts["counterfactual"] = {EVENTS[i]: (float(cf["probs"][i]), float(cf["probs_counterfactual"][i])) for i in show}


def what_if(t):
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    ys = np.arange(len(show))[::-1]
    ax.barh(ys + 0.18, cf["probs"][show], height=0.34, color=t["series"][0], label="history as observed")
    ax.barh(ys - 0.18, cf["probs_counterfactual"][show], height=0.34, color=t["other"], label="without diabetes")
    ax.set_yticks(ys); ax.set_yticklabels([EVENTS[i] for i in show]); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False); ax.set_xlabel("P(next event)")
    ax.grid(axis="x", color=t["grid"], lw=0.6); ax.set_axisbelow(True); ax.legend(loc="lower right", fontsize=9)
    return fig


# ------------------------------------------------------------------ 5. events that forget
gates = model.event_gates().mean(axis=1)
facts["gates"] = {EVENTS[i]: float(gates[i]) for i in range(len(EVENTS))}


def forget(t):
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    order = np.argsort(gates); ys = np.arange(len(order))[::-1]
    ax.barh(ys, gates[order], color=[t["series"][GROUP[EVENTS[i]]] for i in order], height=0.62)
    ax.axvline(0, color=t["muted"], lw=0.8)
    ax.set_yticks(ys); ax.set_yticklabels([EVENTS[i] for i in order]); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False); ax.set_xlabel("mean gate (negative: the event shrinks the history)")
    ax.grid(axis="x", color=t["grid"], lw=0.6); ax.set_axisbelow(True)
    return fig


# ------------------------------------------------------------------ 6. order
states = {}
pairs = np.zeros((len(EVENTS), len(EVENTS)))
for a in range(len(EVENTS)):
    for b in range(len(EVENTS)):
        hab, hba = model.transform([[a, b], [b, a]])
        pairs[a, b] = np.linalg.norm(hab - hba)
facts["order_max"] = (EVENTS[int(np.unravel_index(pairs.argmax(), pairs.shape)[0])],
                      EVENTS[int(np.unravel_index(pairs.argmax(), pairs.shape)[1])], float(pairs.max()))
facts["order_plain"] = float(np.abs(plain2d.transform([[ID["flu"], ID["diabetes"]]])[0] - plain2d.transform([[ID["diabetes"], ID["flu"]]])[0]).max())


def order(t):
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", [t["panel"], t["series"][0]])
    im = ax.imshow(pairs, cmap=cmap)
    ax.set_xticks(range(len(EVENTS))); ax.set_xticklabels(EVENTS, rotation=45, ha="right")
    ax.set_yticks(range(len(EVENTS))); ax.set_yticklabels(EVENTS)
    ax.set_xlabel("second event"); ax.set_ylabel("first event")
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.outline.set_visible(False)
    cb.set_label("distance between the two orders")
    return fig


save("pipeline", big_picture)
save("sum", history_sum)
save("predict", predict)
save("explain", explain)
save("whatif", what_if)
save("forget", forget)
save("order", order)
json.dump(facts, open(os.path.join(OUT, "facts.json"), "w"), indent=1)
print(json.dumps(facts, indent=1))
