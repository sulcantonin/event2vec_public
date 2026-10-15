"""Regenerate the README figures from the library itself.

    python images/make_readme_figures.py --font /path/to/Archivo-ExpandedBold.ttf

Writes
  images/logo-light.svg, images/logo-dark.svg  the event2vec mark: three cells and
      the wordmark (letters converted to outlines, so no font is needed to view it)
  images/geometries.png  the same three events composed by each geometry, a·b·c
      (blue) against c·b·a (red), computed by the library's own model classes;
      where the two endpoints land apart, the geometry remembers order.

The wordmark font is Archivo (SIL OFL), Bold, Expanded width, as on event2vec.com.
Without --font the logo is skipped and only the figure is drawn.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402

from event2vector import (AffineModel, EuclideanModel, HeisenbergModel,  # noqa: E402
                          HyperbolicModel, ProductModel, RotationModel,
                          StereographicModel)

HERE = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, PANEL, RULE = "#15181d", "#687180", "#f4f6f8", "#c9ced6"
BLUE, RED = "#2a7bcf", "#d6453a"
CELLS = {"light": ("#f0be2c", "#2a7bcf", "#d6453a"),   # event2vec.com, light theme
         "dark": ("#f3c94a", "#5ea6ee", "#f06a5d")}    # and dark theme
WORD = {"light": "#15181d", "dark": "#e8ebef"}
ABC, CBA = [0, 1, 2], [2, 1, 0]


# --------------------------------------------------------------------------- #
# logo
# --------------------------------------------------------------------------- #
def write_logo(font, theme, path):
    """Three 9x9 cells (2 px gaps) and the wordmark, as on event2vec.com."""
    fp = FontProperties(fname=font)
    size = 64
    tp = TextPath((0, 0), "event2vec", size=size, prop=fp)
    bb = tp.get_extents()
    # event2vec.com: 9 px cells, 2 px apart, 10 px before 18 px text
    cell, gap, pad = size * 9 / 18, size * 2 / 18, size * 10 / 18
    x0 = 3 * cell + 2 * gap + pad  # wordmark starts after the cells
    height = bb.y1 - min(bb.y0, 0) + 0.3 * size
    base = height - 0.15 * size  # baseline (SVG y grows downward)
    width = x0 + bb.x1 + 0.1 * size

    def path_d(p):
        d, verts, codes = [], p.vertices, p.codes
        i = 0
        while i < len(verts):
            code = codes[i]
            x, y = verts[i][0] + x0, base - verts[i][1]
            if code == 1:
                d.append(f"M{x:.2f} {y:.2f}")
                i += 1
            elif code == 2:
                d.append(f"L{x:.2f} {y:.2f}")
                i += 1
            elif code == 3:  # quadratic: control, end
                x2, y2 = verts[i + 1][0] + x0, base - verts[i + 1][1]
                d.append(f"Q{x:.2f} {y:.2f} {x2:.2f} {y2:.2f}")
                i += 2
            elif code == 4:  # cubic: two controls, end
                pts = [(verts[i + k][0] + x0, base - verts[i + k][1]) for k in range(3)]
                d.append("C" + " ".join(f"{a:.2f} {b:.2f}" for a, b in pts))
                i += 3
            else:  # CLOSEPOLY
                d.append("Z")
                i += 1
        return "".join(d)

    cy = base - 0.5 * 0.70 * size - cell / 2  # cells centred on the x-height
    rects = "".join(
        f'<rect x="{k * (cell + gap):.2f}" y="{cy:.2f}" width="{cell:.2f}" '
        f'height="{cell:.2f}" fill="{c}"/>' for k, c in enumerate(CELLS[theme]))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} '
           f'{height:.0f}" width="{width / 2:.0f}" height="{height / 2:.0f}" '
           f'role="img" aria-label="event2vec">{rects}'
           f'<path d="{path_d(tp)}" fill="{WORD[theme]}"/></svg>\n')
    with open(path, "w") as f:
        f.write(svg)


# --------------------------------------------------------------------------- #
# geometries figure
# --------------------------------------------------------------------------- #
def run(model, events, ids, pre=None):
    """States after each event (including the start), as a list of tensors."""
    with torch.no_grad():
        model.embedding.weight.copy_(torch.as_tensor(events, dtype=torch.float32))
        model.eval()
        h = model.zero_state(1, torch.device("cpu"))
        out = [h[0].clone()]
        for i in ids:
            h = model(torch.tensor([i]), h)[1]
            out.append(h[0].clone())
    return torch.stack(out).numpy()


def sphere_xyz(p, kappa=1.0):
    """Inverse stereographic projection of a kappa>0 point onto the unit sphere."""
    r2 = kappa * (p ** 2).sum(-1, keepdims=True)
    xy = 2 * np.sqrt(kappa) * p / (1 + r2)
    z = (1 - r2) / (1 + r2)
    return np.concatenate([xy, z], -1)


def view(xyz, tilt=0.45):
    """Orthographic view of 3-D points, tilted towards the viewer."""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    return np.stack([x, y * np.cos(tilt) - z * np.sin(tilt)], -1), \
        y * np.sin(tilt) + z * np.cos(tilt)


def draw_path(ax, pts, color, lw=2.2, depth=None):
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, solid_capstyle="round", zorder=3)
    ax.scatter(*pts[-1], s=46, color=color, zorder=4, edgecolors="white", linewidths=1)


def start(ax, p=(0, 0)):
    ax.scatter(*p, s=22, color=INK, zorder=5)


def circle(ax, r=1.0, c=(0, 0), **kw):
    ax.add_patch(plt.Circle(c, r, fill=False, color=kw.pop("color", "#8a919c"),
                            lw=kw.pop("lw", 1.0), **kw))


def fit(ax, pad=0.12):
    """Square limits around every line, marker and circle in the panel."""
    pts = [ln.get_xydata() for ln in ax.lines if ln.get_transform() == ax.transData]
    pts += [c.get_offsets() for c in ax.collections]
    for pa in ax.patches:
        (cx, cy), r = pa.center, pa.radius
        pts.append(np.array([[cx - r, cy - r], [cx + r, cy + r]]))
    p = np.concatenate([np.asarray(q, float).reshape(-1, 2) for q in pts])
    p = p[np.isfinite(p).all(1)]
    lo, hi = p.min(0), p.max(0)
    c, half = (lo + hi) / 2, (hi - lo).max() / 2 * (1 + pad) + 1e-6
    ax.set_xlim(c[0] - half, c[0] + half)
    ax.set_ylim(c[1] - half, c[1] + half)


def panel_euclidean(ax):
    ev = [[0.9, 0.15], [0.35, 0.75], [-0.55, 0.5]]
    for ids, col in ((ABC, BLUE), (CBA, RED)):
        draw_path(ax, run(EuclideanModel(3, 2, dropout_p=0.0), ev, ids), col)
    start(ax)


def panel_hyperbolic(ax):
    ev = [[0.55, 0.05], [0.2, 0.45], [-0.45, 0.25]]
    for ids, col in ((ABC, BLUE), (CBA, RED)):
        draw_path(ax, run(HyperbolicModel(3, 2, dropout_p=0.0, c=1.0), ev, ids), col)
    circle(ax)
    start(ax)


def panel_stereographic(ax):
    k = -0.5
    r = 1 / np.sqrt(-k)
    ev = [[0.75, 0.05], [0.25, 0.65], [-0.6, 0.35]]
    for ids, col in ((ABC, BLUE), (CBA, RED)):
        draw_path(ax, run(StereographicModel(3, 2, dropout_p=0.0, kappa=k), ev, ids), col)
    circle(ax, r)
    start(ax)
    ax.text(0.97, 0.95, "κ = −0.5", transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=MUTED, family="monospace")


def panel_spherical(ax):
    """kappa > 0: the state lives on a sphere (shown tilted towards the viewer)."""
    ev = [[0.5, 0.05], [0.15, 0.45], [-0.45, 0.2]]
    circle(ax)
    for t in np.linspace(-0.8, 0.8, 5):  # faint latitude lines, front half
        th = np.linspace(0, 2 * np.pi, 160)
        lat = np.stack([np.sqrt(1 - t * t) * np.cos(th), np.sqrt(1 - t * t) * np.sin(th),
                        np.full_like(th, t)], -1)
        q, d = view(lat)
        ax.plot(*np.where((d > 0)[:, None], q, np.nan).T, color=RULE, lw=0.6, zorder=1)
    for ids, col in ((ABC, BLUE), (CBA, RED)):
        xyz = sphere_xyz(run(StereographicModel(3, 2, dropout_p=0.0, kappa=1.0), ev, ids))
        fine = np.concatenate([np.linspace(a, b, 16, endpoint=False)
                               for a, b in zip(xyz[:-1], xyz[1:])] + [xyz[-1:]])
        fine /= np.linalg.norm(fine, axis=1, keepdims=True)
        q, _ = view(fine)
        draw_path(ax, q, col)
    q, _ = view(np.array([[0.0, 0.0, 1.0]]))
    ax.scatter(*q[0], s=22, color=INK, zorder=5)


def panel_rotation(ax):
    """Each event rotates the state; follow where it sends the north pole."""
    ev = [[0.9, 0.0, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 0.9]]
    circle(ax)
    north = np.array([0.0, 0.0, 1.0])
    for ids, col in ((ABC, BLUE), (CBA, RED)):
        mats = run(RotationModel(3, n=3, dropout_p=0.0), ev, ids).reshape(-1, 3, 3)
        xyz = np.stack([m @ north for m in mats])
        fine = np.concatenate([np.linspace(a, b, 16, endpoint=False)
                               for a, b in zip(xyz[:-1], xyz[1:])] + [xyz[-1:]])
        fine /= np.linalg.norm(fine, axis=1, keepdims=True)
        q, _ = view(fine)
        draw_path(ax, q, col)
    q, _ = view(north[None])
    ax.scatter(*q[0], s=22, color=INK, zorder=5)


def panel_heisenberg(ax):
    """a·b and b·a reach the same (x, y) but sweep opposite signed areas z."""
    ev = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    for ids, col, sign in (([0, 1], BLUE, "+"), ([1, 0], RED, "−")):
        st = run(HeisenbergModel(3, m=1, dropout_p=0.0), ev, ids)
        draw_path(ax, st[:, :2], col)
        z = st[-1, 2]
        ax.text(0.68 if sign == "+" else 0.32, 0.32 if sign == "+" else 0.68,
                f"z = {z:+.1f}", ha="center", va="center", fontsize=9, color=col,
                family="monospace")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=1)
    start(ax)


def panel_affine(ax):
    """Each event rescales/rotates the history, then shifts it: order matters,
    but every step stays invertible."""
    model = AffineModel(3, embedding_dim=2, dropout_p=0.0)
    v = [[0.9, 0.1], [0.3, 0.7], [-0.5, 0.45]]
    m = [[[-0.35, -0.6], [0.6, -0.35]],      # shrink + rotate
         [[0.25, 0.0], [0.0, -0.5]],         # stretch x, squeeze y
         [[0.0, 0.5], [-0.5, 0.0]]]          # rotate
    gen = [vi + np.array(mi).ravel().tolist() for vi, mi in zip(v, m)]
    for ids, col in ((ABC, BLUE), (CBA, RED)):
        st = run(model, gen, ids)
        draw_path(ax, st[:, 4:], col)  # the point b
        ax.scatter(*st[1:-1, 4:].T, s=14, color=col, zorder=4)  # intermediate states
    start(ax)


def panel_product(ax):
    """Hyperbolic x spherical: each factor updates its own slice of the state."""
    ev = [[0.55, 0.05, 0.6, 0.1], [0.2, 0.45, 0.0, 0.7], [-0.45, 0.25, -0.5, 0.3]]
    off = 1.25
    for ids, col in ((ABC, BLUE), (CBA, RED)):
        prod = ProductModel([StereographicModel(3, 2, dropout_p=0.0, kappa=-1.0),
                             StereographicModel(3, 2, dropout_p=0.0, kappa=1.0)])
        with torch.no_grad():
            for k, f in enumerate(prod.factors):
                f.embedding.weight.copy_(torch.tensor(ev)[:, 2 * k:2 * k + 2])
        prod.eval()
        with torch.no_grad():
            h = prod.zero_state(1, torch.device("cpu"))
            st = [h[0].clone()]
            for i in ids:
                h, _ = prod.encoder(torch.tensor([i]), h)
                st.append(h[0].clone())
        st = torch.stack(st).numpy()
        hyp = st[:, :2] * 0.95 - [off, 0]
        sph, _ = view(sphere_xyz(st[:, 2:]))
        draw_path(ax, hyp, col)
        draw_path(ax, sph * 0.95 + [off, 0], col)
    circle(ax, 0.95, (-off, 0))
    circle(ax, 0.95, (off, 0))
    start(ax, (-off, 0))
    start(ax, (off, 0.95 * view(np.array([[0, 0, 1.0]]))[0][0, 1]))
    for x, t in ((-off, "hyperbolic"), (off, "spherical")):
        ax.text(x, -1.18, t, ha="center", fontsize=8, color=MUTED, family="monospace")


PANELS = [
    ("euclidean", "vector addition", "bags of events, exact\nexplanations", panel_euclidean),
    ("hyperbolic", "Möbius addition", "hierarchies, trees", panel_hyperbolic),
    ("stereographic", "κ-addition, signed κ", "one curvature dial", panel_stereographic),
    ("spherical", "κ-addition, κ > 0", "cycles: weekdays,\nseasons", panel_spherical),
    ("rotation", "product of SO(n) rotations", "histories where order\nchanges the outcome", panel_rotation),
    ("heisenberg", "Heisenberg group product", "same endpoint, opposite\nsigned area", panel_heisenberg),
    ("affine", "x ↦ exp(M) x + v", "events that act on\nthe history", panel_affine),
    ("product", "one update per factor", "two structures at once", panel_product),
]


def write_geometries(path):
    fig, axes = plt.subplots(2, 4, figsize=(14, 9.2), facecolor="white")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.14,
                        wspace=0.1, hspace=0.72)
    for ax, (name, rule, use, fn) in zip(axes.ravel(), PANELS):
        ax.set_facecolor(PANEL)
        fn(ax)
        fit(ax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(0.0, -0.08, f'"{name}"', transform=ax.transAxes, fontsize=12.5,
                color=INK, family="monospace", weight="bold", va="top")
        ax.text(0.0, -0.2, rule, transform=ax.transAxes, fontsize=9, color=MUTED,
                family="monospace", va="top")
        ax.text(0.0, -0.3, use, transform=ax.transAxes, fontsize=10, color=INK, va="top")
        ax.plot([0, 1], [1.02, 1.02], transform=ax.transAxes, color=INK, lw=1.2,
                clip_on=False)
    fig.text(0.02, 0.955, "The same three events, composed by each geometry",
             fontsize=17, color=INK, weight="bold")
    fig.text(0.02, 0.915, "a·b·c", fontsize=12, color=BLUE, family="monospace")
    fig.text(0.065, 0.915, "against", fontsize=12, color=MUTED)
    fig.text(0.115, 0.915, "c·b·a", fontsize=12, color=RED, family="monospace")
    fig.text(0.16, 0.915, "(heisenberg: a·b against b·a). Where the endpoints land apart, "
             "the geometry remembers order.", fontsize=12, color=MUTED)
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--font", help="Archivo Bold Expanded TTF for the wordmark")
    ap.add_argument("--out", default=HERE)
    args = ap.parse_args()
    torch.manual_seed(0)
    write_geometries(os.path.join(args.out, "geometries.png"))
    if args.font:
        for theme in ("light", "dark"):
            write_logo(args.font, theme, os.path.join(args.out, f"logo-{theme}.svg"))
    print("wrote figures to", args.out)


if __name__ == "__main__":
    main()
