"""EMNLP appendix figure: 1x4 layer-sweep heatmaps for Llama-3.1-8B.

Each panel = one alpha_factor (mult) value in {-3, -2, +2, +3}; cells = literal
rate at (source_layer, injection_layer). Cells without data are shown as a
faint dot. Shared colorbar 'Literal rate' on the right.

Reads:  assets/layer_sweep_llama8b_labelsonly.json
Writes: assets/layer_sweep_heatmap_llama8b.{pdf,png}
"""
from __future__ import annotations
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.environ.get("IDIOSTEER_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS = os.path.join(ROOT, "assets")
OUT_DIR = ASSETS
INPUT = "layer_sweep_llama8b_labelsonly.json"

STRENGTHS = [-3.0, -2.0, 2.0, 3.0]
VMAX = 0.5

RE_SWEEP = re.compile(
    r"^mean_diff_s(?P<src>\d+)_a(?P<app>\d+)_(?:alpha|mult)_(?P<val>[+\-]?\d+(?:\.\d+)?)$"
)


def derive_label(j):
    lab = j.get("label")
    if isinstance(lab, str) and lab in ("figurative", "literal", "incoherent"):
        return lab
    if j.get("is_coherent") is False: return "incoherent"
    if j.get("literal_better_supported") is True: return "literal"
    if j.get("is_coherent") is True and j.get("literal_better_supported") is False:
        return "figurative"
    return None


def is_error(j):
    if not isinstance(j, dict): return True
    if j.get("error") or j.get("parse_error") is True: return True
    if j.get("is_coherent") is None: return True
    return False


def literal_rate(labels):
    if not labels: return float("nan")
    return sum(1 for l in labels if l == "literal") / len(labels)


def load_sweep():
    with open(os.path.join(ASSETS, INPUT)) as f: doc = json.load(f)
    rows = defaultdict(list)
    for r in doc["results"]:
        for cond, items in (r.get("judge_v14") or {}).items():
            m = RE_SWEEP.match(cond)
            if not m or not isinstance(items, list): continue
            s, a, v = int(m.group("src")), int(m.group("app")), float(m.group("val"))
            for j in items:
                if is_error(j): continue
                rows[(s, a, v)].append(derive_label(j))
    sources  = sorted({s for (s, a, _) in rows.keys()})
    injects  = sorted({a for (s, a, _) in rows.keys()})
    return rows, sources, injects


def main():
    plt.rcParams.update({
        "font.size":       11,
        "axes.titlesize":  13,
        "axes.labelsize":  12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "pdf.fonttype":    42,
        "ps.fonttype":     42,
    })

    rows, sources, injects = load_sweep()

    # Build matrices: rows indexed by injection layer, cols by source layer.
    grids = {}
    for v in STRENGTHS:
        G = np.full((len(injects), len(sources)), np.nan)
        for i, a in enumerate(injects):
            for j, s in enumerate(sources):
                G[i, j] = literal_rate(rows.get((s, a, v), []))
        grids[v] = G

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.8), sharey=True,
                             gridspec_kw={"wspace": 0.18})
    cmap = "YlOrRd"
    ims = []

    for ax, v in zip(axes, STRENGTHS):
        G = grids[v]
        im = ax.imshow(G, origin="lower", cmap=cmap, vmin=0.0, vmax=VMAX,
                       aspect="auto")
        ims.append(im)
        ax.set_xticks(range(len(sources)))
        ax.set_yticks(range(len(injects)))
        ax.set_xticklabels(sources)
        ax.set_yticklabels(injects)
        sign = "+" if v > 0 else "−"
        ax.set_title(rf"$\alpha_{{\mathrm{{factor}}}} = {sign}{abs(v):.1f}$", pad=4)
        ax.set_xlabel(r"Source layer $L_s$")
        if ax is axes[0]:
            ax.set_ylabel(r"Injection layer $L_i$")
        ax.tick_params(length=2.5, pad=1.5)

        for i in range(G.shape[0]):
            for j in range(G.shape[1]):
                val = G[i, j]
                if np.isnan(val):
                    ax.text(j, i, "·", ha="center", va="center",
                            color="0.65", fontsize=10)
                else:
                    color = "white" if val > VMAX * 0.55 else "black"
                    ax.text(j, i, f"{val*100:.0f}", ha="center", va="center",
                            color=color, fontsize=10)

    cbar = fig.colorbar(ims[-1], ax=axes, fraction=0.018, pad=0.015,
                        shrink=0.96)
    cbar.set_label("Literal rate", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    # Express ticks as 0.0..0.5 (matches the values in cells × 100).
    cbar.set_ticks(np.linspace(0, VMAX, 6))

    out_pdf = os.path.join(OUT_DIR, "layer_sweep_heatmap_llama8b.pdf")
    out_png = os.path.join(OUT_DIR, "layer_sweep_heatmap_llama8b.png")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.01)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
