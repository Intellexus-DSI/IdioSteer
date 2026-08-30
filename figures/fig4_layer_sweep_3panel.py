"""EMNLP two-column 1x3 layer-sweep figure for Llama-3B / Llama-8B / Gemma-2B.

Each panel plots three curves at the strongest negative strength (mult=-3):
  - Source-layer sweep  : vary source layer, inject layer fixed at 2
  - Inject-layer sweep  : vary inject layer, source layer fixed at canonical
  - Same-layer (diag.)  : source == inject

Plus a per-model baseline literal rate (horizontal dotted line).

Reads labels-only sweep JSONs from assets/ + baseline from results/per_condition_label_counts.csv.
Writes PDF + PNG at assets/layer_sweep_3panel_3models.{pdf,png}.
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
RESULTS = os.path.join(ROOT, "results")

# Sweep cell labels ship in assets/ (completions stripped); the baseline literal rate
# comes from the aggregate results/per_condition_label_counts.csv (all-generations denom).
MODELS = [
    {"tag": "Llama-3B", "sweep_file": "layer_sweep_llama3b_labelsonly.json",   "model": "Llama-3.2-3B"},
    {"tag": "Llama-8B", "sweep_file": "layer_sweep_llama8b_labelsonly.json",   "model": "Llama-3.1-8B"},
    {"tag": "Gemma-2B", "sweep_file": "layer_sweep_gemma2_2b_labelsonly.json", "model": "Gemma-2-2B"},
]

STRENGTH = -2.0  # alpha_factor = -2 (= mult=-2); matches the literal_vs_alpha overlay strength

C_SOURCE   = "#1f77b4"  # blue
C_INJECT   = "#d62728"  # red
C_DIAGONAL = "#7f7f7f"  # grey

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


def load_sweep(path):
    """Return (rows_by_(src,app,strength), metadata.sweeps)."""
    with open(path) as f: doc = json.load(f)
    rows = defaultdict(list)
    for r in doc["results"]:
        for cond, items in (r.get("judge_v14") or {}).items():
            m = RE_SWEEP.match(cond)
            if not m or not isinstance(items, list): continue
            s = int(m.group("src")); a = int(m.group("app"))
            v = float(m.group("val"))
            for j in items:
                if is_error(j): continue
                rows[(s, a, v)].append(derive_label(j))
    return rows, doc.get("metadata", {}).get("sweeps", {})


def load_baseline(model_name):
    """Baseline (no-steering) literal rate from the shipped aggregate CSV (all-generations)."""
    import csv
    path = os.path.join(RESULTS, "per_condition_label_counts.csv")
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["model"] == model_name and row["condition"] == "baseline":
                return float(row["literal_rate"])
    return float("nan")


def annotate_peak(ax, xs, ys, color, dy=5, va="bottom"):
    arr = np.array(ys, dtype=float)
    if np.all(np.isnan(arr)): return
    i = int(np.nanargmax(arr))
    x, y = xs[i], arr[i]
    txt = f"{y:.2f}".lstrip("0") if y < 1 else f"{y:.2f}"
    ax.annotate(txt, (x, y), xytext=(0, dy), textcoords="offset points",
                ha="center", va=va, fontsize=9, color=color)


def plot_panel(ax, sweep_rows, sweeps_md, baseline, title, show_ylabel):
    src_sweep = sweeps_md.get("source_sweep", {})
    app_sweep = sweeps_md.get("apply_sweep", {})
    diag      = sweeps_md.get("diagonal", {})

    apply_fixed = src_sweep.get("apply", 2)
    src_fixed   = app_sweep.get("source")

    # Source-layer sweep
    xs = sorted(src_sweep.get("sources", []))
    ys = [literal_rate(sweep_rows.get((s, apply_fixed, STRENGTH), [])) for s in xs]
    ax.plot(xs, ys, color=C_SOURCE, marker="o", ms=4.5, lw=2.0,
            label="Source-layer sweep", zorder=4)
    annotate_peak(ax, xs, ys, C_SOURCE)

    # Inject-layer sweep
    xs = sorted(app_sweep.get("applies", []))
    ys = [literal_rate(sweep_rows.get((src_fixed, a, STRENGTH), [])) for a in xs]
    ax.plot(xs, ys, color=C_INJECT, marker="s", ms=4.5, lw=2.0,
            label="Inject-layer sweep", zorder=4)
    annotate_peak(ax, xs, ys, C_INJECT)

    # Same-layer (diagonal)
    xs = sorted(diag.get("layers", []))
    ys = [literal_rate(sweep_rows.get((L, L, STRENGTH), [])) for L in xs]
    ax.plot(xs, ys, color=C_DIAGONAL, marker="^", ms=4.5, lw=2.0,
            label="Same-layer", zorder=3)
    annotate_peak(ax, xs, ys, C_DIAGONAL, dy=-8, va="top")

    # Baseline
    ax.axhline(baseline, color="black", linestyle=":", lw=1.4, alpha=0.8,
               zorder=2, label="Baseline")

    ax.set_title(title, fontsize=12, pad=3)
    ax.set_xlabel("Layer")
    if show_ylabel:
        ax.set_ylabel("Literal rate")
    ax.set_ylim(0, 0.65)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax.grid(True, axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    ax.tick_params(axis="both", length=2.5, pad=1.5)


def main():
    plt.rcParams.update({
        "font.size":       10,
        "axes.labelsize":  11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.2), sharey=True)

    for i, (m, ax) in enumerate(zip(MODELS, axes)):
        sweep_rows, sweeps_md = load_sweep(os.path.join(ASSETS, m["sweep_file"]))
        baseline = load_baseline(m["model"])
        plot_panel(ax, sweep_rows, sweeps_md, baseline, m["tag"], show_ylabel=(i == 0))

    # Shared legend above panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.10),
               ncol=4, frameon=False, handlelength=1.6,
               handletextpad=0.4, columnspacing=1.2, borderpad=0.2)

    plt.tight_layout(pad=0.2, w_pad=0.6)
    plt.subplots_adjust(top=0.80)
    out_pdf = os.path.join(ASSETS, "layer_sweep_3panel_3models.pdf")
    out_png = os.path.join(ASSETS, "layer_sweep_3panel_3models.png")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.01)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
