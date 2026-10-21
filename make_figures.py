"""
Figures for the final report, from the saved results.
Applied Statistics & Stochastic Processes project.

Reads results/runs.csv, results/stats_friedman.csv and writes:
  figures/fig_hit_vs_k.png        share of runs that found the exact optimum, by k
  figures/fig_gap_k5.png          distribution of relative gaps at k = 5
  figures/fig_convergence.png     mean relative gap of the best-so-far vs evaluations
  figures/fig_cd.png              critical-difference diagrams (Friedman + Nemenyi)
  figures/fig_runtime.png         run time: each optimiser vs the exact DP

(landscape_k2.py, sa_markov.py and bitdepth_scaling.py draw their own figures.)

Colours: one fixed colour per algorithm in every figure (identity never
changes between charts), plus a marker/line style so no figure relies on
colour alone. The two baselines (random search, hill climber) are drawn
in neutral tones so the five metaheuristics stand out.

Run:  python make_figures.py
"""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thresholding import load_grey, histogram, dp_multilevel

RESULTS, FIGURES = Path("results"), Path("figures")
ALGOS = ["random_search", "hill_climber", "sa", "pso", "ga", "de", "gwo"]
LABEL = {"random_search": "Random search", "hill_climber": "Hill climber",
         "sa": "Simulated annealing", "pso": "PSO", "ga": "GA", "de": "DE",
         "gwo": "Grey wolf"}
COLOR = {"sa": "#2a78d6", "pso": "#eb6834", "ga": "#1baf7a", "de": "#eda100",
         "gwo": "#e87ba4", "hill_climber": "#6250d6", "random_search": "#8a8984"}
MARKER = {"sa": "o", "pso": "s", "ga": "^", "de": "D", "gwo": "v",
          "hill_climber": "P", "random_search": "X"}
LS = {a: "-" for a in ALGOS}
LS["random_search"] = "--"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e5e2"
FLOOR = 1e-7          # log-scale floor: a mean gap of 0 is drawn here and labelled

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10,
    "axes.titlecolor": INK, "legend.frameon": False,
})


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def fig_hit_vs_k(runs):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, obj in zip(axes, ("otsu", "kapur")):
        sub = runs[runs.objective == obj]
        hr = sub.groupby(["algorithm", "k"]).hit.mean().unstack()
        for a in ALGOS:
            ax.plot(hr.columns, hr.loc[a] * 100, color=COLOR[a], ls=LS[a], lw=2,
                    marker=MARKER[a], ms=7, mec="white", mew=1, label=LABEL[a])
        ax.set_title(f"{'Otsu' if obj == 'otsu' else 'Kapur'}: runs that found the exact optimum")
        ax.set_xlabel("number of thresholds k")
        ax.set_xticks(sorted(runs.k.unique()))
        ax.set_ylim(-3, 103)
        style(ax)
    axes[0].set_ylabel("hit rate (%), 8 images x 30 seeds")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_hit_vs_k.png", dpi=170)
    plt.close(fig)


def fig_gap_k5(runs):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, obj in zip(axes, ("otsu", "kapur")):
        sub = runs[(runs.objective == obj) & (runs.k == 5)]
        data, zeros = [], []
        for a in ALGOS:
            g = sub[sub.algorithm == a].rel_gap.to_numpy()
            zeros.append((g <= 0).mean())
            data.append(np.maximum(g, FLOOR))
        bp = ax.boxplot(data, vert=False, widths=0.55, patch_artist=True,
                        medianprops=dict(color=INK, lw=1.5),
                        whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                        flierprops=dict(marker=".", ms=3, mfc=MUTED, mec="none", alpha=0.5))
        for patch, a in zip(bp["boxes"], ALGOS):
            patch.set_facecolor(COLOR[a])
            patch.set_alpha(0.75)
            patch.set_edgecolor("white")
        ax.set_xscale("log")
        ax.set_yticks(range(1, len(ALGOS) + 1))
        ax.set_yticklabels([f"{LABEL[a]}  ({z:.0%} exact)" for a, z in zip(ALGOS, zeros)])
        ax.set_title(f"{'Otsu' if obj == 'otsu' else 'Kapur'}, k = 5: relative gap per run")
        ax.set_xlabel(f"relative gap (log; exact hits drawn at {FLOOR:g})")
        style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_gap_k5.png", dpi=170)
    plt.close(fig)


def fig_convergence(runs, k=5):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, obj in zip(axes, ("otsu", "kapur")):
        sub = runs[(runs.objective == obj) & (runs.k == k)]
        for a in ALGOS:
            s = sub[sub.algorithm == a]
            curves = np.array([[float(v) for v in c.split()] for c in s.curve])
            opt = s.opt_score.to_numpy()[:, None]
            gap = (opt - curves) / np.abs(opt)
            gap = np.where(np.isfinite(gap), gap, np.nan)
            mean_gap = np.nanmean(gap, axis=0)
            x = np.arange(1, curves.shape[1] + 1) * 50
            ax.plot(x, np.maximum(mean_gap, FLOOR), color=COLOR[a], ls=LS[a], lw=2,
                    label=LABEL[a])
            ax.plot(x[-1], max(mean_gap[-1], FLOOR), marker=MARKER[a], color=COLOR[a],
                    ms=7, mec="white", mew=1)
        ax.set_yscale("log")
        ax.set_title(f"{'Otsu' if obj == 'otsu' else 'Kapur'}, k = {k}: mean gap of best-so-far")
        ax.set_xlabel("objective-function evaluations")
        style(ax)
    axes[0].set_ylabel("mean relative gap to exact optimum (log)")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_convergence.png", dpi=170)
    plt.close(fig)


def cd_diagram(ax, ranks, cd, title):
    """Demsar-style critical-difference diagram on one axis."""
    K = len(ranks)
    order = sorted(ranks, key=ranks.get)
    ax.set_xlim(0.5, K + 0.5)
    ax.set_ylim(0.35, 1.05)
    ax.invert_xaxis()                        # best (rank 1) on the right
    ax.axis("off")
    y0 = 0.82
    ax.hlines(y0, 1, K, color=INK, lw=1)
    for r in range(1, K + 1):
        ax.vlines(r, y0, y0 + 0.04, color=INK, lw=1)
        ax.text(r, y0 + 0.07, str(r), ha="center", fontsize=8, color=MUTED)
    # CD bar
    ax.hlines(0.98, K, K - cd, color=INK, lw=2)
    ax.text(K - cd / 2, 1.0, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=8)
    # algorithms: half on each side
    left = order[K // 2:][::-1]              # worse ranks, drawn on the left
    right = order[:K // 2 + 0]
    for i, a in enumerate(right):
        y = y0 - 0.12 - i * 0.1
        r = ranks[a]
        ax.vlines(r, y, y0, color=COLOR[a], lw=1.5)
        ax.hlines(y, r, 0.6, color=COLOR[a], lw=1.5)
        ax.text(0.55, y, f"{LABEL[a]} ({r:.2f})", ha="left", va="center", fontsize=8)
    for i, a in enumerate(left):
        y = y0 - 0.12 - i * 0.1
        r = ranks[a]
        ax.vlines(r, y, y0, color=COLOR[a], lw=1.5)
        ax.hlines(y, r, K + 0.4, color=COLOR[a], lw=1.5)
        ax.text(K + 0.45, y, f"({r:.2f}) {LABEL[a]}", ha="right", va="center", fontsize=8)
    # cliques: maximal groups whose rank range < CD
    vals = sorted(ranks.values())
    cliques = []
    for i in range(K):
        j = i
        while j + 1 < K and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i and not any(c[0] <= i and j <= c[1] for c in cliques):
            cliques.append((i, j))
    for n, (i, j) in enumerate(cliques):
        y = y0 - 0.05 - n * 0.035
        ax.hlines(y, vals[i] - 0.03, vals[j] + 0.03, color=INK, lw=3)
    ax.set_title(title, fontsize=10, color=INK, pad=14)


def fig_cd():
    fr = pd.read_csv(RESULTS / "stats_friedman.csv")
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.6))
    for ax, obj in zip(axes, ("otsu", "kapur")):
        sub = fr[fr.analysis == f"{obj} overall"]
        ranks = dict(zip(sub.algorithm, sub.avg_rank))
        cd = float(sub.nemenyi_cd.iloc[0])
        p = float(sub.friedman_p.iloc[0])
        cd_diagram(ax, ranks, cd,
                   f"{'Otsu' if obj == 'otsu' else 'Kapur'}: average rank over 8 images "
                   f"(Friedman p = {p:.1e})")
    fig.text(0.5, 0.005, "Black bars join algorithms whose average ranks differ by less than "
             "the critical difference (not significantly different, Nemenyi, alpha = 0.05).",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_cd.png", dpi=170)
    plt.close(fig)


def fig_runtime(runs):
    med = runs.groupby("algorithm").time_s.median().loc[ALGOS] * 1000
    saved = RESULTS / "stats_dp_time.txt"      # same DP measurement as the report text
    if saved.exists():
        dp = float(saved.read_text()) * 1000
    else:
        dp = None
    if dp is None:                             # analyse_stats.py not run yet: measure here
        times = []
        for image in runs.image.unique():
            p = histogram(load_grey(image))
            for obj in ("otsu", "kapur"):
                for k in sorted(runs.k.unique()):
                    dp_multilevel(p, k, obj)
                    t0 = time.perf_counter()
                    for _ in range(5):
                        dp_multilevel(p, k, obj)
                    times.append((time.perf_counter() - t0) / 5 * 1000)
        dp = float(np.median(times))
    names = ["Exact DP"] + [LABEL[a] for a in ALGOS]
    vals = [dp] + list(med.values)
    cols = [INK] + [COLOR[a] for a in ALGOS]
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.barh(range(len(vals)), [vals[i] for i in order], color=[cols[i] for i in order],
            height=0.6, edgecolor="white")
    for y, i in enumerate(order):
        ax.text(vals[i] * 1.08, y, f"{vals[i]:.1f} ms" if vals[i] < 10 else f"{vals[i]:.0f} ms",
                va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels([names[i] for i in order])
    ax.set_xscale("log")
    ax.set_xlabel("median time per solve (ms, log) - DP is exact; others use 5,000 evaluations")
    ax.set_title("Run time: exact answer vs approximate search (this machine)")
    style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_runtime.png", dpi=170)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    runs = pd.read_csv(RESULTS / "runs.csv")
    fig_hit_vs_k(runs)
    fig_gap_k5(runs)
    fig_convergence(runs)
    if (RESULTS / "stats_friedman.csv").exists():
        fig_cd()
    else:
        print("results/stats_friedman.csv missing - run analyse_stats.py first for fig_cd")
    fig_runtime(runs)
    print("Saved figures/fig_hit_vs_k.png, fig_gap_k5.png, fig_convergence.png, "
          "fig_cd.png, fig_runtime.png")


if __name__ == "__main__":
    main()
