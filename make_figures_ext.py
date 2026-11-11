"""
Figures for the three extensions, from saved results only.
Applied Statistics & Stochastic Processes project.

Writes (figures/extensions/, never touching the original figures/):
  fig_hit_vs_k_24.png        share of runs that found the exact optimum, 24 images
  fig_cd_24.png              critical-difference diagrams, 24 images (Friedman + Nemenyi)
  fig_bitdepth_crossover.png time per solve vs L, DP and optimisers, log-log,
                             with the measured crossover marked
  fig_bitdepth_quality.png   hit rate vs L: what the cheaper answer is worth
  fig_tsallis_hit.png        Tsallis: exact-hit rate at k = 2, 3 and
                             best-known-match rate at k = 4, 5 (drawn differently)

Same colour, marker and label per algorithm as the original figures (imported
from make_figures.py), so a reader never has to re-learn the legend.

Run after the analyses:  python make_figures_ext.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from make_figures import ALGOS, COLOR, LABEL, LS, MARKER, MUTED, cd_diagram, plt, style

RESULTS = Path("results")
OUT = Path("figures/extensions")


def fig_hit_vs_k_24():
    hk = pd.read_csv(RESULTS / "ext24_hit_by_k.csv", header=[0, 1], index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, obj in zip(axes, ("otsu", "kapur")):
        for a in ALGOS:
            ks = [2, 3, 4, 5]
            ax.plot(ks, [100 * hk.loc[a, (obj, str(k))] for k in ks], color=COLOR[a],
                    marker=MARKER[a], ls=LS[a], lw=2, ms=6, label=LABEL[a])
        ax.set_title(f"{obj.capitalize()}: runs that found the exact optimum (24 images)")
        ax.set_xlabel("number of thresholds k")
        ax.set_xticks([2, 3, 4, 5])
        style(ax)
    axes[0].set_ylabel("hit rate (%), 24 images x 30 seeds")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(OUT / "fig_hit_vs_k_24.png", dpi=170)
    plt.close(fig)


def fig_cd_24():
    fr = pd.read_csv(RESULTS / "ext24_friedman.csv")
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.6))
    for ax, obj in zip(axes, ("otsu", "kapur")):
        sub = fr[fr.analysis == f"{obj} overall"]
        cd_diagram(ax, dict(zip(sub.algorithm, sub.avg_rank)), float(sub.nemenyi_cd.iloc[0]),
                   f"{obj.capitalize()}: average rank over 24 images "
                   f"(Friedman p = {float(sub.friedman_p.iloc[0]):.1e})")
    fig.text(0.5, 0.005, "Black bars join algorithms whose average ranks differ by less than "
             "the critical difference (Nemenyi, alpha = 0.05).", ha="center", fontsize=8,
             color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cd_24.png", dpi=170)
    plt.close(fig)


def fig_bitdepth():
    s = pd.read_csv(RESULTS / "bitdepth_crossover.csv")
    s = s[s.source == "camera"]
    for kind in ("time", "quality"):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True,
                                 sharey=(kind == "quality"))
        for row, obj in enumerate(("otsu", "kapur")):
            for col, k in enumerate((2, 3, 5)):
                ax = axes[row, col]
                sub = s[(s.objective == obj) & (s.k == k)]
                if kind == "time":
                    dp = sub.drop_duplicates("L").sort_values("L")
                    ax.plot(dp.L, dp.dp_time_s, color="black", lw=2.5, marker="o",
                            label="Exact DP")
                for a in ALGOS:
                    g = sub[sub.algorithm == a].sort_values("L")
                    y = g.median_time_s if kind == "time" else 100 * g.hit_rate
                    ax.plot(g.L, y, color=COLOR[a], marker=MARKER[a], ls=LS[a], lw=1.5,
                            ms=5, label=LABEL[a])
                    if kind == "time" and g.crossover_L.notna().any():
                        cx = g.crossover_L.dropna().iloc[0]
                        ax.plot([cx], [g[g.L == cx].median_time_s.iloc[0]], marker="o", ms=13,
                                mfc="none", mec=COLOR[a], mew=1.5)
                ax.set_xscale("log", base=2)
                if kind == "time":
                    ax.set_yscale("log")
                ax.set_title(f"{obj.capitalize()}, k = {k}")
                style(ax)
                if row == 1:
                    ax.set_xlabel("grey levels L (log scale)")
        axes[0, 0].set_ylabel("seconds per solve (log)" if kind == "time" else "hit rate (%)")
        axes[1, 0].set_ylabel("seconds per solve (log)" if kind == "time" else "hit rate (%)")
        axes[0, 2].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        note = ("Median of 10 seeds per optimiser, single process, this laptop. Open circle = "
                "first measured L where the exact DP is slower than that optimiser's median run."
                if kind == "time" else
                "Share of the 10 runs that found the exact DP optimum. Dithered camera histograms.")
        fig.text(0.5, 0.005, note, ha="center", fontsize=8, color=MUTED)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        name = "fig_bitdepth_crossover.png" if kind == "time" else "fig_bitdepth_quality.png"
        fig.savefig(OUT / name, dpi=160)
        plt.close(fig)


def fig_tsallis_hit():
    hk = pd.read_csv(RESULTS / "tsallis_hit_by_k.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for a in ALGOS:
        y = [100 * hk.loc[a, str(k)] for k in (2, 3, 4, 5)]
        ax.plot([2, 3], y[:2], color=COLOR[a], marker=MARKER[a], ls="-", lw=2, ms=6,
                label=LABEL[a])
        ax.plot([4, 5], y[2:], color=COLOR[a], marker=MARKER[a], ls=":", lw=1.5, ms=6,
                mfc="white")
    ax.axvspan(3.5, 5.5, color="#f1f0ec", zorder=0)
    ax.text(4.5, 103, "k = 4, 5: matched BEST KNOWN\n(not proven optimal)", ha="center",
            va="bottom", fontsize=8, color=MUTED)
    ax.text(2.5, 103, "k = 2, 3: found EXACT optimum\n(exhaustive search)", ha="center",
            va="bottom", fontsize=8, color=MUTED)
    ax.set_xlim(1.7, 5.3)
    ax.set_ylim(-3, 118)
    ax.set_xticks([2, 3, 4, 5])
    ax.set_xlabel("number of thresholds k")
    ax.set_ylabel("hit rate (%), 24 images x 30 seeds")
    ax.set_title("Tsallis (non-separable, q = 0.7)")
    style(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(OUT / "fig_tsallis_hit.png", dpi=170)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(fig_hit_vs_k_24, "ext24_hit_by_k.csv"), (fig_cd_24, "ext24_friedman.csv"),
            (fig_bitdepth, "bitdepth_crossover.csv"), (fig_tsallis_hit, "tsallis_hit_by_k.csv")]
    for fn, need in jobs:
        if (RESULTS / need).exists():
            fn()
            print(f"{fn.__name__}: saved")
        else:
            print(f"{fn.__name__}: skipped, results/{need} missing")


if __name__ == "__main__":
    main()
