"""
Landscape difficulty at k = 2 (Table 5 of the report).
Applied Statistics & Stochastic Processes project.

The usual reason given for using a metaheuristic is "the objective has
many local optima, so simple local search gets stuck". That is a claim
about the landscape, and at k = 2 it can be CHECKED exactly: the whole
surface J(t1, t2) has only C(254, 2) = 32,131 points, so we evaluate all
of them and count the local optima.

Definition (stated, because the count depends on it):
  a cell (t1, t2) with a finite score is a local optimum if its score is
    non-strict:  >= every neighbour       (counts every cell of a flat plateau)
    strict:      >  every neighbour
  over the 8-neighbourhood in the (t1, t2) lattice, restricted to the valid
  region 1 <= t1 < t2 <= 254. Neighbours outside that region are ignored.

Then a two-sided sign test across the 8 images asks whether Kapur has more
local optima than Otsu more often than chance would give.

Run:  python landscape_k2.py
Writes results/landscape_k2.csv and figures/landscape_k2.png
"""

from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from thresholding import load_grey, histogram, class_tables, class_term

BENCHMARK = ["camera", "coins", "moon", "page", "clock", "astronaut", "coffee", "cell"]
RESULTS, FIGURES = Path("results"), Path("figures")


def surface(p, objective):
    """
    Full k=2 surface. F[t1, t2] = J for 1 <= t1 < t2 <= 254, -inf for an
    empty class, NaN outside the valid region. Uses the same per-class
    terms as the DP (class_term), so the surface's maximum IS the DP optimum.
    """
    L = p.size
    tables = class_tables(p)
    t = np.arange(L)
    t1, t2 = np.meshgrid(t, t, indexing="ij")
    valid = (t1 >= 1) & (t1 < t2) & (t2 <= L - 2)
    F = np.full((L, L), np.nan)
    a, b = t1[valid], t2[valid]
    score = (class_term(objective, tables, -1, a)
             + class_term(objective, tables, a, b)
             + class_term(objective, tables, b, L - 1))
    if objective == "otsu":
        score = score - ((np.arange(L) * p).sum()) ** 2
    F[valid] = score
    return F


def count_local_optima(F):
    """Return (non_strict, strict) counts over the 8-neighbourhood."""
    L = F.shape[0]
    G = np.where(np.isnan(F), -np.inf, F)           # outside region: never better
    pad = np.full((L + 2, L + 2), -np.inf)
    pad[1:-1, 1:-1] = G
    ge_all = np.ones_like(G, dtype=bool)
    gt_all = np.ones_like(G, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            nb = pad[1 + di: L + 1 + di, 1 + dj: L + 1 + dj]
            ge_all &= G >= nb
            gt_all &= G > nb
    candidate = np.isfinite(F)                       # valid, non-degenerate cells
    return int((ge_all & candidate).sum()), int((gt_all & candidate).sum())


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    rows = []
    print(f"{'image':<11}{'Otsu >=':>9}{'Otsu >':>8}{'Kapur >=':>10}{'Kapur >':>9}")
    print("-" * 47)
    surfaces = {}
    for name in BENCHMARK:
        p = histogram(load_grey(name))
        counts = {}
        for obj in ("otsu", "kapur"):
            F = surface(p, obj)
            surfaces[(name, obj)] = F
            counts[obj] = count_local_optima(F)
        rows.append((name, *counts["otsu"], *counts["kapur"]))
        print(f"{name:<11}{counts['otsu'][0]:>9,}{counts['otsu'][1]:>8,}"
              f"{counts['kapur'][0]:>10,}{counts['kapur'][1]:>9,}")

    with open(RESULTS / "landscape_k2.csv", "w") as fh:
        fh.write("image,otsu_nonstrict,otsu_strict,kapur_nonstrict,kapur_strict\n")
        for r in rows:
            fh.write(",".join(map(str, r)) + "\n")

    print()
    for label, i_o, i_k in (("non-strict", 1, 3), ("strict", 2, 4)):
        more = sum(r[i_k] > r[i_o] for r in rows)
        fewer = sum(r[i_k] < r[i_o] for r in rows)
        ties = len(rows) - more - fewer
        p = binomtest(more, more + fewer, 0.5).pvalue if more + fewer else 1.0
        print(f"Sign test ({label}): Kapur > Otsu on {more}, < on {fewer}, "
              f"ties {ties}; two-sided p = {p:.4f}")

    # figure: the two surfaces for the image with the most optima
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        name = "moon"
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
        for ax, obj in zip(axes, ("otsu", "kapur")):
            F = surfaces[(name, obj)]
            show = np.where(np.isfinite(F), F, np.nan)
            im = ax.imshow(show.T, origin="lower", cmap="Blues", interpolation="nearest")
            ax.set_xlabel("t1")
            ax.set_ylabel("t2")
            title = "Otsu between-class variance" if obj == "otsu" else "Kapur entropy (nats)"
            ax.set_title(f"{name}: {title}", fontsize=10, color="#0b0b0b")
            fig.colorbar(im, ax=ax, shrink=0.85)
        fig.tight_layout()
        fig.savefig(FIGURES / "landscape_k2.png", dpi=160)
        print(f"Saved {FIGURES / 'landscape_k2.png'}")
    except ImportError:
        pass
    print(f"Saved {RESULTS / 'landscape_k2.csv'}")


if __name__ == "__main__":
    main()
