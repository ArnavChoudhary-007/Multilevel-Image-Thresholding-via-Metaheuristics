"""
Week 3 - statistical analysis of the 13,440 runs.
Applied Statistics & Stochastic Processes project.

Reads results/runs.csv (from run_experiments.py) and answers Q2:
"what does approximation cost, relative to the exact optimum?"

THE UNIT OF REPLICATION IS THE IMAGE, NOT THE RUN.
The 30 seeds on one image are repeated measurements of ONE problem; treating
them as 30 independent observations in a cross-image comparison would be
pseudo-replication. So, following the paper (section 4.2):

  1. each (image, objective, k, algorithm) CELL is summarised by the MEDIAN
     relative gap over its 30 seeds (plus its hit rate);
  2. for each (objective, k): Friedman test, 8 image blocks x 7 algorithms,
     Kendall's W as effect size, Nemenyi post-hoc with critical difference
     CD = q_0.05 * sqrt(K(K+1) / (6N));
  3. pairwise Wilcoxon signed-rank tests (21 pairs) with Holm correction and
     the matched-pairs rank-biserial correlation as effect size;
  4. an Aligned Rank Transform (ART) ANOVA, algorithm x k with image as a
     block, as a sensitivity check that does not assume normality;
  5. descriptive "cost of approximation": hit rates, gap quantiles, run time
     against the DP's run time;
  6. PSNR / SSIM of the returned segmentation vs the exact one - reported as
     reconstruction fidelity only (the benchmark has no ground truth).

Lower relative gap = better, so rank 1 = best.

Run:  python analyse_stats.py          (a few minutes: step 6 segments images)
Writes results/stats_*.csv, results/cell_summary.csv, results/stats_report.txt
"""

import math
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range, wilcoxon, kendalltau
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from thresholding import load_grey, histogram, dp_multilevel, segment, quality

RESULTS = Path("results")
ALGOS = ["random_search", "hill_climber", "sa", "pso", "ga", "de", "gwo"]
EXPECTED_RUNS = 13440


class Tee:
    """Print to the screen and to results/stats_report.txt at the same time."""

    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        self.file.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.file.flush()


def nemenyi_cd(K, N, alpha=0.05):
    q = studentized_range.ppf(1 - alpha, K, 1e6) / math.sqrt(2)
    return q, q * math.sqrt(K * (K + 1) / (6.0 * N))


def avg_ranks(matrix):
    """matrix: N blocks x K algorithms of gaps; rank within each block (1 = best)."""
    return np.vstack([rankdata(row) for row in matrix]).mean(axis=0)


def rank_biserial(x, y):
    """Matched-pairs rank-biserial r in [-1, 1]; negative = x has smaller gaps."""
    d = np.asarray(x) - np.asarray(y)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def friedman_block(mat, label, out_rows, wil_rows):
    """Friedman + Nemenyi + Wilcoxon/Holm on an N x K matrix (rows = images)."""
    N, K = mat.shape
    ranks = avg_ranks(mat)
    q, cd = nemenyi_cd(K, N)
    if np.allclose(mat, mat[:, :1]):           # every algorithm identical everywhere
        chi2, p, W = 0.0, 1.0, 0.0
    else:
        chi2, p = friedmanchisquare(*mat.T)
        W = chi2 / (N * (K - 1))
    order = np.argsort(ranks)
    print(f"\n[{label}]  Friedman chi2 = {chi2:.2f}, p = {p:.2e}, Kendall's W = {W:.2f};"
          f"  Nemenyi CD = {cd:.2f} (q = {q:.3f}, N = {N}, K = {K})")
    print("   average ranks (1 = smallest gap): " +
          ", ".join(f"{ALGOS[i]} {ranks[i]:.2f}" for i in order))
    best = order[0]
    worse = [ALGOS[i] for i in order if ranks[i] - ranks[best] > cd]
    print(f"   significantly worse than the best ({ALGOS[best]}) by Nemenyi: "
          f"{', '.join(worse) if worse else 'none'}")
    for i, a in enumerate(ALGOS):
        out_rows.append({"analysis": label, "algorithm": a, "avg_rank": ranks[i],
                         "friedman_chi2": chi2, "friedman_p": p, "kendall_W": W,
                         "nemenyi_cd": cd})

    pvals, pairs = [], []
    for i, j in combinations(range(K), 2):
        x, y = mat[:, i], mat[:, j]
        if np.allclose(x, y):
            pv = 1.0
        else:
            try:
                pv = wilcoxon(x, y).pvalue
            except ValueError:
                pv = 1.0
        pvals.append(pv)
        pairs.append((i, j, rank_biserial(x, y)))
    reject, p_holm, _, _ = multipletests(pvals, method="holm")
    n_sig = int(reject.sum())
    print(f"   Wilcoxon + Holm: {n_sig} of {len(pairs)} pairs significant at 0.05 "
          f"(smallest raw p = {min(pvals):.4f}, smallest Holm p = {min(p_holm):.4f})")
    for (i, j, r), pv, ph, rej in zip(pairs, pvals, p_holm, reject):
        wil_rows.append({"analysis": label, "a": ALGOS[i], "b": ALGOS[j], "p_raw": pv,
                         "p_holm": ph, "significant": bool(rej), "rank_biserial": r})
    return ranks


def art_anova(df, response, label, rows):
    """
    Aligned Rank Transform (Wobbrock et al. 2011) for algorithm x k, image as block.
    For each effect: strip all fixed effects (subtract the algorithm-x-k cell mean),
    add back only the estimated effect of interest, rank, fit the full model on the
    ranks, and keep only the F-test for that effect.
    """
    d = df[["image", "algorithm", "k", response]].copy()
    d["k"] = d["k"].astype(str)
    y = d[response]
    grand = y.mean()
    m_a = d.groupby("algorithm")[response].transform("mean")
    m_k = d.groupby("k")[response].transform("mean")
    m_ak = d.groupby(["algorithm", "k"])[response].transform("mean")
    resid = y - m_ak
    effects = {
        "C(algorithm)": m_a - grand,
        "C(k)": m_k - grand,
        "C(algorithm):C(k)": m_ak - m_a - m_k + grand,
    }
    print(f"\n[ART ANOVA, {label}]  response = {response}, block = image")
    for name, eff in effects.items():
        d["ar"] = rankdata(resid + eff)
        fit = smf.ols("ar ~ C(algorithm) * C(k) + C(image)", data=d).fit()
        tab = anova_lm(fit, typ=2)
        F, p = tab.loc[name, "F"], tab.loc[name, "PR(>F)"]
        df1, df2 = tab.loc[name, "df"], tab.loc["Residual", "df"]
        print(f"   {name:<20} F({df1:.0f}, {df2:.0f}) = {F:8.2f}, p = {p:.2e}")
        rows.append({"objective": label, "effect": name, "F": F, "df1": df1,
                     "df2": df2, "p": p})


def main():
    RESULTS.mkdir(exist_ok=True)
    sys.stdout = Tee(RESULTS / "stats_report.txt")

    runs = pd.read_csv(RESULTS / "runs.csv")
    key = ["image", "objective", "k", "algorithm", "seed"]
    print("=" * 78)
    print("STATISTICAL ANALYSIS OF THE MAIN EXPERIMENT")
    print("=" * 78)
    print(f"runs in file: {len(runs):,} (expected {EXPECTED_RUNS:,}); "
          f"duplicates: {runs.duplicated(key).sum()}; "
          f"negative gaps (would mean a bug): {(runs.gap < -1e-9 * runs.opt_score.abs()).sum()}")
    if len(runs) != EXPECTED_RUNS:
        print("WARNING: the grid is incomplete - results below are provisional.")

    # ------------------------------------------------------------ 1. cells
    cells = (runs.groupby(["image", "objective", "k", "algorithm"])
             .agg(median_rel_gap=("rel_gap", "median"),
                  mean_rel_gap=("rel_gap", "mean"),
                  p95_rel_gap=("rel_gap", lambda s: s.quantile(0.95)),
                  max_rel_gap=("rel_gap", "max"),
                  hit_rate=("hit", "mean"),
                  median_time_s=("time_s", "median"))
             .reset_index())
    cells.to_csv(RESULTS / "cell_summary.csv", index=False)

    # ------------------------------------------------------------ 5. descriptive
    print("\n--- Cost of approximation (all images, k and seeds pooled DESCRIPTIVELY) ---")
    desc = (runs.groupby(["objective", "algorithm"])
            .agg(hit_rate=("hit", "mean"),
                 median_rel_gap=("rel_gap", "median"),
                 p95_rel_gap=("rel_gap", lambda s: s.quantile(0.95)),
                 max_rel_gap=("rel_gap", "max"),
                 median_time_s=("time_s", "median"))
            .reset_index())
    for obj in ("otsu", "kapur"):
        print(f"\n{obj.upper()}  {'algorithm':<14}{'hit rate':>9}{'median gap':>12}"
              f"{'95% gap':>11}{'max gap':>11}{'time':>9}")
        sub = desc[desc.objective == obj].set_index("algorithm").loc[ALGOS]
        for a, r in sub.iterrows():
            print(f"      {a:<14}{r.hit_rate:>9.1%}{r.median_rel_gap:>12.2e}"
                  f"{r.p95_rel_gap:>11.2e}{r.max_rel_gap:>11.2e}{r.median_time_s * 1000:>7.0f}ms")
    desc.to_csv(RESULTS / "stats_descriptive.csv", index=False)

    print("\nHit rate by k (share of runs that found the exact optimum):")
    hk = runs.pivot_table(index="algorithm", columns=["objective", "k"], values="hit",
                          aggfunc="mean").loc[ALGOS]
    print(hk.map(lambda v: f"{v:.0%}").to_string())
    hk.to_csv(RESULTS / "stats_hit_by_k.csv")

    # DP time per cell, for the time comparison
    dp_times = []
    for image in runs.image.unique():
        p = histogram(load_grey(image))
        for obj in ("otsu", "kapur"):
            for k in sorted(runs.k.unique()):
                dp_multilevel(p, k, obj)
                t0 = time.perf_counter()
                for _ in range(5):
                    dp_multilevel(p, k, obj)
                dp_times.append((time.perf_counter() - t0) / 5)
    dp_med = float(np.median(dp_times))
    opt_med = float(runs.time_s.median())
    (RESULTS / "stats_dp_time.txt").write_text(f"{dp_med}\n")   # reused by make_figures.py
    print(f"\nRun time: exact DP median {dp_med * 1000:.1f} ms per (image, objective, k);"
          f" metaheuristic run median {opt_med * 1000:.0f} ms "
          f"(~{opt_med / dp_med:.0f}x the DP) - measured on THIS machine.")

    # ------------------------------------------------------------ 2+3. Friedman etc.
    print("\n--- Primary analysis: per (objective, k), 8 image blocks x 7 algorithms ---")
    print("Response: per-cell MEDIAN relative gap over 30 seeds.")
    N_img = runs.image.nunique()
    min_p = 2 / 2 ** N_img
    print(f"Note: with N = {N_img} images the smallest possible two-sided exact Wilcoxon"
          f" p is 2/2^{N_img} = {min_p:.4f}; Holm over 21 pairs multiplies the smallest"
          f" by 21 -> {min(1, 21 * min_p):.3f}. Pairwise Wilcoxon-Holm therefore CANNOT"
          f" reach 0.05 with {N_img} images, by construction; Friedman/Nemenyi is the test with"
          f" power here.")
    fr_rows, wil_rows, ranks_by = [], [], {}
    for obj in ("otsu", "kapur"):
        for k in sorted(cells.k.unique()):
            sub = cells[(cells.objective == obj) & (cells.k == k)]
            mat = (sub.pivot(index="image", columns="algorithm", values="median_rel_gap")
                   [ALGOS].to_numpy())
            ranks_by[(obj, k)] = friedman_block(mat, f"{obj} k={k}", fr_rows, wil_rows)

    print("\n--- Overall ranking per objective (8 image blocks; response = mean over k"
          " of the per-cell median gap) ---")
    for obj in ("otsu", "kapur"):
        sub = cells[cells.objective == obj]
        agg = sub.groupby(["image", "algorithm"]).median_rel_gap.mean().unstack()[ALGOS]
        ranks_by[obj] = friedman_block(agg.to_numpy(), f"{obj} overall", fr_rows, wil_rows)

    tau, tau_p = kendalltau(ranks_by["otsu"], ranks_by["kapur"])
    print(f"\nDoes the ranking change between criteria? Kendall's tau between the Otsu"
          f" and Kapur average-rank orders = {tau:.2f} (p = {tau_p:.3f}, K = 7).")

    pd.DataFrame(fr_rows).to_csv(RESULTS / "stats_friedman.csv", index=False)
    pd.DataFrame(wil_rows).to_csv(RESULTS / "stats_wilcoxon.csv", index=False)

    # ------------------------------------------------------------ 4. ART
    art_rows = []
    for obj in ("otsu", "kapur"):
        art_anova(cells[cells.objective == obj], "median_rel_gap", obj, art_rows)
    pd.DataFrame(art_rows).to_csv(RESULTS / "stats_art.csv", index=False)

    # ------------------------------------------------------------ 6. fidelity
    print("\n--- Reconstruction fidelity of the returned thresholds (PSNR/SSIM) ---")
    print("Difference vs the EXACT optimum's segmentation; negative = worse fidelity.")
    cache = {}
    imgs = {name: load_grey(name) for name in runs.image.unique()}

    def fid(image, thr):
        key2 = (image, thr)
        if key2 not in cache:
            t = tuple(int(v) for v in thr.split())
            cache[key2] = quality(imgs[image], segment(imgs[image], t))
        return cache[key2]

    rows = []
    for r in runs.itertuples(index=False):
        ps, ss = fid(r.image, r.thresholds)
        po, so = fid(r.image, r.opt_thresholds)
        rows.append((r.image, r.objective, r.k, r.algorithm, ps - po, ss - so))
    fd = pd.DataFrame(rows, columns=["image", "objective", "k", "algorithm",
                                     "d_psnr", "d_ssim"])
    fsum = (fd.groupby(["objective", "algorithm"])
            .agg(median_d_psnr=("d_psnr", "median"), p05_d_psnr=("d_psnr", lambda s: s.quantile(0.05)),
                 median_d_ssim=("d_ssim", "median"), p05_d_ssim=("d_ssim", lambda s: s.quantile(0.05)))
            .reset_index())
    for obj in ("otsu", "kapur"):
        print(f"\n{obj.upper()}  {'algorithm':<14}{'median dPSNR':>13}{'5% dPSNR':>10}"
              f"{'median dSSIM':>13}{'5% dSSIM':>10}")
        for a, r in fsum[fsum.objective == obj].set_index("algorithm").loc[ALGOS].iterrows():
            print(f"      {a:<14}{r.median_d_psnr:>11.3f}dB{r.p05_d_psnr:>8.3f}dB"
                  f"{r.median_d_ssim:>13.4f}{r.p05_d_ssim:>10.4f}")
    fsum.to_csv(RESULTS / "stats_fidelity.csv", index=False)
    print(f"\n({len(cache):,} distinct segmentations evaluated)")
    print("\nSaved results/cell_summary.csv, stats_descriptive.csv, stats_hit_by_k.csv,"
          " stats_friedman.csv, stats_wilcoxon.csv, stats_art.csv, stats_fidelity.csv,"
          " stats_report.txt")


if __name__ == "__main__":
    main()
