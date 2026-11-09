"""
Extension analysis - Parts A and C, with the UNCHANGED statistical protocol.
Applied Statistics & Stochastic Processes project.

Why a separate script: analyse_stats.py writes fixed file names
(stats_*.csv, stats_report.txt) that the paper already cites; running it on
new data would overwrite them. This script IMPORTS its functions instead
(friedman_block, art_anova, nemenyi_cd, avg_ranks), so the tests, the
effect sizes and the critical difference are computed by exactly the same
code as the original study - only the input and the output names differ.

Part A  (python analyse_ext.py --part A)
  Reads results/runs.csv (8 images) + results/runs_extended.csv (16 images)
  WITHOUT modifying either, and analyses all 24 images together:
  hit-rate and gap tables, Friedman + Kendall's W + Nemenyi per (criterion,
  k) and overall, 21 Wilcoxon-Holm pairs with rank-biserial r, ART, and two
  things N = 8 could not show:
    * effective N: Wilcoxon drops pairs of images with a ZERO difference
      (both algorithms equally good, typically both at gap 0), so a pair's
      usable N can be far below 24. Reported for every test.
    * ranking stability: Kendall's tau between the 8-image and the
      24-image average-rank orders. (The 8 images are part of the 24, so
      the two rankings are not independent; tau measures stability.)
  Run times are NOT pooled across the two files: runs.csv was timed on a
  different machine. Times are reported for the 16 new images only.

Part C  (python analyse_ext.py --part C)
  Reads results/runs_tsallis.csv and results/tsallis_exhaustive.csv.
  k = 2, 3: gaps against the EXACT optimum (exhaustive search).
  k = 4, 5: gaps against the BEST KNOWN value = the best score of any of
  the 210 runs of that (image, k) cell - not proven optimal, labelled so.
  Same tests as Part A, plus run time vs exhaustive-search time.

Outputs (new names only): results/ext24_* and results/tsallis_*
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from analyse_stats import ALGOS, Tee, art_anova, avg_ranks, friedman_block, nemenyi_cd
from thresholding import BENCHMARK_ORIGINAL

RESULTS = Path("results")
KEY = ["image", "objective", "k", "algorithm", "seed"]


def cells_of(runs):
    """Per (image, objective, k, algorithm) cell: median gap over seeds etc."""
    return (runs.groupby(["image", "objective", "k", "algorithm"])
            .agg(median_rel_gap=("rel_gap", "median"),
                 p95_rel_gap=("rel_gap", lambda s: s.quantile(0.95)),
                 max_rel_gap=("rel_gap", "max"),
                 hit_rate=("hit", "mean"),
                 median_time_s=("time_s", "median"))
            .reset_index())


def effective_n(mat, wil_rows):
    """Add n_eff / n_zero / min attainable p to the 21 rows friedman_block just appended."""
    N = mat.shape[0]
    for row, (i, j) in zip(wil_rows[-21:], combinations(range(mat.shape[1]), 2)):
        d = mat[:, i] - mat[:, j]
        n_eff = int(np.count_nonzero(d))
        row.update(n_images=N, n_zero=N - n_eff, n_eff=n_eff,
                   min_exact_p=(2.0 / 2 ** n_eff) if n_eff else 1.0)


def run_protocol(cells, objectives, ks, tag, label_extra=""):
    """Friedman/Nemenyi/Wilcoxon per (objective, k) and overall; ART per objective."""
    fr_rows, wil_rows, ranks = [], [], {}
    for obj in objectives:
        for k in ks:
            sub = cells[(cells.objective == obj) & (cells.k == k)]
            mat = sub.pivot(index="image", columns="algorithm", values="median_rel_gap")[ALGOS]
            m = mat.to_numpy()
            ranks[(obj, k)] = friedman_block(m, f"{obj} k={k}{label_extra}", fr_rows, wil_rows)
            effective_n(m, wil_rows)
        agg = (cells[cells.objective == obj].groupby(["image", "algorithm"])
               .median_rel_gap.mean().unstack()[ALGOS].to_numpy())
        ranks[obj] = friedman_block(agg, f"{obj} overall{label_extra}", fr_rows, wil_rows)
        effective_n(agg, wil_rows)

    wil = pd.DataFrame(wil_rows)
    print(f"\n--- Effective N of the Wilcoxon tests ({tag}) ---")
    print("A pair of images with zero difference is dropped by the signed-rank test,")
    print("so the usable N is n_eff = N - (number of zero differences).")
    for an, sub in wil.groupby("analysis", sort=False):
        print(f"  [{an}] n_eff over the 21 pairs: min {sub.n_eff.min()}, median "
              f"{sub.n_eff.median():.0f}, max {sub.n_eff.max()} (N = {sub.n_images.iloc[0]}); "
              f"pairs with n_eff < N: {(sub.n_eff < sub.n_images).sum()}/21; "
              f"significant after Holm: {int(sub.significant.sum())}/21")
    print(f"  Overall: {(wil.n_eff < wil.n_images).sum()} of {len(wil)} tests lost at least one"
          f" image to a zero difference; {(wil.n_eff < 10).sum()} had n_eff < 10, where the"
          f" smallest attainable exact p (2/2^n_eff) is above 0.05/21, Holm's threshold for the"
          f" FIRST rejection in a family.")
    print(f"  Significant pairs (Holm, 0.05), all conditions: {int(wil.significant.sum())} of {len(wil)}")
    sig = wil[wil.significant]
    for r in sig.itertuples(index=False):
        better = r.a if r.rank_biserial < 0 else r.b
        print(f"     {r.analysis:<22s} {r.a} vs {r.b}: p_holm = {r.p_holm:.4f}, r = {r.rank_biserial:+.2f}"
              f" (smaller gaps: {better}), n_eff = {r.n_eff}")

    art_rows = []
    for obj in objectives:
        art_anova(cells[cells.objective == obj], "median_rel_gap", obj, art_rows)
    return pd.DataFrame(fr_rows), wil, pd.DataFrame(art_rows), ranks


def part_a():
    sys.stdout = Tee(RESULTS / "ext24_report.txt")
    old = pd.read_csv(RESULTS / "runs.csv")
    new = pd.read_csv(RESULTS / "runs_extended.csv")
    runs = pd.concat([old.assign(source="runs.csv"), new.assign(source="runs_extended.csv")],
                     ignore_index=True)
    print("=" * 78)
    print("EXTENSION PART A - 24 IMAGES (8 original + 16 new), UNCHANGED PROTOCOL")
    print("=" * 78)
    n_img = runs.image.nunique()
    print(f"runs: {len(old):,} (runs.csv) + {len(new):,} (runs_extended.csv) = {len(runs):,}"
          f" (expected 13,440 + 26,880 = 40,320); images: {n_img}; duplicates: "
          f"{runs.duplicated(KEY).sum()}; negative gaps: "
          f"{(runs.gap < -1e-9 * runs.opt_score.abs()).sum()}")
    if len(new) != 26880 or n_img != 24:
        print("WARNING: the extended grid is incomplete - results below are provisional.")
    q, cd = nemenyi_cd(7, n_img)
    print(f"Nemenyi critical difference for K = 7, N = {n_img}: CD = q*sqrt(K(K+1)/(6N)) = "
          f"{q:.3f} * sqrt(56/{6 * n_img}) = {cd:.3f}  (N = 8 gave 3.18)")

    cells = cells_of(runs)
    cells.to_csv(RESULTS / "ext24_cell_summary.csv", index=False)

    print("\n--- Cost of approximation, 24 images (pooled DESCRIPTIVELY over images, k, seeds) ---")
    desc = (runs.groupby(["objective", "algorithm"])
            .agg(hit_rate=("hit", "mean"), median_rel_gap=("rel_gap", "median"),
                 p95_rel_gap=("rel_gap", lambda s: s.quantile(0.95)),
                 max_rel_gap=("rel_gap", "max")).reset_index())
    t_new = new.groupby(["objective", "algorithm"]).time_s.median().rename("median_time_s_new16")
    desc = desc.merge(t_new.reset_index(), on=["objective", "algorithm"])
    for obj in ("otsu", "kapur"):
        print(f"\n{obj.upper():6s}{'algorithm':<14}{'hit rate':>9}{'median gap':>12}{'95% gap':>11}"
              f"{'max gap':>11}{'time*':>9}")
        for a, r in desc[desc.objective == obj].set_index("algorithm").loc[ALGOS].iterrows():
            print(f"      {a:<14}{r.hit_rate:>9.1%}{r.median_rel_gap:>12.2e}{r.p95_rel_gap:>11.2e}"
                  f"{r.max_rel_gap:>11.2e}{r.median_time_s_new16 * 1000:>7.0f}ms")
    print("* median run time on the 16 NEW images only (this laptop, 4 worker processes);"
          " runs.csv was timed on a different machine and is not pooled.")
    desc.to_csv(RESULTS / "ext24_descriptive.csv", index=False)

    print("\nHit rate by k, 24 images (share of runs that found the exact optimum):")
    hk = runs.pivot_table(index="algorithm", columns=["objective", "k"], values="hit",
                          aggfunc="mean").loc[ALGOS]
    print(hk.map(lambda v: f"{v:.0%}").to_string())
    hk.to_csv(RESULTS / "ext24_hit_by_k.csv")
    print("\nSame table, 8 original vs 16 new images (is the new set harder or easier?):")
    for nm, part in (("8 original", runs[runs.image.isin(BENCHMARK_ORIGINAL)]),
                     ("16 new", runs[~runs.image.isin(BENCHMARK_ORIGINAL)])):
        h = part.groupby(["objective", "algorithm"]).hit.mean().unstack(0).loc[ALGOS]
        print(f"  {nm:10s} " + "  ".join(f"{a} {h.loc[a, 'otsu']:.0%}/{h.loc[a, 'kapur']:.0%}"
                                          for a in ALGOS) + "   (Otsu/Kapur)")

    print(f"\n--- Primary analysis: per (criterion, k), {n_img} image blocks x 7 algorithms ---")
    print("Response: per-cell MEDIAN relative gap over 30 seeds (1 = smallest gap).")
    fr, wil, art, ranks24 = run_protocol(cells, ("otsu", "kapur"), sorted(cells.k.unique()),
                                         f"{n_img} images")
    fr.to_csv(RESULTS / "ext24_friedman.csv", index=False)
    wil.to_csv(RESULTS / "ext24_wilcoxon.csv", index=False)
    art.to_csv(RESULTS / "ext24_art.csv", index=False)

    print("\n--- Ranking stability: 8-image vs 24-image average ranks (Kendall's tau, K = 7) ---")
    c8 = cells[cells.image.isin(BENCHMARK_ORIGINAL)]
    tau_rows = []
    for obj in ("otsu", "kapur"):
        for k in (2, 3, 4, 5, "overall"):
            if k == "overall":
                m8 = (c8[c8.objective == obj].groupby(["image", "algorithm"]).median_rel_gap
                      .mean().unstack()[ALGOS].to_numpy())
                r24 = ranks24[obj]
            else:
                m8 = (c8[(c8.objective == obj) & (c8.k == k)]
                      .pivot(index="image", columns="algorithm", values="median_rel_gap")[ALGOS]
                      .to_numpy())
                r24 = ranks24[(obj, k)]
            r8 = avg_ranks(m8)
            tau, p = kendalltau(r8, r24)
            tau_rows.append({"objective": obj, "k": k, "tau": tau, "p": p,
                             **{f"rank8_{a}": v for a, v in zip(ALGOS, r8)},
                             **{f"rank24_{a}": v for a, v in zip(ALGOS, r24)}})
            print(f"  {obj:5s} k={str(k):7s} tau = {tau:+.2f} (p = {p:.3f});  best on 8: "
                  f"{ALGOS[int(np.argmin(r8))]}, best on 24: {ALGOS[int(np.argmin(r24))]}")
    pd.DataFrame(tau_rows).to_csv(RESULTS / "ext24_tau.csv", index=False)
    print("\nSaved results/ext24_report.txt, ext24_cell_summary.csv, ext24_descriptive.csv, "
          "ext24_hit_by_k.csv, ext24_friedman.csv, ext24_wilcoxon.csv, ext24_art.csv, ext24_tau.csv")


def part_c():
    sys.stdout = Tee(RESULTS / "tsallis_report.txt")
    runs = pd.read_csv(RESULTS / "runs_tsallis.csv")
    ex = pd.read_csv(RESULTS / "tsallis_exhaustive.csv")
    print("=" * 78)
    print("EXTENSION PART C - NON-SEPARABLE TSALLIS CRITERION (q = "
          f"{runs.q.iloc[0]}), UNCHANGED PROTOCOL")
    print("=" * 78)
    print(f"runs: {len(runs):,} (expected 20,160); images: {runs.image.nunique()}; "
          f"duplicates: {runs.duplicated(['image', 'k', 'algorithm', 'seed']).sum()}")

    # reference per (image, k): exact for k = 2, 3; best known (max over all runs) for 4, 5
    best = runs.groupby(["image", "k"]).best_score.max().rename("best_known")
    runs = runs.merge(best.reset_index(), on=["image", "k"])
    exact = runs.opt_kind == "exact"
    print(f"Exact references: {exact.sum():,} runs (k = 2, 3); best-known references: "
          f"{(~exact).sum():,} runs (k = 4, 5) - 'best known' = best score of any of the 210 runs"
          " of that (image, k) cell, NOT proven optimal.")
    beat = runs[exact & (runs.best_score > runs.opt_score + 1e-9 * runs.opt_score.abs().clip(lower=1))]
    print(f"Runs scoring ABOVE the exact optimum (would mean a bug): {len(beat)}")
    runs["ref"] = np.where(exact, runs.opt_score, runs.best_known)
    runs["ref_kind"] = np.where(exact, "exact", "best_known")
    runs["gap"] = runs.ref - runs.best_score
    runs["rel_gap"] = runs.gap / runs.ref.abs()
    runs["hit"] = (runs.gap <= 1e-9 * np.maximum(1.0, runs.ref.abs())).astype(int)
    runs["objective"] = "tsallis"
    runs.drop(columns=["curve"]).to_csv(RESULTS / "tsallis_runs_scored.csv", index=False)

    cells = cells_of(runs)
    cells.to_csv(RESULTS / "tsallis_cell_summary.csv", index=False)

    print("\nHit rate by k (k = 2, 3: found the EXACT optimum; k = 4, 5: matched the BEST KNOWN):")
    hk = runs.pivot_table(index="algorithm", columns="k", values="hit", aggfunc="mean").loc[ALGOS]
    print(hk.map(lambda v: f"{v:.0%}").to_string())
    hk.to_csv(RESULTS / "tsallis_hit_by_k.csv")
    print("\nMedian relative gap by k (same references):")
    gk = runs.pivot_table(index="algorithm", columns="k", values="rel_gap", aggfunc="median").loc[ALGOS]
    print(gk.map(lambda v: f"{v:.2e}").to_string())

    # context: Otsu / Kapur hit rates on the same 24 images, if Part A has run
    ext = RESULTS / "runs_extended.csv"
    if ext.exists():
        ok = pd.concat([pd.read_csv(RESULTS / "runs.csv"), pd.read_csv(ext)])
        ok = ok[ok.k.isin([2, 3])]
        h = ok.pivot_table(index="algorithm", columns=["objective", "k"], values="hit",
                           aggfunc="mean").loc[ALGOS]
        print("\nFor comparison, exact-hit rates of the SAME algorithms on the SAME 24 images"
              " for the separable criteria (Part A):")
        print(h.map(lambda v: f"{v:.0%}").to_string())

    print("\n--- Cost: one optimiser run vs one exhaustive solve (k = 2, 3) ---")
    print("Exhaustive search: single process; optimiser runs: 4 worker processes in parallel,"
          " so run times may be inflated by contention (which works AGAINST the optimisers here).")
    t_run = runs[runs.k.isin([2, 3])].groupby(["image", "k", "algorithm"]).time_s.median()
    tt = t_run.reset_index().merge(ex.rename(columns={"exhaustive_time_s": "ex_s"})[["image", "k", "ex_s"]],
                                   on=["image", "k"])
    tt["ratio"] = tt.ex_s / tt.time_s
    for k in sorted(ex.k.unique()):
        e = ex[ex.k == k].exhaustive_time_s
        print(f"\n  k = {k}: exhaustive solve median {e.median():.3f} s (range {e.min():.3f}-{e.max():.3f} s"
              f" over {len(e)} images)")
        for a in ALGOS:
            s = tt[(tt.k == k) & (tt.algorithm == a)]
            print(f"     {a:<14} median run {s.time_s.median() * 1000:7.1f} ms; exhaustive / run ="
                  f" {s.ratio.median():7.1f}x (median over images); run faster on "
                  f"{(s.ratio > 1).sum()}/{len(s)} images")
    tt.to_csv(RESULTS / "tsallis_time_vs_exhaustive.csv", index=False)

    print(f"\n--- Primary analysis: per k, {runs.image.nunique()} image blocks x 7 algorithms ---")
    print("Response: per-cell MEDIAN relative gap (k = 2, 3 vs exact; k = 4, 5 vs best known).")
    fr, wil, art, _ = run_protocol(cells, ("tsallis",), sorted(cells.k.unique()), "Tsallis")
    fr.to_csv(RESULTS / "tsallis_friedman.csv", index=False)
    wil.to_csv(RESULTS / "tsallis_wilcoxon.csv", index=False)
    art.to_csv(RESULTS / "tsallis_art.csv", index=False)
    print("\nNote: the ART pools k = 2, 3 (exact reference) with k = 4, 5 (best known); the"
          " algorithm effect is the robust part, the k effect partly reflects the reference.")
    print("\nSaved results/tsallis_report.txt, tsallis_runs_scored.csv, tsallis_cell_summary.csv,"
          " tsallis_hit_by_k.csv, tsallis_time_vs_exhaustive.csv, tsallis_friedman.csv,"
          " tsallis_wilcoxon.csv, tsallis_art.csv")


def main():
    ap = argparse.ArgumentParser(description="Extension analysis, Parts A and C")
    ap.add_argument("--part", choices=["A", "C"], required=True)
    args = ap.parse_args()
    part_a() if args.part == "A" else part_c()


if __name__ == "__main__":
    main()
